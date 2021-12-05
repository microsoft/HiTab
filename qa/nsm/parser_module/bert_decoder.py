import itertools
from typing import Union, List, Dict, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F

from qa.nsm import nn_util
from qa.nsm.parser_module.encoder import ContextEncoding
from qa.nsm.env import Observation
from qa.nsm.parser_module.bert_encoder import BertEncoder
from qa.nsm.parser_module.decoder import DecoderBase, MultiLayerDropoutLSTMCell, DecoderState


class BertDecoder(DecoderBase):
    def __init__(
        self,
        mem_item_embed_size: int,
        constant_value_embed_size: int,
        encoder_output_size: int,
        hidden_size: int,
        num_layers: int,
        output_feature_num: int,
        builtin_func_num: int,
        memory_size: int,
        encoder: BertEncoder,
        dropout=0.,
        **kwargs
    ):
        DecoderBase.__init__(
            self,
            memory_size,
            mem_item_embed_size,
            constant_value_embed_size,
            builtin_func_num,
            encoder.output_size
        )

        self.hidden_size = hidden_size
        self.decoder_cell_init_linear = nn.Linear(
            encoder.bert_model.bert_config.hidden_size,
            hidden_size)

        self.rnn_cell = MultiLayerDropoutLSTMCell(
            mem_item_embed_size, hidden_size,
            num_layers=num_layers, dropout=dropout)

        self.att_vec_linear = nn.Linear(encoder_output_size + hidden_size, hidden_size, bias=False)

        self.attention_func = self.dot_prod_attention

        # self.constant_value_embedding_linear = nn.Linear(constant_value_embed_size, mem_item_embed_size)
        self.constant_value_embedding_linear = lambda x: x

        # (builtin_func_num, embed_size)
        self.builtin_func_embeddings = nn.Embedding(builtin_func_num, mem_item_embed_size)

        self.output_feature_num = output_feature_num
        if output_feature_num > 0:
            self.output_feature_linear = nn.Linear(output_feature_num, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(_init_weights)

        if self.output_feature_num > 0:
            self.output_feature_linear.weight.data.zero_()

        # set forget gate bias to 1, as in tensorflow
        for name, p in itertools.chain(self.rnn_cell.named_parameters()):
            if 'bias' in name:
                n = p.size(0)
                forget_start_idx, forget_end_idx = n // 4, n // 2
                p.data[forget_start_idx:forget_end_idx].fill_(1.)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def build(cls, config, encoder: BertEncoder, master: str = None) -> 'BertDecoder':
        return cls(
            mem_item_embed_size=config['value_embedding_size'],
            constant_value_embed_size=config['value_embedding_size'],
            encoder_output_size=config['en_embedding_size'],
            hidden_size=config['en_embedding_size'],
            num_layers=config['n_layers'],
            output_feature_num=config['n_de_output_features'],
            builtin_func_num=config['builtin_func_num'],
            memory_size=config['memory_size'],
            encoder=encoder,
            dropout=config['dropout'],
        )

    def get_lstm_init_state(self, context_encoding: ContextEncoding):
        # use encoding of the [CLS] token to initialize the decoder
        question_repr = context_encoding['cls_encoding']

        sc_0_i = self.decoder_cell_init_linear(question_repr)
        sh_0_i = torch.tanh(sc_0_i)

        decoder_init_states = [(sh_0_i, sc_0_i)] * self.rnn_cell.num_layers

        return decoder_init_states

    def get_initial_state(self, context_encoding: ContextEncoding):
        # prepare decoder's initial memory and internal LSTM state

        initial_memory = self.get_initial_memory(context_encoding)  # init embeddings of memory

        decoder_init_states = self.get_lstm_init_state(context_encoding)  # decoder input [CLS] embedding

        state = DecoderState(state=decoder_init_states, memory=initial_memory)

        return state

    def get_initial_memory(self, context_encoding: ContextEncoding):
        constant_encoding = context_encoding['constant_encoding']

        # add built-in functional operator embeddings
        # (batch_size, builtin_func_num, embed_size)
        batch_size = constant_encoding.size(0)
        builtin_func_embedding = self.builtin_func_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # (batch_size, builtin_func_num + mem_size, embed_size)
        initial_memory = torch.cat(  # initial：random memory emb + random func emb + constant bert encoding
            [builtin_func_embedding, constant_encoding],
            dim=1
        )[:, :self.memory_size]

        return initial_memory

    def step(self, x: Union[List[Observation], Observation], state_tm1: DecoderState, context_encoding: Dict):
        """Perform one step of the decoder"""

        # first convert listed input to batched ones
        if isinstance(x, list):
            x = Observation.to_batched_input(x, memory_size=self.memory_size).to(self.device)

        batch_size = x.read_ind.size(0)

        # collect y_tm1 as inputs to inner rnn cells
        # Memory: (batch_size, mem_size, mem_value_dim)
        # (batch_size, mem_value_dim)
        input_mem_entry = state_tm1.memory[torch.arange(batch_size, device=self.device), x.read_ind]

        # (batch_size, hidden_size)
        inner_output_t, inner_state_t = self.rnn_cell(input_mem_entry, state_tm1.state)

        # attention over context
        ctx_t, alpha_t = self.attention_func(query=inner_output_t,
                                             keys=context_encoding['question_encoding_att_linear'],
                                             values=context_encoding['question_encoding'],
                                             entry_masks=context_encoding['question_mask'])

        # (batch_size, hidden_size)
        att_t = torch.tanh(self.att_vec_linear(torch.cat([inner_output_t, ctx_t], 1)))  # E.q. (5)
        # att_t = self.dropout(att_t)

        # compute scores over valid memory entries
        # memory is organized by:
        # [built-in functions, constants and variables]

        # dot product attention
        # (batch_size, mem_size)
        mem_logits = torch.matmul(state_tm1.memory, att_t.unsqueeze(-1)).squeeze(-1)

        # add output features to logits
        # (batch_size, mem_size)
        if self.output_feature_num:
            output_feature = self.output_feature_linear(x.output_features).squeeze(-1)
            mem_logits = mem_logits + output_feature

        # write head of shape (batch_size)
        # mask of shape (batch_size)
        write_mask = torch.ge(x.write_ind, 0).float()
        # mask out negative entries in write_ind
        write_ind = x.write_ind * write_mask.long()
        # (batch_size, hidden_size)
        write_value = att_t * write_mask.unsqueeze(-1)

        # write to memory
        memory_tm1 = state_tm1.memory
        memory_t = memory_tm1.scatter_add(1, write_ind.view(-1, 1, 1).expand(-1, -1, memory_tm1.size(-1)), write_value.unsqueeze(1))

        state_t = DecoderState(state=inner_state_t, memory=memory_t)

        return mem_logits, state_t

    def step_and_get_action_scores_t(self, observations_t, state_tm1, context_encoding):
        mem_logits, state_t = self.step(observations_t, state_tm1, context_encoding=context_encoding)

        # (batch_size, mem_size)
        action_score_t = nn_util.masked_log_softmax(mem_logits, mask=observations_t.valid_action_mask)

        return action_score_t, state_t

    def dot_prod_attention(self,
                           query: torch.Tensor,
                           keys: torch.Tensor,
                           values: torch.Tensor,
                           entry_masks: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, src_sent_len)
        att_weight = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)

        if entry_masks is not None:
            att_weight.data.masked_fill_((1.0 - entry_masks).bool(), -float('inf'))

        att_prob = F.softmax(att_weight, dim=-1)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_prob.view(*att_view), values).squeeze(1)

        return ctx_vec, att_prob