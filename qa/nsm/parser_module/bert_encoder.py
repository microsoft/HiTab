import json
import sys
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from transformers.models.bert.tokenization_bert import BertTokenizer
from qa.table_bert.table_bert import TableBertModel
from torch import nn as nn

from qa.nsm.parser_module.encoder import EncoderBase
from qa.nsm.parser_module.table_bert_helper import get_table_bert_model, get_table_bert_input_from_context

Example = namedtuple('Example', ['question', 'table'])


class BertEncoder(EncoderBase):
    def __init__(
        self,
        bert_model,
        output_size: int,
        config: Dict,
        question_feat_size: int,
        builtin_func_num: int,
        memory_size: int,
        dropout: float = 0.
    ):
        EncoderBase.__init__(self, output_size, builtin_func_num, memory_size)

        self.config = config
        self.bert_model = bert_model
        self.question_feat_size = question_feat_size  # 1
        self.dropout = nn.Dropout(dropout)  # 0.2
        self.max_variable_num_on_memory = memory_size - builtin_func_num
        self.bert_output_project = nn.Linear(
            self.bert_model.output_size + self.question_feat_size,  # 768 + ?
            self.output_size, bias=False  # 200
        )

        self.question_encoding_att_value_to_key = nn.Linear(
            self.output_size,
            self.output_size, bias=False
        )

        self.bert_table_output_project = nn.Linear(
            self.bert_model.output_size, self.output_size, bias=False
        )

        self.constant_value_embedding_linear = lambda x: x

        self.init_weights()

    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.bert_model.bert_config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        modules = [
            module
            for name, module
            in self._modules.items()
            if module and 'bert_model' not in name
        ]

        for module in modules:
            module.apply(_init_weights)

    @classmethod
    def build(cls, config, bert_model=None, master=None):  # build from agent
        if bert_model is None:
            bert_model = get_table_bert_model(
                config,
            )

        return cls(
            bert_model,
            output_size=config['en_embedding_size'],
            question_feat_size=config['n_en_input_features'],
            builtin_func_num=config['builtin_func_num'],
            memory_size=config['memory_size'],
            dropout=config['dropout'],
            config=config
        )

    def example_list_to_batch(self, env_context: List[Dict]) -> Dict:
        batch_dict = dict()
        for key in ('constant_spans', 'question_features'):
            val_list = [x[key] for x in env_context]

            # (batch_size, max_entry_num, entry_dim)
            if key == 'question_features':
                max_entry_num = max(len(val) for val in val_list)  # max question tokens
                dtype = np.float32
            else:
                max_entry_num = self.max_variable_num_on_memory  # 100
                dtype = np.int64

            entry_dim = len(val_list[0][0])  # content spans: len((-1, -1)) = 2
            batch_size = len(env_context)

            batch_value_tensor = np.zeros((batch_size, max_entry_num, entry_dim), dtype=dtype)

            if key == 'constant_spans':
                batch_value_tensor.fill(-1.)

            for i, val in enumerate(val_list):
                entry_num = len(val)
                batch_value_tensor[i, :entry_num] = val

            batch_dict[key] = torch.from_numpy(batch_value_tensor).to(next(self.parameters()).device)

        return batch_dict

    def bert_encode(self, env_context: List[Dict]) -> Any:
        contexts, tables = get_table_bert_input_from_context(
            env_context
        )
        question_encoding, header_encoding, index_name_encoding, info = self.bert_model.encode(
            contexts, tables
        )

        table_bert_encoding = {
            'question_encoding': question_encoding,
            'header_encoding': header_encoding,
            'index_name_encoding': index_name_encoding,
            'input_tables': tables,
        }

        table_bert_encoding.update(info['tensor_dict'])

        return table_bert_encoding

    def encode(self, env_context: List[Dict]):
        batch_size = len(env_context)
        batched_context = self.example_list_to_batch(env_context)  # context_spans/question_features: (b, max_entry_num, entry_dim)
        table_bert_encoding = self.bert_encode(env_context)

        # remove leading [CLS] symbol
        question_encoding = table_bert_encoding['question_encoding'][:, 1:]
        question_mask = table_bert_encoding['context_token_mask'][:, 1:]
        cls_encoding = table_bert_encoding['question_encoding'][:, 0]

        if self.question_feat_size > 0:  # concat question feat and bert question encoding
            question_encoding = torch.cat([
                question_encoding,
                batched_context['question_features']],
                dim=-1)

        question_encoding = self.bert_output_project(question_encoding)
        question_encoding_att_linear = self.question_encoding_att_value_to_key(question_encoding)

        context_encoding = {
            'batch_size': batch_size,
            'question_encoding': question_encoding,
            'question_mask': question_mask,
            'question_encoding_att_linear': question_encoding_att_linear,
        }

        # (batch_size, max_header_num/max_level_num, encoding_size)
        header_encoding = table_bert_encoding['header_encoding']
        index_name_encoding = table_bert_encoding['index_name_encoding']

        # concat header encoding and index_name_encoding
        constant_value_num = batched_context['constant_spans'].size(1)
        constant_value_embedding = torch.zeros((batch_size, constant_value_num, header_encoding.size(-1)), dtype=torch.float32).to(next(self.parameters()).device)
        constant_value_mask = torch.zeros((batch_size, constant_value_num), dtype=torch.int64).to(next(self.parameters()).device)
        for e_id, context in enumerate(env_context):
            valid_header_num = min(len(env_context[e_id]['table'].header2id), constant_value_num)
            constant_value_embedding[e_id, :valid_header_num, :] \
                = header_encoding[e_id, :valid_header_num, :]
            valid_index_name_num = min(len(env_context[e_id]['table'].index_name2id), constant_value_num - valid_header_num)
            constant_value_embedding[e_id, valid_header_num: valid_header_num + valid_index_name_num, :] \
                = index_name_encoding[e_id, :valid_index_name_num, :]
            constant_value_mask[e_id, :valid_header_num + valid_index_name_num] = 1

        constant_value_embedding = self.bert_table_output_project(constant_value_embedding)

        constant_encoding, constant_mask = self.get_constant_encoding(
            question_encoding, batched_context['constant_spans'], constant_value_embedding, constant_value_mask)

        context_encoding.update({
            'cls_encoding': cls_encoding,
            'table_bert_encoding': table_bert_encoding,
            'constant_encoding': constant_encoding,
            'constant_mask': constant_mask
        })

        return context_encoding

    def get_constant_encoding(self, question_token_encoding, constant_span, constant_value_embedding, constant_value_mask):
        """
        Args:
            question_token_encoding: (batch_size, max_question_len, encoding_size)
            constant_span: (batch_size, mem_size, 2)
                This is the indices of span entities identified in the input question
            constant_value_embedding: (batch_size, constant_value_num, embed_size)
                This is the encodings of table columns only
                Encodings of entity spans will be computed later using the encodings of questions
            column_mask: (batch_size, constant_value_num)
        """
        # (batch_size, mem_size)
        constant_span_mask = torch.ge(constant_span, 0)[:, :, 0].float()

        # mask out entries <= 0
        constant_span = constant_span * constant_span_mask.unsqueeze(-1).long()

        constant_span_size = constant_span.size()
        mem_size = constant_span_size[1]
        batch_size = question_token_encoding.size(0)

        # (batch_size, mem_size, 2, embed_size)
        constant_span_embedding = torch.gather(
            question_token_encoding.unsqueeze(1).expand(-1, mem_size, -1, -1),
            index=constant_span.unsqueeze(-1).expand(-1, -1, -1, question_token_encoding.size(-1)),
            dim=2  # over `max_question_len`
        )

        # (batch_size, mem_size, embed_size)
        constant_span_embedding = torch.mean(constant_span_embedding, dim=-2)
        constant_span_embedding = constant_span_embedding * constant_span_mask.unsqueeze(-1)

        # `constant_value_embedding` consists mostly of table header embedding computed by table BERT
        # (batch_size, constant_value_num, embed_size)
        constant_value_embedding = self.constant_value_embedding_linear(constant_value_embedding)

        constant_encoding = constant_value_embedding + constant_span_embedding
        constant_mask = (constant_span_mask.byte() | constant_value_mask.byte()).float()

        return constant_encoding, constant_mask
