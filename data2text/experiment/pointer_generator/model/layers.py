"""Component/Layers of Model: Encoder, ReduceState, and Decoder."""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from experiment.pointer_generator.model import BasicModule
from experiment.pointer_generator.model.attention import Attention


# %% Encoder
class Encoder(BasicModule):
    """For encoding."""

    def __init__(self, config):
        """Initialize the model encoder with config."""
        super(Encoder, self).__init__()
        self.src_word_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, 
            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * config.hidden_dim, 2 * config.hidden_dim, bias=False)

        self.hidden_dim = config.hidden_dim
    
        self.init_params()
    
    def forward(self, enc_input, seq_lens):
        """
        n == 2 * hidden-dim
        args:
            enc_input: [batch-size, seq-len, vocab-size-index]
            seq_lens: [batch-size, ]
        rets:
            encoder_outputs: [batch-size, seq-len, 2 * hidden-dim]
            encoder_feature: [batch-size * seq-len, 2 * hidden-dim]
            hidden: [2, batch-size, hidden-dim] = h, c of [batch-size, hidden-dim]
        Notes: 'seq_lens' should be in descending order.
        """
        embedded = self.src_word_emb(enc_input)     # [batch-size, seq-len, emb-dim]

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)    # [b, l, n]
        encoder_outputs = encoder_outputs.contiguous()                        # [b, l, n]

        encoder_feature = encoder_outputs.view(-1, 2 * self.hidden_dim)       # [b*l, n]
        encoder_feature = self.fc(encoder_feature)                            # [b*l, n]

        return encoder_outputs, encoder_feature, hidden



# %% ReduceState
class ReduceState(BasicModule):

    def __init__(self, config):
        """Initialize the reduce module with config (hidden-dim)."""
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        self.reduce_c = nn.Linear(2 * config.hidden_dim, config.hidden_dim)

        self.hidden_dim = config.hidden_dim

        self.init_params()

    def forward(self, hidden):
        """
        args:
            hidden: [2, batch-size, hidden-dim]
        rets:
            hidden_reduced_h: [1, batch-size, hidden-dim]
            hidden_reduced_c: [1, batch-size, hidden-dim]
        """
        h, c = hidden     # [batch-size, seq_len, hidden-dim]
        h_in = h.transpose(0, 1).contiguous().view(-1, self.hidden_dim * 2)  
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        ret_h = hidden_reduced_h.unsqueeze(0)    # [1, batch-size, hidden-dim]
        c_in = c.transpose(0, 1).contiguous().view(-1, self.hidden_dim * 2)  
        hidden_reduced_c = F.relu(self.reduce_c(c_in))
        ret_c = hidden_reduced_c.unsqueeze(0)    # [1, batch-size, hidden-dim]
        return (ret_h, ret_c)



# %% Decoder

class Decoder(BasicModule):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.attention_network = Attention(config)

        # decoder
        self.tgt_word_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.con_fc = nn.Linear(2 * config.hidden_dim + config.emb_dim, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, 
            batch_first=True, bidirectional=False)
        
        if config.pointer_gen:
            self.p_gen_fc = nn.Linear(4 * config.hidden_dim + config.emb_dim, 1)
        
        # p_vocab
        self.fc1 = nn.Linear(3 * config.hidden_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.vocab_size)

        self.hidden_dim = config.hidden_dim
        self.pointer_gen = config.pointer_gen

        self.init_params()
    
    def forward(
        self, y_t, s_t, 
        enc_out, enc_fea, enc_padding_mask, 
        c_t, extra_zeros, enc_batch_extend_vocab, coverage, step    
    ):  
        """
        args:
            y_t: [batch-size, vocab-size-index]
            s_t: h & c states, ([batch-size, hidden-dim], [batch-size, hidden-dim])
            enc_out: [batch-size, seq-len, 2 * hidden-dim]
            enc_fea: [batch-size * seq-len, 2 * hidden-dim]
            enc_padding_mask: [batch-size, seq-len]
            c_t: [batch-size, 2 * hidden-dim]
            extra_zeros: 
            enc_batch_extend_vocab: 
            coverage: [batch-size, seq-len]
            step: int
        rets:
            c_t: [batch-size, 2 * hidden-dim]
            attn_dist: output of attention-network, [batch-size, seq-len]
            p_gen: geneation of pointer-network, [batch-size, 1]
            coverage: coverage over the input words, [batch-size, seq-len]
        """
        if (not self.training) and (step == 0):
            dec_h, dec_c = s_t     # [batch-size, hidden-dim]
            s_t_hat = torch.cat(
                tensors=(
                    dec_h.view(-1, self.hidden_dim),
                    dec_c.view(-1, self.hidden_dim)
                ), 
                dim=1
            )                      # [batch-size, 2 * hidden-dim]
            c_t, _, coverage_next = self.attention_network(
                s_t_hat, enc_out, enc_fea, enc_padding_mask, coverage)
            coverage = coverage_next    # [batch-size, seq-len]
        
        y_t_embed = self.tgt_word_emb(y_t)    # [b, emb-dim]  [16, 128]
        x = self.con_fc( torch.cat((c_t, y_t_embed), dim=1) )    # [b,2*hid+emb] >> [b,emb]
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t)

        dec_h, dec_c = s_t   # [b, hid]
        s_t_hat = torch.cat(
            tensors=(
                dec_h.view(-1, self.hidden_dim),
                dec_c.view(-1, self.hidden_dim)
            ), 
            dim=1
        )   # [b,2*hid]
        c_t, attn_dist, coverage_next = self.attention_network(
            s_t_hat, enc_out, enc_fea, enc_padding_mask, coverage)
        
        if self.training or (step > 0):
            coverage = coverage_next
        
        p_gen = None
        if self.pointer_gen:
            p_gen_inp = torch.cat((c_t, s_t_hat, x), 1)  # [b, 2*hid+2*hid+emb]
            p_gen = self.p_gen_fc(p_gen_inp)             # [b, 1]
            p_gen = torch.sigmoid(p_gen)
        
        output = torch.cat(
            tensors=(lstm_out.view(-1, self.hidden_dim), c_t), 
            dim=1
        )   # [b, hid+2*hid]
        output = self.fc1(output)   # [b, hid]
        # output = F.relu(output)

        output = self.fc2(output)   # [b, vocab-size]
        vocab_dist = F.softmax(output, dim=1)

        if self.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=1)
            
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist
        
        return final_dist, s_t, c_t, attn_dist, p_gen, coverage