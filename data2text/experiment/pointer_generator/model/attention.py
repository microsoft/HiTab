"""Attention Module."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment.pointer_generator.model import BasicModule


class Attention(BasicModule):
    """Encoder-Decoder Attention Module."""

    def __init__(self, config):
        """Initialize with config (hidden-dim)."""
        super(Attention, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.is_coverage = config.is_coverage

        self.fc = nn.Linear(2 * config.hidden_dim, 1, bias=False)
        self.dec_fc = nn.Linear(2 * config.hidden_dim, 2 * config.hidden_dim)
        if self.is_coverage:
            self.con_fc = nn.Linear(1, 2 * config.hidden_dim, bias=False)
        
    def forward(self, s_t, enc_out, enc_fea, enc_padding_mask, coverage):
        """
        b = batch-size, l == seq-len, n == 2 * hidden_dim

        args:
            s_t: [batch-size, 2 * hidden-dim]
            enc_out: [batch-size, seq-len, 2 * hidden-dim]
            enc_fea: [batch-size * seq-len, 2 * hidden-dim]
            enc_padding_mask: [batch-size, seq-len]
            coverage: [batch-size, seq-len]
        rets:
            c_t: [batch-size, 2 * hidden-dim]
            attn_dist: [batch-size, seq-len]
            coverage: [batch-size, seq-len]
        """
        b, l, n = list(enc_out.size())

        dec_fea = self.dec_fc(s_t)    # [b, 2*hid]
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b,l,n).contiguous()   # [b,l,2*hid]
        dec_fea_expanded = dec_fea_expanded.view(-1, n)                      # [b*l,2*hid]

        att_features = enc_fea + dec_fea_expanded   # [b*l,2*hid]
        if self.is_coverage:
            coverage_inp = coverage.view(-1, 1)          # [b*l, 1]
            coverage_fea = self.con_fc(coverage_inp)     # [b*l,2*hid]
            att_features = att_features + coverage_fea   # [b*l,2*hid]
        
        e = torch.tanh(att_features)                     # [b*l,2*hid]
        scores = self.fc(e)                              # [b*l,1]
        scores = scores.view(-1, l)                      # [b,l]

        attn_dist_ = F.softmax(scores, dim=-1) * enc_padding_mask   # [b,l]
        normalization_factor_ = attn_dist_.sum(1, keepdim=True)     # [b]
        attn_dist = attn_dist_ / (normalization_factor_ + 1e-6)     # [b,l]
        
        attn_dist = attn_dist.unsqueeze(1)               # [b,1,l]
        c_t = torch.bmm(attn_dist, enc_out)              # [b,1,n]
        c_t = c_t.view(-1, 2 * self.hidden_dim)          # [b,2*hid]

        attn_dist = attn_dist.view(-1, l)                # [b, l]

        if self.is_coverage:
            coverage = coverage.view(-1, l)
            coverage = coverage + attn_dist
        
        return c_t, attn_dist, coverage