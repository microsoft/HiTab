import collections

from torch import nn as nn


class DecoderBase(nn.Module):
    def __init__(self,
                 memory_size,
                 mem_item_embed_size,
                 constant_value_embed_size,
                 builtin_func_num,
                 encoder_output_size):
        nn.Module.__init__(self)

        self.memory_size = memory_size
        self.mem_item_embed_size = mem_item_embed_size
        self.constant_value_embed_size = constant_value_embed_size
        self.builtin_func_num = builtin_func_num
        self.encoder_output_size = encoder_output_size

    def step(self, *args, **kwargs):
        raise NotImplementedError


Hypothesis = collections.namedtuple('Hypothesis', ['env', 'score'])


class MultiLayerDropoutLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0., use_skip_connection=True):
        super(MultiLayerDropoutLSTMCell, self).__init__()

        cells = []
        for i in range(num_layers):
            x_dim = input_size if i == 0 else hidden_size
            cell = nn.LSTMCell(x_dim, hidden_size)
            cells.append(cell)

        self.num_layers = num_layers
        self.cell_list = nn.ModuleList(cells)
        self.dropout = nn.Dropout(dropout)
        self.use_skip_connection = use_skip_connection

    def forward(self, x, s_tm1):
        # x: (batch_size, input_size)
        o_i = None
        state = []
        for i in range(self.num_layers):
            h_i, c_i = self.cell_list[i](x, s_tm1[i])

            if i > 0 and self.use_skip_connection:
                o_i = h_i + x
            else:
                o_i = h_i

            o_i = self.dropout(o_i)

            s_i = (h_i, c_i)
            state.append(s_i)

            x = o_i

        return o_i, state


class DecoderState(object):
    def __init__(self, state, memory):
        self.state = state
        self.memory = memory

    def __getitem__(self, indices):
        sliced_state = [(s[0][indices], s[1][indices]) for s in self.state]
        sliced_memory = self.memory[indices]

        return DecoderState(sliced_state, sliced_memory)