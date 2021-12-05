from typing import List, Dict

import torch
from torch import nn as nn


class EncoderBase(nn.Module):
    def __init__(self,
                 output_size: int,
                 builtin_func_num: int,
                 memory_size: int):
        nn.Module.__init__(self)

        self.output_size = output_size
        self.builtin_func_num = builtin_func_num
        self.memory_size = memory_size

    def encode(self, examples: List) -> Dict:
        raise NotImplementedError


ContextEncoding = Dict[str, torch.Tensor]
COLUMN_TYPES = ['string', 'date', 'number', 'num1', 'num2']