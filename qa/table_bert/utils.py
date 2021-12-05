#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum


class TransformerVersion(Enum):
    PYTORCH_PRETRAINED_BERT = 0
    TRANSFORMERS = 1

from transformers import BertTokenizer    # noqa
from transformers.models.bert.modeling_bert import (    # noqa
    BertForMaskedLM, BertForPreTraining, BertModel,
    BertSelfOutput, BertIntermediate, BertOutput,
    BertLMPredictionHead, # BertLayerNorm, gelu
)
from transformers import BertConfig  # noqa

hf_flag = 'new'
TRANSFORMER_VERSION = TransformerVersion.TRANSFORMERS
