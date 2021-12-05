#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import sys
from typing import List, Any, Tuple, Dict
import numpy as np
from fairseq import distributed_utils
from tqdm import tqdm
import json

import torch
from torch.nn import CrossEntropyLoss
from torch_scatter import scatter_max, scatter_mean

from qa.table_bert.utils import BertForPreTraining, BertForMaskedLM, TRANSFORMER_VERSION, TransformerVersion
from qa.table_bert.table_bert import TableBertModel
from qa.table_bert.config import TableBertConfig, BERT_CONFIGS
from qa.table_bert.input_formatter import VanillaTableBertInputFormatter
from qa.table_bert.hm_table import *


class VanillaTableBert(TableBertModel):
    CONFIG_CLASS = TableBertConfig

    def __init__(
        self,
        config: TableBertConfig,
        **kwargs
    ):
        super(VanillaTableBert, self).__init__(config, **kwargs)

        self._bert_model = BertForMaskedLM.from_pretrained(config.base_model_name)

        self.input_formatter = VanillaTableBertInputFormatter(self.config, self.tokenizer)

    def encode_context_and_table(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_token_indices: torch.Tensor,
        context_token_mask: torch.Tensor,
        direction_token_indices: torch.Tensor,
        level_token_indices: torch.Tensor,
        level_mask: torch.Tensor,
        index_name_token_indices: torch.Tensor,
        index_name_mask: torch.Tensor,
        header_token_indices: torch.Tensor,
        header_mask: torch.Tensor,
        return_bert_encoding: bool = False,
        **kwargs
    ):

        kwargs = (
            {}
            if TRANSFORMER_VERSION == TransformerVersion.TRANSFORMERS
            else {'output_all_encoded_layers': False}
        )

        sequence_output = self.bert(  # bert embedding: (b, max_seq_len, encoding_size)
            input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask,
            **kwargs
        ).last_hidden_state

        header_encoding = self.get_header_representation(
            sequence_output,
            header_token_indices,
            header_mask,
        )

        index_name_encoding = self.get_index_name_representation(
            sequence_output,
            level_token_indices,
            level_mask,
            index_name_token_indices,
            index_name_mask
        )

        context_encoding = torch.gather(  # context embedding
            sequence_output,
            dim=1,
            index=context_token_indices.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1)),
        )
        context_encoding = context_encoding * context_token_mask.unsqueeze(-1)

        encoding_info = {}
        if return_bert_encoding:
            encoding_info['bert_encoding'] = sequence_output

        return context_encoding, header_encoding, index_name_encoding, encoding_info

    @staticmethod
    def get_index_name_representation(
            flattened_index_name_encoding: torch.Tensor,
            level_token_indices: torch.Tensor,
            level_mask: torch.Tensor,
            index_name_token_indices: torch.Tensor,
            index_name_mask: torch.Tensor,
            encode_field: str = 'level',
            aggregator: str = 'mean_pool'
    ):
        """ Aggregate encoding of each index name according to index_name_indices/index_name_mask."""
        if encode_field == 'level':
            token_indices = level_token_indices
            mask = level_mask
        elif encode_field == 'index_name':
            token_indices = index_name_token_indices
            mask = index_name_mask
        else:
            raise ValueError(f"Unknown encode_field. select from ['level'|'index_name']")

        if aggregator.startswith('max_pool'):
            agg_func = scatter_max
            flattened_index_name_encoding[mask == 0] = float('-inf')
        elif aggregator.startswith('mean_pool') or aggregator.startswith('first_token'):
            agg_func = scatter_mean
        else:
            raise ValueError(f'Unknown index name representation method {aggregator}')

        max_level_num = mask.size(-1)
        # column_token_to_column_id: (batch_size, max_column_num)
        # (batch_size, max_column_size + 1, encoding_size)
        result = agg_func(flattened_index_name_encoding,  # src
                          token_indices.unsqueeze(-1).expand(-1, -1, flattened_index_name_encoding.size(-1)),  # idx
                          dim=1,
                          dim_size=max_level_num + 1)

        # remove the last "garbage collection" entry, mask out padding columns
        result = result[:, :-1] * mask.unsqueeze(-1)  # (b, col_num, enc)=(b, 8, 768)

        if aggregator == 'max_pool':  # why the first?
            header_encoding = result[0]
        else:
            header_encoding = result

        return header_encoding

    @staticmethod
    def get_header_representation(
        flattened_header_encoding: torch.Tensor,
        header_token_indices: torch.Tensor,
        header_mask: torch.Tensor,
        aggregator: str = 'mean_pool'
    ):
        """ Aggregate encoding of each header according to header_token_indices/header_mask."""
        if aggregator.startswith('max_pool'):
            agg_func = scatter_max
            flattened_header_encoding[header_mask == 0] = float('-inf')
        elif aggregator.startswith('mean_pool') or aggregator.startswith('first_token'):
            agg_func = scatter_mean
        else:
            raise ValueError(f'Unknown header representation method {aggregator}')

        max_header_num = header_mask.size(-1)
        # column_token_to_column_id: (batch_size, max_column_num)
        # (batch_size, max_column_size + 1, encoding_size)
        try:
            minv, maxv = torch.min(header_token_indices).detach().cpu(), torch.max(header_token_indices).detach().cpu()
            result = agg_func(flattened_header_encoding,  # src
                              header_token_indices.unsqueeze(-1).expand(-1, -1, flattened_header_encoding.size(-1)),  # idx
                              dim=1,
                              dim_size=max_header_num + 1)
            # print("normal: ", minv, maxv, header_mask.size())
        except:
            print(flattened_header_encoding.size())
            print(minv, maxv)
            print(header_token_indices.size())
            print(header_mask.size())

        # remove the last "garbage collection" entry, mask out padding columns
        result = result[:, :-1] * header_mask.unsqueeze(-1)  # (b, col_num, enc)=(b, 8, 768)

        if aggregator == 'max_pool':  # why the first?
            header_encoding = result[0]  # FIXME: bug
        else:
            header_encoding = result

        return header_encoding

    def to_tensor_dict(self, contexts: List[List[str]], tables: [List[HMTable]], table_specific_tensors=True):
        """ HMT convert header info to bert input tensors."""
        instances = []
        for e_id, (context, table) in enumerate(zip(contexts, tables)):
            instance = self.input_formatter.get_input(context, table)
            instances.append(instance)

        batch_size = len(contexts)
        max_sequence_len = max(len(x['tokens']) for x in instances)

        # basic tensors
        input_array = np.zeros((batch_size, max_sequence_len), dtype=np.int)
        mask_array = np.zeros((batch_size, max_sequence_len), dtype=np.bool)
        segment_array = np.zeros((batch_size, max_sequence_len), dtype=np.bool)

        # table specific tensors
        if table_specific_tensors:
            # max_column_num = max(len(x['column_spans']) for x in instances)
            max_context_len = max(x['context_length'] for x in instances)
            max_level_num = max(len(table.index_name2id) for table in tables)
            max_header_num = max(len(table.header2id) for table in tables)

            context_token_indices = np.zeros((batch_size, max_context_len), dtype=np.int)
            context_mask = np.zeros((batch_size, max_context_len), dtype=np.bool)
            direction_token_indices = np.zeros((batch_size, max_sequence_len), dtype=np.int)
            level_token_indices = np.zeros((batch_size, max_sequence_len), dtype=np.int)
            level_token_indices.fill(max_level_num)
            level_mask = np.zeros((batch_size, max_level_num), dtype=np.int)
            index_name_token_indices = np.zeros((batch_size, max_sequence_len), dtype=np.int)
            index_name_token_indices.fill(max_level_num)
            index_name_mask = np.zeros((batch_size, max_level_num), dtype=np.bool)
            header_token_indices = np.zeros((batch_size, max_sequence_len), dtype=np.int)
            header_token_indices.fill(max_header_num)
            header_mask = np.zeros((batch_size, max_header_num), dtype=np.bool)  # hmt differs in header num, thus differ in dim of header_encoding

        for i, instance in enumerate(instances):
            token_ids = self.tokenizer.convert_tokens_to_ids(instance['tokens'])

            input_array[i, :len(token_ids)] = token_ids
            segment_array[i, instance['segment_a_length']: len(token_ids)] = 1
            mask_array[i, :len(token_ids)] = 1.

            if table_specific_tensors:
                context_token_indices[i, :instance['context_length']] = list(range(*instance['context_span'])) #instance['context_token_indices']
                context_mask[i, :instance['context_length']] = 1.

                direction_token_indices[i, list(range(*instance['left_spans']['whole_span']))] = 1.
                direction_token_indices[i, list(range(*instance['top_spans']['whole_span']))] = 2.
                level_mask[i, :len(tables[i].index_name2id)] = 1
                index_name_mask[i, :len(tables[i].index_name2id)] = 1
                header_mask[i, :len(tables[i].header2id)] = 1
                # print("max header num:" , max_header_num)
                for level_span in instance['left_spans']['level_spans']:
                    level_token_indices[i, list(range(*level_span['whole_span']))] = level_span['level_id']
                    index_name_token_indices[i, list(range(*level_span['index_name_span']))] = level_span['level_id']
                    for header_span in level_span['header_spans']:
                        header_token_indices[i, list(range(*header_span['whole_span']))] = header_span['header_id']
                        # print(header_span['header_id'])
                for level_span in instance['top_spans']['level_spans']:
                    level_token_indices[i, list(range(*level_span['whole_span']))] = level_span['level_id']
                    index_name_token_indices[i, list(range(*level_span['index_name_span']))] = level_span['level_id']
                    for header_span in level_span['header_spans']:
                        header_token_indices[i, list(range(*header_span['whole_span']))] = header_span['header_id']
                        # print(header_span['header_id'])

                # print(f"tokens => {instance['tokens']}\n")
                # print(f"direction_token_indices => {direction_token_indices[i]}")
                # print(f"level_token_indices => {level_token_indices[i]}")
                # print(f"table index_name2id => {tables[i].index_name2id}")
                # print(f"index_name_token_indices => {index_name_token_indices[i]}")
                # print(f"index_name_mask => {index_name_mask[i]}")
                # print(f"table header2id => {tables[i].header2id}")
                # print(f"header_token_indices => {header_token_indices[i]}")
                # print(f"header_mask => {header_mask[i]}")

        tensor_dict = {
            'input_ids': torch.tensor(input_array.astype(np.int64)),
            'segment_ids': torch.tensor(segment_array.astype(np.int64)),
            'attention_mask': torch.tensor(mask_array, dtype=torch.float32),
        }

        if table_specific_tensors:
            tensor_dict.update({
                'context_token_indices': torch.tensor(context_token_indices.astype(np.int64)),
                'context_token_mask': torch.tensor(context_mask, dtype=torch.float32),
                'direction_token_indices': torch.tensor(direction_token_indices, dtype=torch.int64),
                'level_token_indices': torch.tensor(level_token_indices, dtype=torch.int64),
                'level_mask': torch.tensor(level_mask, dtype=torch.float32),
                'index_name_token_indices': torch.tensor(index_name_token_indices, dtype=torch.int64),
                'index_name_mask': torch.tensor(index_name_mask, dtype=torch.float32),
                'header_token_indices': torch.tensor(header_token_indices),
                'header_mask': torch.tensor(header_mask)
            })

        return tensor_dict, instances

    def encode(
        self,
        contexts: List[List[str]],
        tables: List,
        return_bert_encoding: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:

        # tensor_dict: bert的输入，包括input_ids,segment_ids,attention_mask,以及一些特别的context/column mask，但不包括data。
        # instances: table的格式化处理，标注出context/column(header) span的在input_ids中的具体切片位置，但不包括data
        tensor_dict, instances = self.to_tensor_dict(contexts, tables)
        device = next(self.parameters()).device
        for key in tensor_dict.keys():
            tensor_dict[key] = tensor_dict[key].to(device)
        """
        tensor_dict:
            input_ids tensor([[  101,  2265,  2033,  3032,  4396,  2011, 14230,   102,  3842,  1064,
          3793,  1064,  2142,  2163,   102,  7977,  4968,  4031,  1064,  2613,
          1064,  2538,  1010,  4724,  2683,  1010,   102]])
            segment_ids tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1]])
            attention_mask tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1.]])
            context_token_indices tensor([[0, 1, 2, 3, 4, 5, 6]])     [INAME], male, female, [SEP], age, 18, to, 19
            context_token_mask tensor([[1., 1., 1., 1., 1., 1., 1.]])
            column_token_to_column_id tensor([[2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 2]])
            column_token_mask tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 0.]])
            column_mask tensor([[1., 1.]])
        instances:
           [{'tokens': ['[CLS]', 'show', 'me', 'countries', 'ranked', 'by', 'gdp', '[SEP]', 'nation', '|', 'text', '|', 'united', 'states', '[SEP]', 'gross', 'domestic', 'product', '|', 'real', '|', '21', ',', '43', '##9', ',', '[SEP]'], 'segment_a_length': 8, 'column_spans': [{'first_token': (8, 9), 'column_name': (8, 9), 'other_tokens': [9, 11], 'type': (10, 11), 'value': (12, 14), 'whole_span': (8, 14)}, {'first_token': (15, 16), 'column_name': (15, 18), 'other_tokens': [18, 20], 'type': (19, 20), 'value': (21, 26), 'whole_span': (15, 26)}], 'context_length': 7, 'context_span': (0, 7)}] 
        """

        context_encoding, header_encoding, index_name_encoding, encoding_info = self.encode_context_and_table(
            **tensor_dict,
            return_bert_encoding=True
        )

        info = {
            'tensor_dict': tensor_dict,
            'instances': instances,
            **encoding_info
        }

        return context_encoding, header_encoding, index_name_encoding, info

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        if not any(key.startswith('_bert_model') for key in state_dict):
            logging.warning('warning: loading model from an old version')
            self._bert_model.load_state_dict(state_dict, strict)
        else:
            super(VanillaTableBert, self).load_state_dict(state_dict, strict)
