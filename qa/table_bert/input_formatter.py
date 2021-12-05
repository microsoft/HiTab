from math import ceil
from random import choice, shuffle, sample, random
from typing import List, Callable, Dict, Any

from qa.table_bert.utils import BertTokenizer
from qa.table_bert.table_bert import MAX_BERT_INPUT_LENGTH
from qa.table_bert.config import TableBertConfig
from qa.table_bert.hm_table import *


class TableBertBertInputFormatter(object):
    def __init__(self, config: TableBertConfig, tokenizer: BertTokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.vocab_list = list(self.tokenizer.vocab.keys())


class TableTooLongError(ValueError):
    pass


class VanillaTableBertInputFormatter(TableBertBertInputFormatter):
    def get_header_input(self, header_info, header2id, token_offset, header_input_template=None):
        """ HMT converts a header into bert tokens."""
        input = []
        span_map = {
            'first_token': (token_offset, token_offset + 1)
        }
        if not header_input_template:
            header_input_template = self.config.header_input_template
        for token in header_input_template:
            start_token_abs_position = len(input) + token_offset
            if token == 'name':
                span_map['name'] = (start_token_abs_position,
                                           start_token_abs_position + len(header_info['tokenized_name']))
                input.extend(header_info['tokenized_name'])
            elif token == 'type':
                span_map['type'] = (start_token_abs_position,
                                    start_token_abs_position + len(header_info['tokenized_type']))
                input.extend(header_info['tokenized_type'])
            elif token == 'first_token':
                span_map['name'] = (start_token_abs_position,
                                    start_token_abs_position + 1)
                input.append(header_info['tokenized_name'][0])
            else:
                span_map.setdefault('other_tokens', []).append(start_token_abs_position)
                input.append(token)

        span_map['whole_span'] = (token_offset, token_offset + len(input))
        span_map['header_id'] = header2id[header_info['name']]

        return input, span_map

    def get_sequence_input(self, hmt, start_idx, header_input_template=None):
        """ Wrap get sequence input."""
        sequence_input_tokens = []
        left_token_span_maps, top_token_span_maps = {'level_spans': []}, {'level_spans': []}
        left_depth, top_depth = hmt.get_tree_depth(hmt.left_root), hmt.get_tree_depth(hmt.top_root)
        for direction in ['<LEFT>', '<TOP>']:
            max_level = left_depth if direction == '<LEFT>' else top_depth
            direction_token_span_maps = left_token_span_maps if direction == '<LEFT>' else top_token_span_maps
            direction_start_idx = start_idx
            for level in range(max_level):
                level_start_idx = start_idx
                level_token_span_maps = {'header_spans': []}
                index_name = f"INAME_{direction}_{level}"
                level_token_span_maps['level_id'] = hmt.index_name2id[index_name]
                level_header_info = hmt.index_name2header_info[index_name]

                # add index name at the beginning of each level
                if hmt.index_name_map[index_name] is None:
                    index_name_input_tokens = [self.config.iname_placeholder]
                else:
                    index_name_input_tokens = hmt.index_name_map[index_name].copy()
                level_token_span_maps['index_name_span'] = (start_idx, start_idx + len(index_name_input_tokens))
                if len(hmt.index_name2header_info[index_name]) > 0:
                    index_name_input_tokens.append(self.config.header_delimiter)

                sequence_input_tokens.extend(index_name_input_tokens)
                start_idx = start_idx + len(index_name_input_tokens)

                # add headers
                for header_idx in range(len(level_header_info)):
                    header_info = level_header_info[header_idx]
                    header_input_tokens, header_span_map = self.get_header_input(
                        header_info,
                        hmt.header2id,
                        start_idx,
                        header_input_template
                    )
                    if header_idx != len(level_header_info) - 1:
                        header_input_tokens.append(self.config.header_delimiter)

                    sequence_input_tokens.extend(header_input_tokens)
                    start_idx = start_idx + len(header_input_tokens)
                    level_token_span_maps['header_spans'].append(header_span_map)

                level_token_span_maps['whole_span'] = (level_start_idx, start_idx)  # not include [SEP]
                sequence_input_tokens.append(self.config.level_delimiter)
                start_idx += 1
                direction_token_span_maps['level_spans'].append(level_token_span_maps)

            direction_token_span_maps['whole_span'] = (direction_start_idx, start_idx - 1)  # not include [SEP]

        return sequence_input_tokens, left_token_span_maps, top_token_span_maps

    def get_input(self, context: List[str], hmt: HMTable, trim_long_table=False):
        """ HMT get input formatter."""
        if self.config.context_first:
            table_tokens_start_idx = len(context) + 2  # account for [CLS] and [SEP]
            # account for [CLS] and [SEP], and the ending [SEP]
            max_table_token_length = MAX_BERT_INPUT_LENGTH - len(context) - 2 - 1
        else:
            table_tokens_start_idx = 1  # account for starting [CLS]
            # account for [CLS] and [SEP], and the ending [SEP]
            max_table_token_length = MAX_BERT_INPUT_LENGTH - len(context) - 2 - 1

        try:
            result = self.get_sequence_input(hmt, table_tokens_start_idx)
            sequence_input_tokens, left_token_span_maps, top_token_span_maps = result
            if len(sequence_input_tokens) > max_table_token_length:
                raise TableTooLongError(f"Error: Sequence length {len(sequence_input_tokens)} exceeds 512.")
        except TableTooLongError as e:
            # TODO: degrade in an elegant way
            result = self.get_sequence_input(hmt, table_tokens_start_idx, ['name'])
            sequence_input_tokens, left_token_span_maps, top_token_span_maps = result
            if len(sequence_input_tokens) > max_table_token_length:
                result = self.get_sequence_input(hmt, table_tokens_start_idx, ['first_token'])
                sequence_input_tokens, left_token_span_maps, top_token_span_maps = result
                if len(sequence_input_tokens) > max_table_token_length:
                    raise TableTooLongError(f"Error: Sequence length(first token) {len(sequence_input_tokens)} exceeds 512.")

        if sequence_input_tokens[-1] == self.config.level_delimiter:
            del sequence_input_tokens[-1]

        if self.config.context_first:
            sequence = ['[CLS]'] + context + ['[SEP]'] + sequence_input_tokens + ['[SEP]']
            segment_a_length = len(context) + 2
            context_span = (0, 1 + len(context))
        else:
            sequence = ['[CLS]'] + sequence_input_tokens + ['[SEP]'] + context + ['[SEP]']
            segment_a_length = len(sequence_input_tokens) + 2
            context_span = (len(sequence_input_tokens) + 1, len(sequence_input_tokens) + 1 + 1 + len(context) + 1)

        instance = {
            'tokens': sequence,
            'segment_a_length': segment_a_length,
            'left_spans': left_token_span_maps,
            'top_spans': top_token_span_maps,
            'context_length': 1 + len(context),  # beginning [CLS]/[SEP] + input question
            'context_span': context_span,
        }

        return instance