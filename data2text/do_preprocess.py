"""Preprocess train/dev/test data into source/target pairs. """

import os
import json
import argparse
from typing import Any, Dict, Iterable, List, Tuple

import logging
logger = logging.getLogger(__name__)


# %% table info

def load_table(table_id: str) -> Dict: 
    table_path = os.path.join(args.dataset_dir, args.table_subdir, args.table_type, f"{table_id}.json")
    with open(table_path, 'r') as fr: 
        table = json.load(fr) 
    return table 

def get_ascendant_cells(cell_coords: Tuple[int, int], table: Dict, field: str) -> Dict[Tuple, str]: 
    """From a cell coordinate, find all its ascendant cells. 
    args: 
        cell_coords: header cell (1, 1), or data cell (17, 4) 
    rets: 
        asc_dict: {(0, 2): 'cell_str2', ...}
    """ 
    asc_cells = set([cell_coords])
    
    # find leaf coords 
    if field == 'top_root': 
        base_col_idx = cell_coords[1]
        
        n = table['top_header_rows_num']
        if cell_coords[0] >= n: 
            base_row_idx = n - 1 
        else: 
            base_row_idx = cell_coords[0]
    elif field == 'left_root': 
        base_row_idx = cell_coords[0] 
        
        n = table['left_header_columns_num'] 
        if cell_coords[1] >= n: 
            base_col_idx = n - 1 
        else: 
            base_col_idx = cell_coords[1] 
    else: 
        base_row_idx, base_col_idx = cell_coords

    dfs(table[field], (base_row_idx, base_col_idx), asc_cells) 
    
    asc_dict = {} 
    for coords in asc_cells: 
        if max(coords) > -1 and coords[0] < len(table['texts']) and coords[1] < len(table['texts'][coords[0]]): 
            asc_dict[coords] = table['texts'][coords[0]][coords[1]] 

    return asc_dict 

def dfs(node: Dict, ref_coords: Tuple[int, int], ascendants: List[Tuple[int, int]]) -> bool: 
    """Searching from the (current) node. 
    If node coordinates match, return True to propagate back to ascendants. 
    Else: continue to children if have any. Otherwise would terminate the path. 
    """
    if (node['row_index'] == ref_coords[0]) and (node['column_index'] == ref_coords[1]): 
        ascendants.add(ref_coords)
        return True 
    for child_node in node['children']: 
        if dfs(child_node, ref_coords, ascendants): 
            r, c = node['row_index'], node['column_index']
            ascendants.add( (r, c) ) 
            return True 
    return False    # no 'children' or not any that matches 


# table list for parent (metric) evaluation 

def clean_text(text: str) -> str:
    """Only has single blankspace as delimiters."""
    parts = text.split()
    parts = [p for part in parts for p in part.split('\t')]
    parts = [p for part in parts for p in part.split('\n')]
    cleaned_text = ' '.join(parts)
    return cleaned_text

def get_tuple(attr: str, text: str) -> Tuple[str, str]:
    """Return table-parent entry: attr|||value """
    raw_value = clean_text(text)
    value = raw_value.replace('|', '-')
    return (attr, value)

def get_table_parent_list(linked_cells: Dict, table: Dict) -> List:
    """Return a list of tuples as required by the PARENT metric.
    args:
        linked_cells: {'corner', 'top', 'left', 'data'}
    rets:
        *table_parent_array: List[Tuple(attribute, value)]
        table_parent_str: '\t'-separated
    """
    table_parent_array = []

    title_tuple = get_tuple('title', table['title'])
    table_parent_array.append(title_tuple)

    for coords, cellstr in linked_cells.items(): 
        cell_tuple = get_tuple(attr='cell', text=str(cellstr)) 
        table_parent_array.append(cell_tuple) 

    return table_parent_array



# %% iterate

def iterate_entity_link_by_field(entity_link: Dict, field: str, return_text: bool = False) -> Iterable[Tuple]: 
    """Iterate the cells in the `entity_link` field. 
    args: 
        entity_link: {'
            'top': {'the fy 2017 r&d budget': {'(0, 1)': '2017 actual'}}, 
			'left': {'pre-production development activities': {'(18, 0)': 'total'}}, 
			'top_left_corner': {}
        }
    rets: 
        Iterate(cell_coords): [(0,1), ...] 
    """
    field_links = entity_link[field]
    for text_span, ref_cells in field_links.items(): 
        for cell_coords, cell_text in ref_cells.items(): 
            cell_coords = eval(cell_coords)
            int_cell_coords = (int(cell_coords[0]), int(cell_coords[1]))
            if return_text: yield {int_cell_coords: cell_text}
            else: yield int_cell_coords

def iterate_entity_link(entity_link: Dict, return_text: bool = False) -> Iterable[Tuple]: 
    """Iterate the cells in the `entity_link` field. 
    args: 
        entity_link: {'
            'top': {'the fy 2017 r&d budget': {'(0, 1)': '2017 actual'}}, 
			'left': {'pre-production development activities': {'(18, 0)': 'total'}}, 
			'top_left_corner': {}
        }
    rets: 
        Iterate(cell_coords): [(0,1), ...] 
    """
    for field in entity_link.keys():  # ['top', 'left', 'top_left_corner'] 
        for item in iterate_entity_link_by_field(entity_link, field, return_text): 
            yield item


def iterate_quantity_link(quantity_link: Dict, return_text: bool = True) -> Iterable[Tuple]: 
    """Iterate the cells in the `quantity_link` field. 
    args: 
        quantity_link: {'
            ''125.3 billion': {'(17, 1)': 125289.0}, 
			'[ANSWER]': {'(18, 1)': 154983.0}
        }
    rets: 
        Iterate(cell_coords): [(17,1), ...] 
    """ 
    for text_span, ref_cells in quantity_link.items(): 
        for cell_coords, cell_text in ref_cells.items(): 
            cell_coords = eval(cell_coords)
            int_cell_coords = (int(cell_coords[0]), int(cell_coords[1]))
            if return_text: yield {int_cell_coords: cell_text}
            else: yield int_cell_coords 


def iterate_cells_coords(highlighted_cells: Dict) -> List[Tuple[int, int]]: 
    cell_coords = list(highlighted_cells.keys())
    return sorted(cell_coords, key=lambda x: (x[0], x[1]))



# %% cell string serialization 

def join_cells(cell_strings: List[Any], cell_delimiter: str = '|') -> str: 
    return f" {cell_delimiter} ".join([str(cs) for cs in cell_strings])


def join_aggrs(aggregation: List[str], answer: List[Any]) -> str: 
    return f"{' '.join(aggregation)} {' '.join([str(a) for a in answer])}"


def add_tag(text: str, tag: str, do_head: bool = True, do_tail: bool = False):
    """Add field tags to the text.""" 
    if do_head == True: prefix = f'{tag} '
    else: prefix = ''

    if do_tail == True: suffix = f' {tag}'
    else: suffix = ''

    return f'{prefix}{text}{suffix}'



# %% main pipieline 

def prepare_model_input(sample: Dict) -> Dict:
    table = load_table(sample['table_id'])

    source_texts = [] 
    source_texts.append( add_tag(table['title'], '<title>') )

    linked_cells = sample['linked_cells']
    
    highlight_cells = {}
    if args.no_asc: 
        for ent_cell_dict in iterate_entity_link_by_field(linked_cells['entity_link'], 'top', return_text=True): 
            highlight_cells.update(ent_cell_dict)
        for ent_cell_dict in iterate_entity_link_by_field(linked_cells['entity_link'], 'left', return_text=True): 
            highlight_cells.update(ent_cell_dict)
    else: 
        for ent_cell_coords in iterate_entity_link_by_field(linked_cells['entity_link'], 'top', return_text=False): 
            ent_cell_dict = get_ascendant_cells(ent_cell_coords, table, 'top_root') 
            highlight_cells.update(ent_cell_dict)
        for ent_cell_coords in iterate_entity_link_by_field(linked_cells['entity_link'], 'left', return_text=False): 
            ent_cell_dict = get_ascendant_cells(ent_cell_coords, table, 'left_root') 
            highlight_cells.update(ent_cell_dict)

    
    for ent_cell_dict in iterate_entity_link_by_field(linked_cells['entity_link'], 'top_left_corner', return_text=True): 
        highlight_cells.update(ent_cell_dict) 

    if args.no_asc: 
        for qtt_cell_dict in iterate_quantity_link(linked_cells['quantity_link']): 
            highlight_cells.update(qtt_cell_dict)
    else: 
        for qtt_cell_coords in iterate_quantity_link(linked_cells['quantity_link'], return_text=False): 
            top_cell_dict = get_ascendant_cells(qtt_cell_coords, table, 'top_root') 
            highlight_cells.update(top_cell_dict)
            left_cell_dict = get_ascendant_cells(qtt_cell_coords, table, 'left_root') 
            highlight_cells.update(left_cell_dict)
    
    if args.no_split_fields: 
        cell_strings = [highlight_cells[k] for k in iterate_cells_coords(highlight_cells)]
        source_texts.append( add_tag(join_cells(cell_strings), '<cell>') )
    else: 
        top_header_rows_num = table['top_header_rows_num']
        left_header_columns_num = table['left_header_columns_num'] 
        top_strings, left_strings, corner_strings, data_strings = [], [], [], [] 
        for coords in iterate_cells_coords(highlight_cells): 
            cell_str = highlight_cells[coords]
            r, c = coords 
            if (r < top_header_rows_num) and (c < left_header_columns_num): 
                corner_strings.append(cell_str) 
            elif (r < top_header_rows_num): 
                top_strings.append(cell_str)
            elif (c < left_header_columns_num): 
                left_strings.append(cell_str) 
            else: 
                data_strings.append(cell_str) 
                
        source_texts.append( add_tag(join_cells(top_strings), '<top>') )
        source_texts.append( add_tag(join_cells(left_strings), '<left>') )
        source_texts.append( add_tag(join_cells(corner_strings), '<corner>') )
        source_texts.append( add_tag(join_cells(data_strings), '<data>') )

    if args.add_aggr: 
        aggr_str = join_aggrs(sample['aggregation'], sample['answer'])
        source_texts.append( add_tag(aggr_str, '<agg>') )
    
    table_list = get_table_parent_list(highlight_cells, table)
    
    return {
        'source': ' '.join(source_texts), 
        'target': sample['sub_sentence'], 
        'table_parent': table_list, 
    } 



def main(): 
    for in_path, out_path in zip(args.input_paths, args.output_paths): 
        print(f"from [{in_path}] >> to [{out_path}]")

        with open(in_path, 'r') as fr: 
            dataset = [json.loads(l.strip()) for l in fr]
        if args.test_topk: 
            dataset = dataset[: args.test_topk]
        print(f"collected {len(dataset)} samples")
        
        fw = open(out_path, 'w') 
        for idx, sample in enumerate(dataset): 
            if (idx + 1) % args.logging_steps == 0: 
                print(f"finished processing {idx + 1} samples")

            result = prepare_model_input(sample)
            fw.write(f"{json.dumps(result)}\n")

        fw.close()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--file_names', type=str, nargs='+', 
        default=['test_samples.jsonl', 'dev_samples.jsonl', 'train_samples.jsonl'])
    parser.add_argument('--output_dir', type=str, default='data')

    parser.add_argument('--table_subdir', type=str, default='tables')
    parser.add_argument('--table_type',type=str, default='hmt', choices=['hmt', 'raw'], 
        help='Use `raw` if including ascendant cells, otherwise use `hmt`.')

    parser.add_argument('--no_asc', action='store_true')
    parser.add_argument('--add_aggr', action='store_true')
    parser.add_argument('--no_split_fields', action='store_true')

    parser.add_argument('--logging_steps', type=int, default=500)
    parser.add_argument('--test_topk', type=int, default=None)

    args = parser.parse_args()

    args.input_paths = [os.path.join(args.dataset_dir, fn) for fn in args.file_names]
    args.output_paths = [os.path.join(args.output_dir, fn) for fn in args.file_names]
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.no_asc: canonical_table_type = 'hmt'
    else: canonical_table_type = 'raw' 
    if args.table_type != canonical_table_type: 
        logging.info(f"Should use `{canonical_table_type}` version of data. ")
        args.table_type = canonical_table_type

    main() 