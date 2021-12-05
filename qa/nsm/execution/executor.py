"""Functions implementations of logic forms on hierarchical matrix table. """

from typing import Dict, List, Tuple, Union
import collections
from statistics import mean, stdev, variance, median
import copy

from qa.table_bert.hm_table import TreeNode, DataNode, AGGREGATION2ID
from qa.nsm.execution.type_system import get_hmt_type_hierarchy


class HMTExecutor(object):

    def __init__(self, table_kg: Dict):
        """ Hierarchical Table Executor."""
        self.hmt = table_kg["kg"]

    # Logic Form Functions (filter_function)
    def filter_tree_str_contain(self, op_region, index_name, *string_headers):
        """ Filter rows/cols with at least one header in string_headers. Operating region will be changed."""
        assert len(op_region.left_ids) > 0 and len(op_region.top_ids) > 0
        _, direction, _ = index_name.split('_')
        coord_list = self.get_coord(direction, string_headers)
        new_ids = self.get_new_ids_by_tree(coord_list, direction)
        if direction == '<LEFT>':
            op_region.left_ids = list(set(new_ids).intersection(set(op_region.left_ids)))
        else:
            op_region.top_ids = list(set(new_ids).intersection(set(op_region.top_ids)))
        return copy.deepcopy(op_region)

    def filter_tree_str_not_contain(self, op_region, index_name, *string_headers):
        """ Filter rows/cols with no header in string_headers. Operating region will be changed."""
        assert len(op_region.left_ids) > 0 and len(op_region.top_ids) > 0
        _, direction, _ = index_name.split('_')
        coord_list = self.get_coord(direction, string_headers)
        contain_ids = self.get_new_ids_by_tree(coord_list, direction)
        new_ids = [_ for _ in range(len(self.hmt.data_region)) if _ not in contain_ids] if direction == '<LEFT>' \
            else [_ for _ in range(len(self.hmt.data_region[0])) if _ not in contain_ids]
        if direction == '<LEFT>':
            op_region.left_ids = list(set(new_ids).intersection(set(op_region.left_ids)))
        else:
            op_region.top_ids = list(set(new_ids).intersection(set(op_region.top_ids)))
        return copy.deepcopy(op_region)

    def filter_level(self, op_region, index_name):
        """ Filter rows/cols on target level on target direction. Operating region will be changed."""
        _, direction, level = index_name.split('_')
        level = int(level)
        if direction == '<LEFT>':
            op_region.left_ids = [left_id for left_id in op_region.left_ids if
                                  self.hmt.left_line_idx2level[left_id] == level]
        else:
            op_region.top_ids = [top_id for top_id in op_region.top_ids if
                                 self.hmt.top_line_idx2level[top_id] == level]
        return copy.deepcopy(op_region)

    # Logic Form Functions (operation_function)
    def max(self, op_region, index_name: str):
        """ Max aggregation. Operating region will be changed.
        Returns:
            a list of values
        """
        return self.aggregate(op_region, index_name, 'max')

    def min(self, op_region, index_name: str):
        """ Min aggregation over cols. Operating region will be changed.
        Returns:
            a list of values
        """
        return self.aggregate(op_region, index_name, 'min')

    def average(self, op_region, index_name: str):
        """ Average aggregation. Operating region will be changed.
        Returns:
            a list of values
        """
        return self.aggregate(op_region, index_name, 'avg')

    def sum(self, op_region, index_name: str):
        """ Sum aggregation. Operating region will be changed.
        Returns:
            a list of values
        """
        return self.aggregate(op_region, index_name, 'sum')

    def stdev(self, op_region, index_name: str):
        """ Standard deviation aggregation. Operating region will be changed.
        Returns:
            a list of values
        """
        return self.aggregate(op_region, index_name, 'std')

    def variance(self, op_region, index_name: str):
        """ Variance aggregation. Operating region will be changed.
        Returns:
            a list of values
        """
        return self.aggregate(op_region, index_name, 'var')

    def median(self, op_region, index_name: str):
        """ Median aggregation. Operating region will be changed.
        Returns:
            a list of values
        """
        return self.aggregate(op_region, index_name, 'med')

    def argmax(self, op_region):
        """ Argmax of left/top.
        (Assert top_ids/left_ids has one element)
        Returns:
            a list(mostly with one element) of headers in its type
        """
        return self.arg_extremum(op_region, 'max')

    def argmin(self, op_region):
        """ Argmin of left/top.
        (Assert top_ids/left_ids has one element)
        Returns:
            a list(mostly with one element) of left/top headers in its type
        """
        return self.arg_extremum(op_region, 'min')

    def count(self, op_region, index_name: str):
        """ Count number of rows/cols.
        Returns:
            a number
        """
        # assert len(op_region.left_ids) > 0 and len(op_region.top_ids) > 0
        _, direction, level = index_name.split('_')
        # level = int(level)
        return len(op_region.left_ids) if direction == '<LEFT>' else len(op_region.top_ids)

    def difference(self, op_region):
        """ Difference of two numbers, i.e. abs(b-a)
        (Assert (len(left_ids) == 1 and len(top_ids) == 2) or (len(left_ids)==2 and len(top_ids) == 1))
        Returns:
            a number
        """
        # assert (len(op_region.left_ids) == 1 and len(op_region.top_ids) == 2) or (  # check in valid functions
        #             len(op_region.left_ids) == 2 and len(op_region.top_ids) == 1)
        direction = '<LEFT>' if (len(op_region.top_ids) == 1) else '<TOP>'
        data_nodes = self.get_data_nodes_of_single_line(op_region, direction)
        value1, value2 = data_nodes[0].value, data_nodes[1].value

        if not ((isinstance(value1, float) or isinstance(value1, int))
                and (isinstance(value2, float) or isinstance(value2, int))):
            raise NonNumericOperationError("Difference on non-numeric data.")
        return abs(value1 - value2)

    def difference_rate(self, op_region):
        """ Difference rate of two numbers, i.e. (b-a)/a
        (Assert (len(left_ids) == 1 and len(top_ids) == 2) or (len(left_ids)==2 and len(top_ids) == 1))
        Returns:
            a number
        """
        direction = '<LEFT>' if (len(op_region.top_ids) == 1) else '<TOP>'
        data_nodes = self.get_data_nodes_of_single_line(op_region, direction)
        value1, value2 = data_nodes[0].value, data_nodes[1].value

        if not ((isinstance(value1, float) or isinstance(value1, int))
                and (isinstance(value2, float) or isinstance(value2, int))):
            raise NonNumericOperationError("Difference rate on non-numeric data.")
        if value1 == 0:
            return None
        return abs((value2 - value1) / value1)

    def proportion(self, op_region):
        """ Proportion of two numbers, i.e. b/a
        (Assert (len(left_ids) == 1 and len(top_ids) == 2) or (len(left_ids)==2 and len(top_ids) == 1))
        Returns:
            a number
        """
        direction = '<LEFT>' if (len(op_region.top_ids) == 1) else '<TOP>'
        data_nodes = self.get_data_nodes_of_single_line(op_region, direction)
        value1, value2 = data_nodes[0].value, data_nodes[1].value

        if not ((isinstance(value1, float) or isinstance(value1, int))
                and (isinstance(value2, float) or isinstance(value2, int))):
            raise NonNumericOperationError("Proportion on non-numeric data.")
        if value1 == 0:
            return None
        return value2 / value1

    def rank(self, op_region):
        """ Rank data in current op_region.
        (Assert top_ids/left_ids has only one element)
        Returns:
            a list of values
        """
        direction = '<LEFT>' if (len(op_region.top_ids) == 1) else '<TOP>'
        data_nodes = self.get_data_nodes_of_single_line(op_region, direction)
        sorted_data_nodes = sorted(data_nodes, reverse=True,
                                   key=lambda x: x.value if isinstance(x.value, float) or isinstance(x.value, int) else float('-inf'),)
        return [dn.value for dn in sorted_data_nodes]

    def argrank(self, op_region):
        """ Header of ranked data in current op_region.
        (Assert top_ids/left_ids has only one element)
        Returns:
            a list of headers in its type
        """
        direction = '<LEFT>' if (len(op_region.top_ids) == 1) else '<TOP>'
        data_nodes = self.get_data_nodes_of_single_line(op_region, direction)
        sorted_data_nodes = sorted(data_nodes, reverse=True,
                                   key=lambda x: x.value if isinstance(x.value, float) or isinstance(x.value, int) else float('-inf'),)
        return self.hop(sorted_data_nodes, direction)

    def opposite(self, op_region):
        """ Take opposite value of current op_region.
        (Assert top_ids/left_ids has only one element)
        Returns:
            a list of data
        """
        if len(op_region.left_ids) == 1 and len(op_region.top_ids) == 1:
            data_nodes = self.get_data_node_of_single_cell(op_region)
        else:
            direction = '<LEFT>' if (len(op_region.top_ids) == 1) else '<TOP>'
            data_nodes = self.get_data_nodes_of_single_line(op_region, direction)
        return list(map(lambda x: -x.value if isinstance(x.value, float) or isinstance(x.value, int) else x.value, data_nodes))

    # -----------------------------------------------------------------------------------------
    # Autocomplete
    def autocomplete_identity(self, op_region, exp, exp_vals, tokens, token_vals, pc, debug=False):
        valid_tks = tokens
        # if debug:
        #     print('*' * 30)
        #     print(exp_vals)
        #     print(tokens)
        #     print(valid_tks)
        #     print('*' * 30)
        return valid_tks

    def autocomplete_filter_tree_str(self, op_region, exp, exp_vals, tokens, token_vals, pc, contain, debug=False):
        """ Assert: 1) queries are on the same level; 2) query level decreases when filtering; 3) filter result is
        not empty; 4) filter "top" must go behind filter "left". """
        l = len(exp_vals)
        token_val_dict = dict(zip(tokens, token_vals))
        valid_tks = []
        if l == 1:  # to generate direction
            for tk, tk_info in token_val_dict.items():
                _, direction, _ = tk_info['value'].split('_')
                if pc.n_filter_left_level == 1 and direction == '<TOP>':
                    valid_tks.append(tk)
                elif pc.n_filter_left_level == 0 and direction == '<LEFT>':
                    valid_tks.append(tk)
        elif l == 2:  # to generate first header
            _, direction, _ = exp_vals[1]['value'].split('_')
            cur_max_level = pc.cur_left_level if direction == '<LEFT>' else pc.cur_top_level
            # print(f"cur_max_level: {cur_max_level}")
            for tk, tk_info in token_val_dict.items():
                root = self.hmt.left_root if direction == "<LEFT>" else self.hmt.top_root
                if not contain and not self.search_header(root, tk_info['value']):
                    continue
                queries = (tk_info['value'],)
                if self.hmt.header2level[tk_info['value']] > cur_max_level \
                        and not self.empty_after_filter_tree(op_region, direction, queries, contain=contain):
                    valid_tks.append(tk)
        elif l == 3:  # to generate second header
            _, direction, _ = exp_vals[1]['value'].split('_')
            cur_max_level = pc.cur_left_level if direction == '<LEFT>' else pc.cur_top_level
            valid_tks.append(')')
            # print(f"cur_max_level: {cur_max_level}")
            for tk, tk_info in token_val_dict.items():
                if int(tk[1:]) <= int(exp[2][1:]):  # constraint the second header variable order must be higher than first one, e.g. filter left v27 v28
                    continue
                if exp_vals[2]['value'] == tk_info['value'] \
                        or (not self.hmt.header2level[exp_vals[2]['value']] == self.hmt.header2level[tk_info['value']]):
                    continue
                root = self.hmt.left_root if direction == "<LEFT>" else self.hmt.top_root
                if not contain and not self.search_header(root, tk_info['value']):
                    continue
                queries = (exp_vals[2]['value'], tk_info['value'])
                if self.hmt.header2level[tk_info['value']] > cur_max_level \
                        and not self.empty_after_filter_tree(op_region, direction, queries[1:], contain=contain) \
                        and not self.empty_after_filter_tree(op_region, direction, queries, contain=contain):
                    valid_tks.append(tk)
        return valid_tks

    def autocomplete_filter_tree_str_contain(self, op_region, exp, exp_vals, tokens, token_vals, pc, debug=False):
        """ Wrapper for autocomplete filter tree str contain."""
        return self.autocomplete_filter_tree_str(op_region, exp, exp_vals, tokens, token_vals, pc, contain=True)

    def autocomplete_filter_tree_str_not_contain(self, op_region, exp, exp_vals, tokens, token_vals, pc, debug=False):
        """ Wrapper for autocomplete filter tree str not contain."""
        return self.autocomplete_filter_tree_str(op_region, exp, exp_vals, tokens, token_vals, pc, contain=False)

    def autocomplete_filter_level(self, op_region, exp, exp_vals, tokens, token_vals, pc, debug=False):
        """ Assert 1) target level is larger than pc.cur_left_level or pc.cur_top_level; 2) level has data;
            3) filter "top" must go after filter "left". """
        l = len(exp_vals)
        token_val_dict = dict(zip(tokens, token_vals))
        valid_tks = []
        for tk, tk_info in token_val_dict.items():
            _, direction, level = tk_info['value'].split('_')
            level = int(level)
            new_ids = []
            if pc.n_filter_left_level == 1 and direction == '<TOP>':
                max_level = self.hmt.get_tree_depth(self.hmt.top_root) - 1
                cur_max_level = pc.cur_top_level if pc.cur_top_level > 0 else max_level
                # If no filter_tree, default the deepest level;
                # Else all levels deeper or equal to cur_max_level is valid, as long as new region is not empty.
                if level >= cur_max_level:
                    new_ids = [top_id for top_id in op_region.top_ids if
                               self.hmt.top_line_idx2level[top_id] == level]
            elif pc.n_filter_left_level == 0 and direction == '<LEFT>':
                max_level = self.hmt.get_tree_depth(self.hmt.left_root) - 1
                cur_max_level = pc.cur_left_level if pc.cur_left_level > 0 else max_level
                if level >= cur_max_level:
                    new_ids = [left_id for left_id in op_region.left_ids if
                               self.hmt.left_line_idx2level[left_id] == level]
            if len(new_ids) > 0:
                valid_tks.append(tk)

        return valid_tks

    def autocomplete_aggr(self, op_region, exp, exp_vals, tokens, token_vals, pc, debug=False):
        """ Assert 1) aggr level is no deeper than filter_level level."""
        l = len(exp_vals)
        token_val_dict = dict(zip(tokens, token_vals))
        valid_tks = []
        for tk, tk_info in token_val_dict.items():
            _, direction, level = tk_info['value'].split('_')
            level = int(level)
            if direction == "<LEFT>" and level <= pc.cur_left_level:
                valid_tks.append(tk)
            elif direction == "<TOP>" and level <= pc.cur_top_level:
                valid_tks.append(tk)
        return valid_tks

    # -----------------------------------------------------------------------------------------
    # Get apis
    def get_api(self):
        func_dict = collections.OrderedDict()
        # TODO: when adding new funcs, remember to change hard-coded action ids, e.g. random_explore.py

        func_dict['filter_tree_str_contain'] = dict(
            name='filter_tree_str_contain',
            args=[{'types': ['direction_index_name']}, {'types': ['header']}],
            return_type='region',
            autocomplete=self.autocomplete_filter_tree_str_contain,
            type='filter_tree_function',
            value=self.filter_tree_str_contain
        )

        func_dict['filter_tree_str_not_contain'] = dict(
            name='filter_tree_str_not_contain',
            args=[{'types': ['direction_index_name']}, {'types': ['header']}],
            return_type='region',
            autocomplete=self.autocomplete_filter_tree_str_not_contain,
            type='filter_tree_function',
            value=self.filter_tree_str_not_contain
        )

        func_dict['filter_level'] = dict(
            name='filter_level',
            args=[{'types': ['non_direction_index_name']}],
            return_type='region',
            autocomplete=self.autocomplete_filter_level,
            type='filter_level_function',
            value=self.filter_level
        )

        func_dict['max'] = dict(
            name='max',
            args=[{'types': ['index_name']}],
            return_type='num_list',
            autocomplete=self.autocomplete_aggr,
            type='aggregation_function',
            value=self.max
        )

        func_dict['min'] = dict(
            name='min',
            args=[{'types': ['index_name']}],
            return_type='num_list',
            autocomplete=self.autocomplete_aggr,
            type='aggregation_function',
            value=self.min
        )

        func_dict['sum'] = dict(
            name='sum',
            args=[{'types': ['index_name']}],
            return_type='num_list',
            autocomplete=self.autocomplete_aggr,
            type='aggregation_function',
            value=self.sum
        )

        func_dict['average'] = dict(
            name='average',
            args=[{'types': ['index_name']}],
            return_type='num_list',
            autocomplete=self.autocomplete_aggr,
            type='aggregation_function',
            value=self.average
        )

        func_dict['argmax'] = dict(  # as for 'return_function' type，return type is not used since there is no proceeding function
            name='argmax',
            args=[],
            return_type='list',
            autocomplete=self.autocomplete_identity,
            type='return_function',
            value=self.argmax
        )

        func_dict['argmin'] = dict(
            name='argmin',
            args=[],
            return_type='list',
            autocomplete=self.autocomplete_identity,
            type='return_function',
            value=self.argmin
        )

        func_dict['count'] = dict(
            name='count',
            args=[{'types': ['direction_index_name']}],
            return_type='int',
            autocomplete=self.autocomplete_identity,
            type='return_function',
            value=self.count
        )

        func_dict['difference'] = dict(
            name='difference',
            args=[],
            return_type='num',
            autocomplete=self.autocomplete_identity,
            type='return_function',
            value=self.difference
        )

        func_dict['difference_rate'] = dict(
            name='difference_rate',
            args=[],
            return_type='num',
            autocomplete=self.autocomplete_identity,
            type='return_function',
            value=self.difference_rate
        )

        func_dict['proportion'] = dict(
            name='proportion',
            args=[],
            return_type='num',
            autocomplete=self.autocomplete_identity,
            type='return_function',
            value=self.proportion
        )

        func_dict['rank'] = dict(
            name='rank',
            args=[],
            return_type='num_list',
            autocomplete=self.autocomplete_identity,
            type='return_function',
            value=self.rank
        )

        func_dict['argrank'] = dict(
            name='argrank',
            args=[],
            return_type='num_list',
            autocomplete=self.autocomplete_identity,
            type='return_function',
            value=self.argrank
        )

        func_dict['opposite'] = dict(
            name='opposite',
            args=[],
            return_type='num_list',
            autocomplete=self.autocomplete_identity,
            type='return_function',
            value=self.opposite
        )

        constant_dict = collections.OrderedDict()
        # add header constants
        left_tree_depth = self.hmt.get_tree_depth(self.hmt.left_root)
        top_tree_depth = self.hmt.get_tree_depth(self.hmt.top_root)
        for direction in ['<LEFT>', '<TOP>']:
            max_level = left_tree_depth if direction == '<LEFT>' else top_tree_depth
            for level in range(max_level):
                index_name = f"INAME_{direction}_{level}"
                for header_info in self.hmt.index_name2header_info[index_name]:
                    tp = 'string_header'
                    if header_info['type'] == 'number':
                        tp = 'num_header'
                    elif header_info['type'] == 'datetime':
                        tp = 'datetime_header'
                    constant_dict[header_info['name']] = dict(
                        value=header_info['name'], type=tp, name=header_info['name']
                    )

        # add index name constants, traverse two times to make sure index name appear behind all headers in memory
        for direction in ['<LEFT>', '<TOP>']:
            max_level = left_tree_depth if direction == '<LEFT>' else top_tree_depth
            for level in range(max_level):
                index_name = f"INAME_{direction}_{level}"
                if level == 0:
                    tp = 'direction_index_name'
                elif (direction == '<LEFT>' and level == left_tree_depth - 1) \
                        or (direction == '<TOP>' and level == top_tree_depth - 1):
                    tp = 'leaf_index_name'
                else:
                    tp = 'non_direction_index_name'
                constant_dict[index_name] = dict(
                    value=index_name, type=tp, name=index_name
                )

        type_hierarchy = get_hmt_type_hierarchy()
        return dict(type_hierarchy=type_hierarchy,
                    func_dict=func_dict,
                    constant_dict=constant_dict)

    # -----------------------------------------------------------------------------------------
    # Utility Functions
    def get_coord_with_direction(self, queries: Tuple[str]):
        """ Find coord list and direction. Queries must be on the same direction."""
        coord_list, direction = [], None
        for query in queries:
            new_coord_list = self.search_header(self.hmt.left_root, query)
            if new_coord_list:
                direction = "<LEFT>"
            else:
                new_coord_list = self.search_header(self.hmt.top_root, query)
                direction = "<TOP>"
            coord_list.extend(new_coord_list)
        coord_list = list(set(coord_list))
        return coord_list, direction

    def get_coord(self, direction: str, queries: Tuple[str]):
        """ Find coord list on given direction."""
        coord_list = []
        for query in queries:
            if direction == '<LEFT>':
                new_coord_list = self.search_header(self.hmt.left_root, query)
            else:
                new_coord_list = self.search_header(self.hmt.top_root, query)
            coord_list.extend(new_coord_list)
        coord_list = list(set(coord_list))
        return coord_list

    def empty_after_filter_tree(self, op_region, direction, queries, contain) -> bool:
        """ Check whether only empty regions after filter. Used in autocomplete for fiter tree."""
        coord_list = self.get_coord(direction, queries)
        new_ids = self.get_new_ids_by_tree(coord_list, direction)
        if not contain:
            new_ids = [_ for _ in range(len(self.hmt.data_region)) if _ not in new_ids] if direction == '<LEFT>' \
                else [_ for _ in range(len(self.hmt.data_region[0])) if _ not in new_ids]
        if (direction == '<LEFT>' and len(set(new_ids).intersection(set(op_region.left_ids))) > 0) \
                or (direction == '<TOP>' and len(set(new_ids).intersection(set(op_region.top_ids))) > 0):
            return False
        else:
            return True

    def get_data_nodes_of_single_line(self, op_region, direction):
        """ Get data nodes when op_region is a single line."""
        ref_id = op_region.top_ids[0] if direction == '<LEFT>' else op_region.left_ids[0]
        if direction == '<LEFT>':
            if ref_id in op_region.top_aggr:
                data_nodes = [_ for _ in op_region.top_aggr[ref_id] if _ is not None]
            else:
                data_nodes = [self.hmt.data_region[left_id][ref_id] for left_id in op_region.left_ids]
        else:
            if ref_id in op_region.left_aggr:
                data_nodes = [_ for _ in op_region.left_aggr[ref_id] if _ is not None]
            else:
                data_nodes = [self.hmt.data_region[ref_id][top_id] for top_id in op_region.top_ids]
        return data_nodes

    def get_data_node_of_single_cell(self, op_region):
        """ Get data node when op_region is a single cell."""
        left_id, top_id = op_region.left_ids[0], op_region.top_ids[0]
        if left_id in op_region.left_aggr:
            return op_region.left_aggr[left_id]
        elif top_id in op_region.top_aggr:
            return op_region.top_aggr[top_id]
        else:
            return [self.hmt.data_region[left_id][top_id]]

    def hop(self, data_nodes, direction):
        """ Hop from data nodes to left/top headers. """
        if direction == '<LEFT>':
            headers = [self.hmt.coord2left[data_node.left_coord].name for data_node in data_nodes if
                       data_node is not None]
        else:
            headers = [self.hmt.coord2top[data_node.top_coord].name for data_node in data_nodes if
                       data_node is not None]
        return headers

    def search_header(self, root: TreeNode, query: str):
        """ Wrapper for searching input value from root node."""
        result = []
        self._search_header(root, query, result)
        return result

    def _search_header(self, root: TreeNode, query: str, result: List):
        """ DFS."""
        if self.match_string(root.name, query):
            result.append(root.coord)
        else:
            for child in root.children:
                self._search_header(child, query, result)

    def match_string(self, name: str, query: str, fuzzy: int = 0):
        """Binary classify n-gram match of the header name and query. """
        if fuzzy == 0:
            return name == query
        else:
            return False

    def arg_extremum(self, op_region, goal: str):
        """ Find the left/top header with max/min data value on current left_ids/top_ids.
        (Assert top_ids/left_ids has one element)
        Returns:
            a list(mostly with one element) of headers in its type
        """
        # assert (len(op_region.top_ids) == 1 or len(op_region.left_ids) == 1)  # check in valid functions
        if len(op_region.top_ids) == 1:
            direction = '<LEFT>'
        else:
            direction = '<TOP>'
        data_nodes = self.get_data_nodes_of_single_line(op_region, direction)
        data_nodes = [x for x in data_nodes if isinstance(x.value, int) or isinstance(x.value, float)]

        if len(data_nodes) == 0:
            raise NonNumericOperationError("Argmax/argmin on non-numeric data.")
        desc_order_flag = True if goal == 'max' else False
        sorted_data_nodes = sorted(data_nodes, key=lambda x: x.value, reverse=desc_order_flag)
        best_value = sorted_data_nodes[0].value
        results, new_ids = [], []
        for data_node in sorted_data_nodes:
            if data_node.value != best_value:
                break
            if direction == '<LEFT>':
                results.append(self.hmt.coord2left[data_node.left_coord].name)
                new_ids.append(self.hmt.coord2left[data_node.left_coord].start_idx)
            else:
                results.append(self.hmt.coord2top[data_node.top_coord].name)
                new_ids.append(self.hmt.coord2top[data_node.top_coord].start_idx)
        return results

    def aggregate(self, op_region, index_name: str, goal: str):
        """ Aggregation over rows/cols. Operating left_ids/top_ids will be changed.
        Returns:
            a list of values
            i.e. we can only take the whole line from left_aggr/top_aggr, rather than specific positions.
        """
        assert len(op_region.left_ids) > 0 and len(op_region.top_ids) > 0
        assert goal in AGGREGATION2ID

        _, direction, _ = index_name.split('_')
        data_nodes = []
        for left_id in op_region.left_ids:
            for top_id in op_region.top_ids:
                data_nodes.append(self.hmt.data_region[left_id][top_id])

        data_node_group = self.group_by_index_name(data_nodes, index_name)
        aggr_group = {}
        for coord, group in data_node_group.items():
            aggr_group[coord] = self.calculate_aggregation(group, goal)

        if direction == '<LEFT>':
            aggr_line = [DataNode(v, None, (AGGREGATION2ID[goal],), k) for k, v in aggr_group.items()]
            aggr_line.sort(key=lambda x: x.top_coord)
            op_region.top_ids = [AGGREGATION2ID[goal]]
            op_region.top_aggr[AGGREGATION2ID[goal]] = aggr_line
        else:
            aggr_line = [DataNode(v, None, k, (AGGREGATION2ID[goal],)) for k, v in aggr_group.items()]
            aggr_line.sort(key=lambda x: x.left_coord)
            op_region.left_ids = [AGGREGATION2ID[goal]]
            op_region.left_aggr[AGGREGATION2ID[goal]] = aggr_line
        return list(map(lambda x: x.value, aggr_line))

    def group_by_index_name(self, data_nodes: List[DataNode], index_name: str):
        """ Group data nodes by index name."""
        _, direction, level = index_name.split('_')
        level = int(level)
        max_len_coord = len(data_nodes[0].left_coord)
        data_node_group = {}
        for data_node in data_nodes:
            if direction == '<LEFT>':
                coord = data_node.left_coord[:level] + tuple([-1] * (max_len_coord - level))
            else:
                coord = data_node.top_coord[:level] + tuple([-1] * (max_len_coord - level))
            if coord not in data_node_group:
                data_node_group[coord] = [data_node]
            else:
                data_node_group[coord].append(data_node)
        return data_node_group

    def calculate_aggregation(self, data_nodes, goal):
        """ Calculate aggregation value given a row/col of data nodes."""
        agg_value = 0
        value_line = [x.value for x in data_nodes if isinstance(x.value, int) or isinstance(x.value, float)]
        if len(value_line) == 0:
            raise NonNumericOperationError("Aggregation on non-numeric data.")
        if goal == 'max':
            agg_value = max(value_line)
        elif goal == 'min':
            agg_value = min(value_line)
        elif goal == 'avg':
            agg_value = mean(value_line)
        elif goal == 'sum':
            agg_value = sum(value_line)
        elif goal == 'std':
            agg_value = stdev(value_line)
        elif goal == 'var':
            agg_value = variance(value_line)
        elif goal == 'med':
            agg_value = median(value_line)
        return agg_value

    def get_new_ids_by_tree(self, coord_list, direction):
        """ Get new left/top ids using derived coord list."""
        new_ids = []
        for coord in coord_list:
            header = self.hmt.coord2left[coord] if direction == '<LEFT>' else self.hmt.coord2top[coord]
            new_ids.extend([_ for _ in range(header.start_idx, header.end_idx + 1)])
        return new_ids

    def update_branch_per_level(self, op_region, left_ids, top_ids):  # NOT USED
        """ Update branch per level after filtering."""
        left_coord_list = [self.hmt.data_region[left_id][0].left_coord \
                           for left_id in left_ids]
        top_coord_list = [self.hmt.data_region[0][top_id].top_coord \
                          for top_id in top_ids]

        left_branch_per_level = self.find_branch_per_level(left_coord_list)
        top_branch_per_level = self.find_branch_per_level(top_coord_list)

        new_branch_level = op_region.branch_per_level.copy()
        for i in range(len(left_branch_per_level)):
            iname = 'INAME_' + '<LEFT>_' + str(i)
            if iname not in op_region.branch_per_level:
                break
            new_branch_level[iname] = left_branch_per_level[i]
        for i in range(len(top_branch_per_level)):
            iname = 'INAME_' + '<TOP>_' + str(i)
            if iname not in op_region.branch_per_level:
                break
            new_branch_level[iname] = top_branch_per_level[i]
        return new_branch_level

    def find_branch_per_level(self, coord_list):  # NOT USED
        """ Find branch per level by coord list."""
        branch_per_level = [False]  # virtual node <LEFT> | <TOP> always not branch
        if len(coord_list) == 0:
            return branch_per_level
        for level in range(len(coord_list[0])):
            group_set = set()
            for coord in coord_list:
                if coord[level] != -1:
                    group_set.add(coord[level])
            branch_per_level.append(len(group_set) > 1)
        return branch_per_level


class NonNumericOperationError(Exception):
    """ Operation on non-numeric data cells."""

    def __init__(self, message):
        super().__init__(self)
        self.message = message

    def __str__(self):
        return self.message
