"""Hierarchical table classes. """

from typing import Dict, List, Tuple, Union
import copy
from transformers import BertTokenizer  # noqa
from icecream import ic
import time

from qa.nsm.execution.type_system import DateTime

AGGREGATION2ID = {
    "max": 1001,
    "min": 1011,
    "avg": 1021,
    "sum": 1031,
    "var": 1041,
    "std": 1051,
    'med': 1061
}
INVALID_TOKEN = '#'


class Region(object):
    """Current operating region."""

    def __init__(self,
                 left_ids: List[int],
                 top_ids: List[int],
                 branch_per_level: Dict[str, bool]
                 ):
        self.left_ids = left_ids
        self.top_ids = top_ids
        self.branch_per_level = branch_per_level  # NOT USED

        self.left_aggr = {}
        self.top_aggr = {}

    def __repr__(self):
        return f"Region=> left_ids: {self.left_ids}; top_ids: {self.top_ids}\n" \
            # f"          Branches per level: {self.branch_per_level}\n"

    __str__ = __repr__


class TreeNode(object):
    """Node instantces of hierarchical header tree (top or left)."""

    def __init__(
            self,
            name: str,  # cell string of the corresponding table cell
            value: Union[str, Tuple[float], Tuple[DateTime]],  # real value of the corresponding table cell,
            line_idx: int,  # line index of current node
            start_idx: int,  # start index of row/column in charge, inclusive
            end_idx: int,  # end index of row/column in charge, inclusive
            children: List['TreeNode'] = [],
            coord: Tuple = None,  # coordinate on the tree, e.g. (0, 1, -1, -1)
            direction: str = None,  # left/top. # NOT USED
            index_name: str = None,  # index name of header, maybe None
            cell_type: str = None,  # index name|index|value name   # NOT USED
            type: str = "string",  # string|num|datetime, used in bert input
            is_leaf: bool = False,  # indicate if is a leaf node
            **kwargs  # avoid error when init-dict has additional keys
    ):
        self.name = name
        self.value = value

        self.line_idx = line_idx
        self.start_idx = start_idx  # including the line_idx
        self.end_idx = end_idx
        self.children = children
        self.coord = coord
        self.direction = direction

        self.index_name = index_name
        self.cell_type = cell_type
        self.type = type
        self.is_leaf = is_leaf

    def add_child(self, child: 'TreeNode'):
        self.children.append(child)

    @classmethod
    def node_from_dict(cls, dict: Dict, dir: str = None) -> 'TreeNode':
        dict.update({"direction": dir})

        children_dicts = dict.get('children_dict', None)
        if children_dicts is None or len(children_dicts) == 0:  # leaf
            dict.update({"is_leaf": True, "children": [],
                         "line_idx": dict['line_idx'],
                         "start_idx": dict['line_idx'], "end_idx": dict['line_idx']})
        else:  # non-leaf
            children_nodes = []
            for child_dict in children_dicts:
                child_node = cls.node_from_dict(child_dict, dir)
                children_nodes.append(child_node)
            dict.update({"children": children_nodes})
            dict.update({"line_idx": dict['line_idx']})
            dict["start_idx"] = min([cn.start_idx for cn in children_nodes])
            dict["end_idx"] = max([cn.end_idx for cn in children_nodes])
            if dict['line_idx'] is not None:
                dict["start_idx"] = min(dict['start_idx'], dict['line_idx'])
        return cls(**dict)

    def traverse(self):
        """Depth first traversal of sub-tree, self node as the root."""
        candidates = [self]
        for child in self.children:
            child_candidates = child.traverse()
            candidates.extend(child_candidates)
        return candidates

    def traverse_level(self, title=None):
        """Level order traversal."""
        queue = [self]
        level_inames, level_header_info, level_line_idx, branch_per_level = [], [], [], []
        while len(queue) > 0:
            curr_level_len = len(queue)
            curr_level_iname, curr_level_header_names, curr_level_header_info, curr_level_line_idx = None, [], [], []
            for i in range(curr_level_len):
                node = queue.pop(0)
                if node.line_idx is not None:
                    curr_level_line_idx.append(node.line_idx)
                if node.name not in curr_level_header_names:
                    curr_level_header_names.append(node.name)
                    if node.name not in ['<LEFT>', '<TOP>']:
                        curr_level_header_info.append(dict(
                            name=node.name,
                            type=node.type,
                            value=node.value,
                            line_idx=node.line_idx
                        ))
                for child in node.children:
                    queue.append(child)
                if node.index_name and node.index_name != 'none' and curr_level_iname is None:
                    curr_level_iname = node.index_name
            level_inames.append(curr_level_iname)
            level_header_info.append(curr_level_header_info)
            level_line_idx.append(curr_level_line_idx)
            branch_per_level.append(len(curr_level_header_names) > 1)
        return level_inames, level_header_info, level_line_idx, branch_per_level

    def __eq__(self, other):
        if isinstance(other, TreeNode):
            return self.name == other.name
        else:
            return False

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Name: {self.name}, Value: {self.value}, Type: {self.type}, " \
               f"Index Name: {self.index_name}, Coords: {self.coord}, " \
               f"Start Idx: {self.start_idx}, End Idx: {self.end_idx}, " \
               f"Child Num: {len(self.children)}, Is Leaf: {self.is_leaf}"

    __str__ = __repr__


class DataNode(object):
    """Cells at the value (non-header) region, index-able by both top & left coords."""

    def __init__(
            self,
            value: Union[str, int, float, None],  # may be relevant to 'cast_type_func‘
            value_name: str = None,  # value name, e.g. "population" | "number"    NOT USED
            top_coord: Tuple = None,  # coord of its tracing top leaf node, i.e. topNode.coord
            left_coord: Tuple = None,  # coordinate of its sourcing left leaf node
            **kwargs  # avoid error due to additional dict keys
    ):
        self.value = value
        if self.value == INVALID_TOKEN:
            self.value = 0
        self.value_name = value_name
        self.top_coord = top_coord
        self.left_coord = left_coord

    @classmethod
    def node_from_dict(cls, dict: Dict) -> 'DataNode':
        return cls(**dict)

    def clone(self):
        return DataNode(self.value, self.value_name, self.top_coord, self.left_coord)

    def __repr__(self) -> str:
        return f"DATA NODE [Value: {self.value}\t" \
               f"top: {self.top_coord}\tleft: {self.left_coord}]"


class HMTable(object):
    """Hierarchical Matrix Table. """

    def __init__(
            self,
            title: str,  # table title
            top_root: TreeNode,  # top root node
            left_root: TreeNode,  # left root node
            data_region: List[List[DataNode]],  # matrix form of data region
            matrix_dict: Dict,  # matrix form of the whole table, including header and data region. for TUTA, NOT USED
            header2id: Dict,  # header name to unique id, to identify header in bert encoder and memory
            index_name2id: Dict,  # index name to unique id, used to identify index in bert encoder and memory
            header2level: Dict,  # header name to level
            left_line_idx2level: Dict,  # left line idx to level
            top_line_idx2level: Dict,  # top line idx to level
            index_name_map: Dict,
            # index name symbol to real index name, maybe no real name, e.g. {'INAME_<LEFT>_1': 'year'}
            index_name2header_info: Dict,  # index name to header_info on its level # used for tokenize and bert input
            branch_per_level: Dict,  # branches at each level. NOT USED
            top_coord_to_node_map: Dict = None,  # map top coords to tree nodes
            left_coord_to_node_map: Dict = None,  # map left coords to tree nodes
    ):
        self.title = title
        self.top_root = top_root
        self.left_root = left_root
        self.data_region = data_region

        self.header2id = header2id
        self.header2level = header2level
        self.index_name2id = index_name2id
        self.index_name_map = index_name_map
        self.index_name2header_info = index_name2header_info
        self.branch_per_level = branch_per_level
        self.left_line_idx2level = left_line_idx2level
        self.top_line_idx2level = top_line_idx2level

        self.init_tree_coords(self.top_root)
        self.init_tree_coords(self.left_root)

        self.matrix_dict = matrix_dict

        if top_coord_to_node_map:
            self.coord2top = top_coord_to_node_map
        else:
            self.coord2top = self.map_coords_to_tree_node(self.top_root)
        if left_coord_to_node_map:
            self.coord2left = left_coord_to_node_map
        else:
            self.coord2left = self.map_coords_to_tree_node(self.left_root)

        self.tokenized = False

    @classmethod
    def from_dict(cls, table_info: Dict):
        # hmt from dict
        hmt_dict = table_info
        top_root = TreeNode.node_from_dict(hmt_dict["top_root"], "top")
        left_root = TreeNode.node_from_dict(hmt_dict["left_root"], "left")
        data_region = []
        for data_row in hmt_dict["data"]:
            data_nodes = [DataNode.node_from_dict(d) for d in data_row]
            data_region.append(data_nodes)

        left_level_inames, left_level_header_info, left_level_line_idx, _ = left_root.traverse_level(hmt_dict['title'])
        top_level_inames, top_level_header_info, top_level_line_idx, _ = top_root.traverse_level(hmt_dict['title'])

        index_name2id, header2id, header2level = {}, {}, {}  # identifier for each index name and header
        index_name_map, index_name2header_info, left_line_idx2level, top_line_idx2level = {}, {}, {}, {}
        index_name_id, header_id = 0, 0
        for i in range(len(left_level_inames)):
            index_name_map['INAME_' + '<LEFT>_' + str(i)] = left_level_inames[i]
            index_name2header_info['INAME_' + '<LEFT>_' + str(i)] = left_level_header_info[i]
            # branch_per_level['INAME_' + '<LEFT>_' + str(i)] = left_branch_per_level[i]
            index_name2id['INAME_' + '<LEFT>_' + str(i)] = index_name_id
            index_name_id += 1
            for line_idx in left_level_line_idx[i]:  # deeper level will catch the line_idx
                left_line_idx2level[line_idx] = int(i)
            for header_info in left_level_header_info[i]:
                if header_info['name'] in header2id:  # avoid duplicate header in different levels
                    continue
                header2id[header_info['name']] = header_id
                header2level[header_info['name']] = int(i)
                header_id += 1
        for i in range(len(top_level_inames)):
            index_name_map['INAME_' + '<TOP>_' + str(i)] = top_level_inames[i]
            index_name2header_info['INAME_' + '<TOP>_' + str(i)] = top_level_header_info[i]
            # branch_per_level['INAME_' + '<TOP>_' + str(i)] = top_branch_per_level[i]
            index_name2id['INAME_' + '<TOP>_' + str(i)] = index_name_id
            index_name_id += 1
            for line_idx in top_level_line_idx[i]:
                top_line_idx2level[line_idx] = int(i)
            for header_info in top_level_header_info[i]:
                if header_info['name'] in header2id:  # avoid duplicate header in left/top.
                    continue
                header2id[header_info['name']] = header_id
                header2level[header_info['name']] = int(i)
                header_id += 1

        return cls(
                   title=hmt_dict['title'],
                   top_root=top_root,
                   left_root=left_root,
                   data_region=data_region,
                   header2id=header2id,
                   index_name2id=index_name2id,
                   header2level=header2level,
                   left_line_idx2level=left_line_idx2level,
                   top_line_idx2level=top_line_idx2level,
                   index_name_map=index_name_map,
                   index_name2header_info=index_name2header_info,
                   matrix_dict=None,
                   branch_per_level=dict()
                   )

    def init_tree_coords(self, node, node_coord=(-1, -1, -1, -1), depth=0):
        """Traverse a tree to initialize a coord for each node.
        Restrict a maximum depth of '3'.
        No degree restriction for now, i.e. each node can have arbitrary number of children.
        Default coordinate [-1, -1, -1] and depth = 0 for (null) roots.
        """
        if self.get_tree_depth(node) + depth - 1 > len(node_coord):
            raise TableInitError("Tree Coords initialization error due to exceeding depth.")
        if not node.coord:
            node.coord = node_coord
        for i, child in enumerate(node.children):  # non-leaf
            child.coord = node_coord[:depth] + (i,) + node_coord[depth + 1:]
            self.init_tree_coords(child, child.coord, depth + 1)
        if node.line_idx is not None:
            if node.direction == "top":
                column_idx = node.line_idx
                for data in [row[column_idx] for row in self.data_region]:
                    data.top_coord = node.coord
            elif node.direction == "left":
                row_idx = node.line_idx
                for data in self.data_region[row_idx]:
                    data.left_coord = node.coord

    def get_tree_depth(self, node):
        """Get the depth of current branch/sub-tree."""
        depth = 0
        for child in node.children:
            child_depth = self.get_tree_depth(child)
            depth = max(depth, child_depth)
        return depth + 1

    @staticmethod
    def map_coords_to_tree_node(root):
        coord_node_map = {}
        for node in root.traverse():
            coord_node_map.update({node.coord: node})
        return coord_node_map

    def iter_tree_nodes(self):
        """Iterate through header nodes on both trees, TOP first, then LEFT."""
        for tnode in self.top_root.traverse():
            yield tnode
        for lnode in self.left_root.traverse():
            yield lnode

    def build_kg_info(self) -> Dict:  # only used in random_explore.py
        header, num_header, datetime_header = [], [], []
        tree_nodes = self.iter_tree_nodes()
        for tree_node in tree_nodes:
            if tree_node.name in header:
                continue
            header.append(tree_node.name)
            if tree_node.type == "num":
                num_header.append(tree_node)
            elif tree_node.type == "datetime":
                datetime_header.append(tree_node)

        kg_info = {
            "kg": self,  # in table.jsonl, only 'kg' will be used by executor. Others are computed.
            "header": header,
            "num_header": num_header,
            "datetime_header": datetime_header,
            "index_name": self.index_name_map.keys()
        }
        return kg_info

    def get_init_op_region(self):
        """ Initialize operating region for computer and executor."""
        return Region([i for i in range(self.left_root.start_idx, self.left_root.end_idx + 1)],
                      [j for j in range(self.top_root.start_idx, self.top_root.end_idx + 1)],
                      self.branch_per_level.copy())

    def tokenize(self, tokenizer: BertTokenizer):
        """ Tokenize headers and return the HMT."""
        if self.tokenized:
            return self
        for index_name, level_header_info in self.index_name2header_info.items():
            for header_info in level_header_info:
                header_info['tokenized_name'] = tokenizer.tokenize(header_info['name'])
                header_info['tokenized_type'] = tokenizer.tokenize(header_info['type'])
                header_info['tokenized_value'] = tokenizer.tokenize(str(header_info['value']))
        for index_name, real_index_name in self.index_name_map.items():
            if real_index_name is not None and isinstance(real_index_name, str):
                self.index_name_map[index_name] = tokenizer.tokenize(real_index_name)
        self.tokenized = True
        return self

    def find_header_range(self, root, is_top, top2left_flag=[False]):
        """ Find the header rows/columns hierarchy. """
        max_rows, max_columns = root['RI'], root['CI']
        for child in root['Cd']:
            tmp_rows, tmp_columns = self.find_header_range(child, is_top, top2left_flag)
            if not is_top and max_columns > tmp_columns:  # put top header on the left tree
                top2left_flag[0] = True
                max_columns = tmp_columns
            max_rows = max(max_rows, tmp_rows)
            max_columns = max(max_columns, tmp_columns)
        if is_top:
            return max_rows, 0
        else:
            if top2left_flag[0]:
                return 1, max_columns
            else:
                return 0, max_columns

    def fill_matrix_dict(self):
        """ Fill matrix_dict with 'TopHeaderRowsNumber', 'LeftHeaderColumnsNumber' """
        left_max_header_rows, left_max_header_columns = self.find_header_range(self.matrix_dict['LeftTreeRoot'], is_top=False)
        top_max_header_rows, top_max_header_columns = self.find_header_range(self.matrix_dict['TopTreeRoot'], is_top=True)
        assert top_max_header_columns == 0 and (left_max_header_rows == 0 or left_max_header_rows == 1)
        if left_max_header_rows == 1:
            top_max_header_rows += 1
        self.matrix_dict['TopHeaderRowsNumber'] = top_max_header_rows + 1
        self.matrix_dict['LeftHeaderColumnsNumber'] = left_max_header_columns + 1

    def __repr__(self) -> str:
        data_info = '\n'.join([str(node) for row in self.data_region for node in row])
        top_map_msg = '{' + ' | '.join(f"{k}: {self.coord2top[k].name}" for k in self.coord2top) + '}'
        left_map_msg = '{' + ' | '.join(f"{k}: {self.coord2left[k].name}" for k in self.coord2left) + '}'
        top_msg = '\n'.join([str(node) for node in self.top_root.traverse()])
        left_msg = '\n'.join([str(node) for node in self.left_root.traverse()])
        return f"Size of Data Region: #row: {len(self.data_region)}, #column: {len(self.data_region[0])}\n" \
               f"\nData Info: {data_info}" \
               f"\nTop Map: {top_map_msg}\nLeft Map: {left_map_msg}\n" \
               f"\nTop Msg: \n{top_msg}\n\nLeft Msg: \n{left_msg}\n" \
               f"\nheader2id: {self.header2id}\n" \
               f"\nindex_name2id: {self.index_name2id}\n" \
               f"\nheader2level: {self.header2level}\n" \
               f"\nleft_line_idx2level: {self.left_line_idx2level}\n" \
               f"\ntop_line_idx2level: {self.top_line_idx2level}\n" \
               f"\nindex_name_map: {self.index_name_map}\n" \
               f"\ntitle: {self.title}\n"

    __str__ = __repr__


class TableInitError(Exception):
    """ Operation on non-numeric data cells."""

    def __init__(self, message):
        super().__init__(self)
        self.message = message

    def __str__(self):
        return self.message


def main():
    from qa.examples.example_data import KG_JULY_4
    table = HMTable.from_dict(KG_JULY_4)
    print(table)


if __name__ == "__main__":
    main()
