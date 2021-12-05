"""Computers can read in tokens, parse them into a program, and execute it."""

from __future__ import print_function

import json
from collections import OrderedDict
import re
import qa.nsm.data_utils as data_utils
import pprint
import traceback

from qa.table_bert.hm_table import *

END_TK = data_utils.END_TK  # End of program token
ERROR_TK = '<ERROR>'
# SPECIAL_TKS = [END_TK, ERROR_TK, '(', ')']
SPECIAL_TKS = [ERROR_TK, '(', ')']


class LispInterpreter(object):
    """Interpreter reads in tokens, parse them into a program and execute it."""

    def __init__(self, op_region, type_hierarchy, max_mem, max_n_exp, hmt: HMTable, assisted=True):
        """
        max_mem: maximum number of memory slots.  60, num of variables
        max_n_exp: maximum number of expressions.  3
        assisted: whether to provide assistance to the programmer (used for neural programmer).
        """
        # Create namespace.
        self.namespace = Namespace()  # memory+constant+function

        self.assisted = assisted
        # Configs.
        # Functions used to call
        # Signature: autocomplete(evaled_exp, valid_tokens, evaled_tokens)
        # return a subset of valid_tokens that passed the
        # filter. Used to implement filtering with denotation.
        self.op_region = op_region

        self.type_hierarchy = type_hierarchy
        self.type_ancestry = create_type_ancestry(type_hierarchy)

        self.max_mem = max_mem
        self.max_n_exp = max_n_exp

        self.hmt = hmt

        # Initialize the parser state.
        self.history = []  # decode tokens
        self.exp_stack = []  # expression stack
        self.done = False  # if program ends
        self.result = None  # execution result of the program
        self.selected_region = None
        self.selected_ops = []
        self.selected_headers = []

        # Program constraints
        self.pc = ProgramConstraint()

    @property
    def primitive_names(self):  # not used
        primitive_names = []
        for k, v in self.namespace.iteritems():
            if ('property' in self.type_ancestry[v['type']] or
                    'primitive_function' in self.type_ancestry[v['type']]):
                primitive_names.append(k)
        return primitive_names

    @property
    def primitives(self):  # not used
        primitives = []
        for k, v in self.namespace.iteritems():
            if ('property' in self.type_ancestry[v['type']] or
                    'primitive_function' in self.type_ancestry[v['type']]):
                primitives.append(v)
        return primitives

    def add_constant(self, value, type, line_idx=None, name=None):
        """Generate the code and variables to hold the constants."""
        if name is None:
            name = self.namespace.generate_new_name()
        self.namespace[name] = dict(
            value=value, type=type, is_constant=True)
        return name

    def add_function(self, name, value, args, return_type,
                     autocomplete, type):
        """Add function into the namespace."""
        if name in self.namespace:
            raise ValueError('Name %s is already used.' % name)
        else:
            self.namespace[name] = dict(
                value=value, type=type,
                autocomplete=autocomplete,
                return_type=return_type, args=args)

    # def init_executor_op_region(self):
    #     """ Initialize operating region for executor."""
    #     self.namespace['init_op_region']['value'](self.op_region)

    def autocomplete(self, exp, exp_vals, tokens, token_vals, namespace):
        function = exp_vals[0]

        return function['autocomplete'](self.op_region, exp, exp_vals, tokens, token_vals, self.pc)

    def valid_functions(self, tokens, token_vals):
        """ Remove invalid functions after generating '(' to prune search space. """
        # TODO: use "type" to refine the constraints.
        valid_tokens = tokens

        if self.pc.n_filter_top_level > 0:  # Operation Stage
            self.safe_remove_tk(valid_tokens, 'filter_tree_str_contain')
            self.safe_remove_tk(valid_tokens, 'filter_tree_str_not_contain')
            self.safe_remove_tk(valid_tokens, 'filter_level')
            if self.pc.n_aggr == 0:  # no aggregation yet, focus on op_region
                if (len(self.op_region.left_ids) > 1 and len(self.op_region.top_ids) > 1) \
                        or (len(self.op_region.left_ids) == 1 and len(self.op_region.top_ids) == 1):
                    self.safe_remove_tk(valid_tokens, 'argmax')
                    self.safe_remove_tk(valid_tokens, 'argmin')
                    self.safe_remove_tk(valid_tokens, 'rank')
                    self.safe_remove_tk(valid_tokens, 'argrank')
                if len(self.op_region.left_ids) > 1 and len(self.op_region.top_ids) > 1:
                    self.safe_remove_tk(valid_tokens, 'opposite')
                if not ((len(self.op_region.left_ids) == 1 and len(self.op_region.top_ids) == 2)
                        or (len(self.op_region.left_ids) == 2 and len(self.op_region.top_ids) == 1)):
                    self.safe_remove_tk(valid_tokens, 'difference')
                    self.safe_remove_tk(valid_tokens, 'difference_rate')
                    self.safe_remove_tk(valid_tokens, 'proportion')
                if len(self.op_region.left_ids) == 1 and len(self.op_region.top_ids) == 1:
                    self.safe_remove_tk(valid_tokens, 'max')
                    self.safe_remove_tk(valid_tokens, 'min')
                    self.safe_remove_tk(valid_tokens, 'sum')
                    self.safe_remove_tk(valid_tokens, 'average')
            else:  # aggregation yet, focus on left_aggr/top_aggr
                valid_tokens = ['argmax', 'argmin', 'difference', 'difference_rate', 'proportion',
                                'rank', 'argrank', 'opposite']
                aggr_map = self.op_region.left_aggr if len(self.op_region.left_aggr) > 0 else self.op_region.top_aggr
                num_aggr_group = 0
                for aggr_id, aggr_line in aggr_map.items():  # only one aggr_line in practice
                    num_aggr_group = len(aggr_line)
                    break
                if num_aggr_group == 1:
                    self.safe_remove_tk(valid_tokens, 'argmax')
                    self.safe_remove_tk(valid_tokens, 'argmin')
                    self.safe_remove_tk(valid_tokens, 'rank')
                    self.safe_remove_tk(valid_tokens, 'argrank')
                if num_aggr_group != 2:
                    self.safe_remove_tk(valid_tokens, 'difference')
                    self.safe_remove_tk(valid_tokens, 'difference_rate')
                    self.safe_remove_tk(valid_tokens, 'proportion')
        else:  # Filter Stage
            valid_tokens = ['filter_tree_str_contain', 'filter_tree_str_not_contain', 'filter_level']
            # control number of filter_tree
            if self.pc.n_filter_left_tree == self.pc.MAX_N_FILTER_TREE_LEFT and self.pc.n_filter_left_level == 0 \
                    or self.pc.n_filter_top_tree == self.pc.MAX_N_FILTER_TREE_TOP:
                valid_tokens = ['filter_level']
            # when filter the max depth tree, no more filter tree
            max_left_depth = self.hmt.get_tree_depth(self.hmt.left_root) - 1
            max_top_depth = self.hmt.get_tree_depth(self.hmt.top_root) - 1
            if self.pc.cur_left_level == max_left_depth and self.pc.n_filter_left_level == 0 \
                    or self.pc.cur_top_level == max_top_depth:
                valid_tokens = ['filter_level']
        return valid_tokens

    def reset(self, only_reset_variables=False):
        """Reset all the interpreter state."""
        if only_reset_variables:
            self.namespace.reset_variables()
        else:
            self.namespace = Namespace()
        self.history = []

        self.exp_stack = []
        self.done = False
        self.result = None
        self.selected_region = None
        self.selected_ops = []
        self.selected_headers = []

        self.pc = ProgramConstraint()


    def read_token_id(self, token_id):
        token = self.rev_vocab[token_id]
        return self.read_token(token)

    def read_token(self, token):
        """Read in one token, parse and execute the expression if completed."""
        if ((self.pc.n_exp >= self.max_n_exp) or
                (self.namespace.n_var >= self.max_mem) or
                (self.pc.n_op >= 2) or
                (self.pc.n_return >= 1)):
            token = END_TK
        new_exp = self.parse_step(token)
        # If reads in end of program, then return the last value as result.
        if token == END_TK:
            self.done = True
            self.result = self.namespace.get_last_value()  # if Region, return current op_region instead of get_last_value()
            if isinstance(self.result, Region):
                self.result = self.op_region
            if self.pc.n_exp > 0 and self.history[-2] == '(':
                self.history.pop(-2)
            return self.result
        elif new_exp:  # one exp ends
            if self.assisted:
                name = self.namespace.generate_new_name()  # v_i
                result = self.eval(['define', name, new_exp])
                # If there are errors in the execution, self.eval
                # will return None. We can also give a separate negative
                # reward for errors.
                if result is None:
                    self.namespace.n_var -= 1
                    self.done = True
                    self.result = [ERROR_TK]
            else:
                result = self.eval(new_exp)
            return result
        else:
            return None

    def valid_tokens(self):
        """Return valid tokens for the next step for programmer to pick."""
        # If already exceeded max memory or max expression
        # limit, then must end the program.
        result = []
        if ((self.pc.n_exp >= self.max_n_exp) or
                (self.namespace.n_var >= self.max_mem) or
                (self.pc.n_op >= 2) or
                (self.pc.n_return >= 1)):
            result = [END_TK]
        # If last expression is finished, either start a new one
        # or end the program.
        elif not self.history:
            result = ['(']
        # If not in an expression, either start a new expression or end the program.
        elif not self.exp_stack:
            if self.pc.n_filter_top_level > 0:   # have to be after filter_level left/top.
                result = ['(', END_TK]
            else:
                result = ['(']
        # If currently in an expression.
        else:
            exp = self.exp_stack[-1]
            # If in the middle of a new expression.
            if exp:
                # Use number of arguments to check if all arguments are there.
                head = exp[0]
                args = self.namespace[head]['args']
                pos = len(exp) - 1
                if self.namespace[head]['type'] == 'filter_tree_function':  # arg nums uncertain
                    if pos == len(args) + 1:  # at most two headers union
                        result = [')']
                else:
                    if pos == len(args):  # normal program ends
                        result = [')']
                if result != [')']:
                    if self.namespace[head]['type'] == 'filter_tree_function':
                        if pos == len(args):  # the second header is the same as the first
                            pos = pos - 1
                    result = self.namespace.valid_tokens(  # step1，filter with type hierarchy
                        args[pos], self.get_type_ancestors)
                    if self.autocomplete is not None:  # step2, filter with autocomplete
                        valid_tokens = result
                        evaled_exp = [self.eval(item) for item in exp]
                        evaled_tokens = [self.eval(tk) for tk in valid_tokens]
                        result = self.autocomplete(  # 用autocomplete计算valid tokens
                            exp, evaled_exp, valid_tokens, evaled_tokens, self.namespace)
            # If at the beginning of a new expression, select function token
            else:
                result = self.namespace.valid_tokens(
                    {'types': ['function']}, self.get_type_ancestors)
                valid_tokens = result
                evaled_tokens = [self.eval(tk) for tk in valid_tokens]
                result = self.valid_functions(valid_tokens, evaled_tokens)  # filter functions
        return result

    def parse_step(self, token):
        """Run the parser for one step with given token which parses tokens into expressions."""
        self.history.append(token)
        if token == END_TK:
            self.done = True
        elif token == '(':
            self.exp_stack.append([])
        elif token == ')':
            # One list is finished.
            new_exp = self.exp_stack.pop()
            if self.exp_stack:
                self.exp_stack[-1].append(new_exp)
            else:
                self.exp_stack = []
                return new_exp
        elif self.exp_stack:
            self.exp_stack[-1].append(token)
        else:
            # Atom expression.
            return token

    def tokenize(self, chars):
        """Convert a string of characters into a list of tokens."""
        return chars.replace('(', ' ( ').replace(')', ' ) ').split()

    def get_type_ancestors(self, type):
        return self.type_ancestry[type]

    def infer_type(self, return_type, arg_types):
        """Infer the type of the returned value of a function."""
        if hasattr(return_type, '__call__'):
            return return_type(*arg_types)
        else:
            return return_type

    def eval(self, x, namespace=None):
        """Another layer above _eval to handle exceptions."""
        try:
            result = self._eval(x, namespace)
        except Exception as e:
            # print('Error in eval(): ', e)
            # # traceback.print_exc()
            # exc_type, exc_obj, exc_tb = sys.exc_info()
            # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # print(exc_type, fname, exc_tb.tb_lineno)
            # print('when evaluating ', x)
            # print(self.history)
            # print(self.op_region)
            # pprint.pprint(self.namespace)
            result = None
            # raise e
        return result

    def _eval(self, x, namespace=None):
        """Evaluate an expression in an namespace."""
        if namespace is None:
            namespace = self.namespace
        if is_symbol(x):  # variable reference name
            return namespace.get_object(x).copy()
        elif x[0] == 'define':  # (define name exp)
            (_, name, exp) = x
            obj = self._eval(exp, namespace)
            namespace[name] = obj  # save intermediate result into namespace
            return obj
        else:
            # Execute a function.
            proc = self._eval(x[0], namespace)
            args = [self._eval(exp, namespace) for exp in x[1:]]
            arg_values = [arg['value'] for arg in args]
            value = proc['value'](self.op_region, *(arg_values))
            arg_types = [arg['type'] for arg in args]
            type = self.infer_type(proc['return_type'], arg_types)
            self.update_program_constraints(proc, args)
            self.update_selected_region_and_ops(proc, args, value, type)
            return {'value': value, 'type': type, 'is_constant': False}

    def step(self, token):
        """Open AI gym inferface."""
        result = self.read_token(token)
        observation = token
        reward = 0.0
        done = self.done
        if (result is None) or self.done:
            write_pos = None
        else:
            write_pos = self.namespace.n_var - 1

        info = {'result': result,
                'write_pos': write_pos}
        return observation, reward, done, info

    def get_last_var_loc(self):
        return self.namespace.n_var - 1

    def interactive(self, prompt='> ', assisted=True):
        """A prompt-read-eval-print loop."""
        self.assisted = assisted
        while True:
            query = input(prompt).strip()
            tokens = self.tokenize(query)
            for tk in tokens:
                result = self.read_token(tk)
                print('Read in [{}], valid tokens: {}'.format(tk, self.valid_tokens()))
                if result:
                    print('Result: ', result)

    def has_extra_work(self):
        """ Modified for hmt. Results are usually not used, so currently not extra work."""
        # TODO: define hmt extra work. e.g. len(program) > len(gold); filter unchanged region
        return False

    def clone(self):
        """Make a copy of itself, used in sample/beam search"""
        new = LispInterpreter(
                            copy.deepcopy(self.op_region),
                            self.type_hierarchy,
                            self.max_mem,
                            self.max_n_exp,
                            self.hmt,
                            self.assisted)

        new.history = self.history[:]
        new.exp_stack = copy.deepcopy(self.exp_stack)
        new.namespace = self.namespace.clone()
        new.selected_region = copy.deepcopy(self.selected_region)
        new.selected_ops = copy.deepcopy(self.selected_ops)
        new.selected_headers = copy.deepcopy(self.selected_headers)

        new.pc = copy.deepcopy(self.pc)
        return new

    def get_vocab(self):
        mem_tokens = []
        for i in range(self.max_mem):
            mem_tokens.append('v{}'.format(i))
        vocab = data_utils.Vocab(
            list(self.namespace.get_all_names()) + SPECIAL_TKS + mem_tokens)
        return vocab

    def update_program_constraints(self, proc, args):
        """ Update program constraint variables when executing a new exp."""
        self.pc.n_exp += 1
        if proc['type'] == 'filter_tree_function':
            _, direction, _ = args[0]['value'].split('_')
            if direction == '<LEFT>':
                self.pc.n_filter_left_tree += 1
                self.pc.cur_left_level = int(self.hmt.header2level[args[1]['value']])
            else:
                self.pc.n_filter_top_tree += 1
                self.pc.cur_top_level = int(self.hmt.header2level[args[1]['value']])
        elif proc['type'] == 'filter_level_function':
            _, direction, level = args[0]['value'].split('_')
            level = int(level)
            if direction == '<LEFT>':
                self.pc.n_filter_left_level += 1
                self.pc.cur_left_level = level
            else:
                self.pc.n_filter_top_level += 1
                self.pc.cur_top_level = level
        elif proc['type'] in ['aggregation_function', 'return_function']:
            self.pc.n_op += 1
            if proc['type'] == 'aggregation_function':
                self.pc.n_aggr += 1
            if proc['type'] == 'return_function':
                self.pc.n_return += 1

    def update_selected_region_and_ops(self, proc, args, value, type):
        """ Update selected region when filtering phase is over. And update selected operations in operating phase."""
        # filtering phase ends, update selected region
        if proc['type'] == 'filter_level_function' and self.pc.n_filter_top_level > 0:
            self.selected_region = value
        # operating phase
        if 'operation_function' in self.type_ancestry[proc['type']]:
            for k, v in self.namespace.items():
                if v['value'] == proc['value']:
                    self.selected_ops.append(k)
        # filtering phase, filter tree with headers
        if proc['type'] == 'filter_tree_function':
            if len(args) == 2:
                self.selected_headers.extend([args[1]['value']])
            elif len(args) == 3:
                self.selected_headers.extend([args[1]['value'], args[2]['value']])

    def safe_remove_tk(self, valid_tks, tk):
        """ Remove tk from valid_tks if tk in valid tks."""
        try:
            valid_tks.remove(tk)
        except:
            pass


class Namespace(OrderedDict):
    """Namespace is a mapping from names to values.

  Namespace maintains the mapping from names to their
  values. It also generates new variable names for memory
  slots (v0, v1...), and support finding a subset of
  variables that fulfill some type constraints, (for
  example, find all the functions or find all the entity
  lists).
  """

    def __init__(self, *args, **kwargs):
        """Initialize the namespace with a list of functions."""
        super(Namespace, self).__init__(*args, **kwargs)
        self.n_var = 0
        self.last_var = None

    def clone(self):
        new = Namespace(self)
        new.n_var = self.n_var
        new.last_var = self.last_var
        return new

    def clone_and_reset(self):
        copy = self.clone()
        copy.reset_variables()

        return copy

    def generate_new_name(self):
        """Create and return a new variable."""
        name = 'v{}'.format(self.n_var)
        self.last_var = name
        self.n_var += 1
        return name

    def valid_tokens(self, constraint, get_type_ancestors):
        """Return all the names/tokens that fulfill the constraint."""
        return [k for k, v in self.items()
                if self._is_token_valid(v, constraint, get_type_ancestors)]  # constraint: {'types': 'head'}

    def _is_token_valid(self, token, constraint, get_type_ancestors):
        """Determine if the token fulfills the given constraint."""
        type = token['type']  # e.g. "primitive_function"
        return set(get_type_ancestors(type) + [type]).intersection(constraint['types'])

    def get_value(self, name):
        return self[name]['value']

    def get_object(self, name):
        return self[name]

    def get_last_value(self):
        if self.last_var is None:
            return None
        else:
            return self.get_value(self.last_var)

    def get_all_names(self):
        return self.keys()

    def reset_variables(self):
        keys = list(self.keys())
        for k in keys:
            if re.match(r'v\d+', k):
                del self[k]
        self.n_var = 0
        self.last_var = None


class ProgramConstraint(object):
    """ Program Constraint is some constraints that control valid tokens,
    about both valid functions and valid arguments.
    Used in valid_tokens(), valid_functions(), and autocomplete()."""

    def __init__(self):
        # records
        self.n_exp = 0
        self.n_filter_left_tree = 0
        self.n_filter_left_level = 0
        self.n_filter_top_tree = 0
        self.n_filter_top_level = 0
        self.n_op = 0
        self.n_aggr = 0
        self.n_return = 0

        self.cur_left_level = 0
        self.cur_top_level = 0

        # constants
        self.MAX_N_FILTER_TREE_LEFT = 3
        self.MAX_N_FILTER_TREE_TOP = 3
        self.MAX_N_OP = 2

    def __repr__(self):
        return f"n_exp: {self.n_exp}; " \
               f"n_filter_left_tree: {self.n_filter_left_tree}; n_filter_top_tree: {self.n_filter_top_tree}"\
               f"n_filter_left_level: {self.n_filter_left_level}; n_filter_top_level: {self.n_filter_left_level}"\
               f"cur_left_level: {self.cur_left_level}; cur_top_level: {self.cur_top_level}"

    __str__ = __repr__


def is_symbol(x):
    return isinstance(x, str)


def create_type_ancestry(type_tree):
    type_ancestry = {}
    try:
        for type, _ in type_tree.items():
            _get_type_ancestors(type, type_tree, type_ancestry)
    except:
        pass
    return type_ancestry


def _get_type_ancestors(type, type_hrchy, type_ancestry):
    """Compute the ancestors of a type with memorization."""
    if type in type_ancestry:
        return type_ancestry[type]
    else:
        parents = type_hrchy[type]
        result = parents[:]
        for p in parents:
            ancestors = _get_type_ancestors(p, type_hrchy, type_ancestry)
            for a in ancestors:
                if a not in result:
                    result.append(a)
        type_ancestry[type] = result
        return result
