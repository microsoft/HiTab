from typing import List, Dict, Any, Union
import collections
import pprint
import numpy as np
import functools
import bloom_filter
import torch

import qa.nsm.computer as computer_factory
from qa.table_bert.hm_table import Region
from qa.nsm import AGGR_MAP
from qa.datadump.utils import linked_cell_compare


class Observation(object):
    def __init__(self, read_ind, write_ind, valid_action_indices, output_features=None, valid_action_mask=None):
        self.read_ind = read_ind
        self.write_ind = write_ind
        self.valid_action_indices = valid_action_indices
        self.output_features = output_features
        self.valid_action_mask = valid_action_mask

    def to(self, device: torch.device):
        if self.read_ind.device == device:
            return self

        self.read_ind = self.read_ind.to(device)
        self.write_ind = self.write_ind.to(device)
        if self.valid_action_indices is not None:
            self.valid_action_indices = self.valid_action_indices.to(device)
        self.output_features = self.output_features.to(device)
        self.valid_action_mask = self.valid_action_mask.to(device)

        return self

    def slice(self, t: int):
        return Observation(self.read_ind[:, t],
                           self.write_ind[:, t],
                           None,
                           self.output_features[:, t],
                           self.valid_action_mask[:, t])

    def remove_action(self, action_id):
        # action_rel_id is id of valid action list, action_id is id vocab
        action_rel_id = self.valid_action_indices.index(action_id)
        del self.valid_action_indices[action_rel_id]
        if self.output_features:
            del self.output_features[action_rel_id]

    @staticmethod
    def empty():
        """create an empty observation for padding"""
        return Observation(0, -1, [], [])

    @staticmethod
    def get_valid_action_masks(obs: List['Observation'], memory_size):
        batch_size = len(obs)

        # initialize valid action mask
        valid_action_mask = torch.zeros(batch_size, memory_size)
        for i, observation in enumerate(obs):
            valid_action_mask[i, observation.valid_action_indices] = 1.

        return valid_action_mask

    @staticmethod
    def to_batched_input(obs: List['Observation'], memory_size) -> 'Observation':
        batch_size = len(obs)

        read_ind = torch.tensor([ob.read_ind for ob in obs])
        write_ind = torch.tensor([ob.write_ind for ob in obs])

        # pad output features
        feat_num = len(obs[0].output_features[0])  # 1
        output_feats = np.zeros((batch_size, memory_size, feat_num), dtype=np.float32)
        valid_action_mask = torch.zeros(batch_size, memory_size)

        for i, observation in enumerate(obs):
            if observation.valid_action_indices:
                output_feats[i, observation.valid_action_indices] = observation.output_features
                valid_action_mask[i, observation.valid_action_indices] = 1.

        output_feats = torch.from_numpy(output_feats)

        return Observation(read_ind, write_ind, None, output_feats, valid_action_mask)

    @staticmethod
    def to_batched_sequence_input(obs_seq: List[List['Observation']], memory_size) -> 'Observation':
        batch_size = len(obs_seq)
        seq_len = max(len(ob_seq) for ob_seq in obs_seq)

        read_ind = torch.zeros(batch_size, seq_len, dtype=torch.long)
        write_ind = torch.zeros(batch_size, seq_len, dtype=torch.long).fill_(-1.)
        valid_action_mask = torch.zeros(batch_size, seq_len, memory_size)
        feat_num = len(obs_seq[0][0].output_features[0])
        output_feats = np.zeros((batch_size, seq_len, memory_size, feat_num), dtype=np.float32)

        for batch_id in range(batch_size):
            ob_seq_i = obs_seq[batch_id]
            for t in range(len(ob_seq_i)):
                ob = obs_seq[batch_id][t]
                read_ind[batch_id, t] = ob.read_ind
                write_ind[batch_id, t] = ob.write_ind

                valid_action_mask[batch_id, t, ob.valid_action_indices] = 1.
                output_feats[batch_id, t, ob.valid_action_indices] = ob.output_features

        output_feats = torch.from_numpy(output_feats)

        return Observation(read_ind, write_ind, None, output_feats, valid_action_mask)

    def __repr__(self):
        return f'Observation(read_id={repr(self.read_ind)}, write_id={repr(self.write_ind)}, ' \
            f'valid_actions={repr(self.valid_action_indices)})'

    __str__ = __repr__


class Trajectory(object):
    """ A neat record of 'Environment'. """
    def __init__(self, environment_name: str,
                 observations: List[Observation],
                 context: Dict,
                 tgt_action_ids: List[int],
                 answer: Any,
                 reward: float,
                 program: List[str] = None,
                 human_readable_program: List[str] = None,
                 id: str = None):
        self.id = id
        self.environment_name = environment_name

        self.observations = observations
        self.context = context
        self.tgt_action_ids = tgt_action_ids
        self.answer = answer
        self.reward = reward
        self.program = program
        self.human_readable_program = human_readable_program

        self._hash = hash((self.environment_name, ' '.join(str(a) for a in self.tgt_action_ids)))

    def __hash__(self):
        return self._hash

    def __repr__(self):
        if self.human_readable_program:
            return ' '.join(self.human_readable_program)

        elif self.program:
            return ' '.join(self.program)

        return '[Undecoded Program] ' + ' '.join(map(str, self.tgt_action_ids))

    __str__ = __repr__

    @classmethod
    def from_environment(cls, env):
        return Trajectory(
            env.name,
            observations=env.obs,
            context=env.get_context(),
            tgt_action_ids=env.mapped_actions,
            answer=env.interpreter.result,
            reward=env.rewards[-1],
            program=env.program,
            human_readable_program=env.to_human_readable_program()
        )

    @classmethod
    def from_program(cls, env, program):
        env = env.clone()
        env.use_cache = False
        ob = env.start_ob

        for token in program:
            action_id = env.de_vocab.lookup(token)
            rel_action_id = ob.valid_action_indices.index(action_id)
            ob, _, _, _ = env.step(rel_action_id)
        trajectory = Trajectory.from_environment(env)

        return trajectory

    @classmethod
    def to_batched_sequence_tensors(cls, trajectories: List['Trajectory'], memory_size):
        batch_size = len(trajectories)

        obs_seq = [traj.observations for traj in trajectories]
        max_seq_len = max(len(ob_seq) for ob_seq in obs_seq)

        batched_obs_seq = Observation.to_batched_sequence_input(obs_seq, memory_size=memory_size)

        tgt_action_ids = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        tgt_action_mask = torch.zeros(batch_size, max_seq_len)
        for batch_id in range(batch_size):
            traj_i_action_ids = trajectories[batch_id].tgt_action_ids
            tgt_action_ids[batch_id, :len(traj_i_action_ids)] = traj_i_action_ids
            tgt_action_mask[batch_id, :len(traj_i_action_ids)] = 1.

        tgt_action_ids = torch.from_numpy(tgt_action_ids)

        return batched_obs_seq, dict(tgt_action_ids=tgt_action_ids, tgt_action_mask=tgt_action_mask)


class Environment(object):
    """Environment with OpenAI Gym like interface."""

    def step(self, action):
        """
        Args:
          action: an action to execute against the environment.

        Returns:
          observation:
          reward:
          done:
          info:
        """
        raise NotImplementedError


# Use last action and the new variable's memory location as input.
ProgrammingObservation = collections.namedtuple(
    'ProgramObservation', ['last_actions', 'output', 'valid_actions'])


class QAProgrammingEnv(Environment):
    """
    An RL environment wrapper around an interpreter to
    learn to write programs based on question.
    """

    def __init__(self,
                 question_annotation,
                 kg,
                 answer,
                 score_fn, interpreter,
                 de_vocab=None, constants=None,
                 punish_extra_work=True,
                 init_interp=True, trigger_words_dict=None,
                 max_cache_size=1e4,
                 context=None, id_feature_dict=None,
                 cache=None,
                 reset=True,
                 name='qa_programming',
                 alpha_region=0.,
                 alpha_op=0.,
                 alpha_entity_link=0.):

        self.name = name
        self.de_vocab = de_vocab or interpreter.get_vocab()
        self.end_action = self.de_vocab.end_id
        self.score_fn = score_fn
        self.interpreter = interpreter
        self.answer = answer
        self.question_annotation = question_annotation
        self.kg = kg
        self.constants = constants
        self.punish_extra_work = punish_extra_work
        self.error = False
        self.trigger_words_dict = trigger_words_dict
        self.alpha_region = alpha_region
        self.alpha_op = alpha_op
        self.alpha_entity_link = alpha_entity_link
        tokens = question_annotation['tokens']

        self.n_builtin = len(self.de_vocab.vocab) - interpreter.max_mem
        self.n_mem = interpreter.max_mem  # 60
        self.n_exp = interpreter.max_n_exp  # 3
        max_n_constants = self.n_mem - self.n_exp

        self.overflow = False

        if context:
            self.context = context
        else:
            # initialize constants to be used in the interpreter

            constant_spans = []
            constant_values = []
            if constants is None:
                constants = []
            for c in constants:  # props
                constant_spans.append([-1, -1])
                constant_values.append(c['value'])
                if init_interp:
                    self.interpreter.add_constant(
                        value=c['value'], type=c['type'])

            for entity in question_annotation['entities']:  # question entity
                constant_spans.append(
                    [entity['token_start'], entity['token_end'] - 1])  # entity position in question
                constant_values.append(entity['value'])

                if init_interp:
                    self.interpreter.add_constant(
                        value=entity['value'], type=entity['type'])

            constant_spans = constant_spans[:max_n_constants]

            if len(constant_values) > (self.n_mem - self.n_exp):
                 print('Not enough memory slots for example {}, which has {} constants.'.format(
                    self.name, len(constant_values)))
                 self.overflow = True

            self.context = dict(
                constant_spans=constant_spans,
                question_features=question_annotation['features'],
                question_tokens=tokens,
                table=question_annotation['table'] if 'table' in question_annotation else None

            )

        # Create output features.
        if id_feature_dict:
            self.id_feature_dict = id_feature_dict
        else:
            prop_features = question_annotation['prop_features']  # indicate if a 'prop' appears in the question
            feat_num = len(list(prop_features.values())[0])  # 1
            self.id_feature_dict = {}
            for name, id in self.de_vocab.vocab.items():
                self.id_feature_dict[id] = [0] * feat_num
                if name in self.interpreter.namespace:
                    val = self.interpreter.namespace[name]['value']
                    if (isinstance(val, str)) and val in prop_features:
                        self.id_feature_dict[id] = prop_features[val]
        self.context['id_feature_dict'] = self.id_feature_dict

        if 'original_tokens' in self.context:
            self.context['original_tokens'] = question_annotation['original_tokens']

        self.entity_link_actions, self.entity_link_tokens = \
            self.get_entity_link_action_indices_and_tokens(self.question_annotation['linked_cells']['entity_link'])
        self.data_cells = self.get_data_cells(self.question_annotation['linked_cells']['quantity_link'])
        self.aggregation = self.question_annotation['aggregation']
        self.aggregation_actions = self.get_aggregation_indices(self.question_annotation['aggregation'])

        if cache:
            self.cache = cache
        else:
            self.cache = SearchCache(name=name, max_elements=max_cache_size)

        self.use_cache = False

        if reset:
            self.reset()

    def get_context(self):
        return self.context

    def step(self, action, debug=False):
        self.actions.append(action)
        if debug:
            print('-' * 50)
            print(f"actions: {self.actions}")
            print(self.de_vocab.lookup(self.valid_actions, reverse=True))
            print('pick #{} valid action'.format(action))
            print('history:')
            print(self.de_vocab.lookup(self.mapped_actions, reverse=True))
            print('env: {}, cache size: {}'.format(self.name, -1))
            print('obs')
            pprint.pprint(self.obs)

        if 0 <= action <= len(self.valid_actions):
            mapped_action = self.valid_actions[action]
        else:
            print('-' * 50)
            print('action out of range.')
            print('action:')
            print(action)
            print('valid actions:')
            print(self.de_vocab.lookup(self.valid_actions, reverse=True))  # id2token
            print('pick #{} valid action'.format(action))
            print('history:')
            print(self.de_vocab.lookup(self.mapped_actions, reverse=True))
            print('obs')
            pprint.pprint(self.obs)
            print('-' * 50)
            mapped_action = self.valid_actions[action]

        self.mapped_actions.append(mapped_action)
        mapped_action_token = self.de_vocab.lookup(mapped_action, reverse=True)
        self.program.append(mapped_action_token)

        result = self.interpreter.read_token(mapped_action_token)

        self.done = self.interpreter.done
        # Only when the program is finished and it doesn't have
        # extra work or we don't care, its result will be
        # scored, and the score will be used as reward.
        if self.done and not (self.punish_extra_work and self.interpreter.has_extra_work()):
            # try:
                if isinstance(self.interpreter.result, Region):  # Region presented as data region value in answer
                    left_ids, top_ids = self.interpreter.result.left_ids, self.interpreter.result.top_ids
                    self.interpreter.result = [[self.kg['kg'].data_region[left_id][top_id].value
                                                for top_id in top_ids]
                                               for left_id in left_ids]
                reward_region = self.calculate_region_reward(self.interpreter.selected_region, self.data_cells)
                reward_op = self.calculate_op_reward(self.interpreter.selected_ops, self.aggregation)
                reward_entity_link = self.calculate_entity_link_reward(self.interpreter.selected_headers, self.entity_link_tokens)
                reward_answer = self.calculate_answer_reward(self.interpreter.result, self.answer)
                reward = reward_answer \
                         + self.alpha_region * reward_region \
                         + self.alpha_op * reward_op \
                         + self.alpha_entity_link * reward_entity_link
        else:
            reward = 0.0

        if self.done and self.interpreter.result == [computer_factory.ERROR_TK]:
            self.error = True

        if result is None or self.done:
            new_var_id = -1
        else:
            new_var_id = self.de_vocab.lookup(self.interpreter.namespace.last_var)

        valid_tokens = self.interpreter.valid_tokens()
        valid_actions = self.de_vocab.lookup(valid_tokens)
        if self.trigger_words_dict is not None:
            valid_actions = self.prune_with_trigger_words(valid_actions)

        # For each action, check the cache for the program, if
        # already tried, then not valid anymore.
        if self.use_cache:
            new_valid_actions = []
            cached_actions = []
            partial_program = self.de_vocab.lookup(self.mapped_actions, reverse=True)
            for ma in valid_actions:
                new_program = partial_program + [self.de_vocab.lookup(ma, reverse=True)]
                if not self.cache.check(new_program):
                    new_valid_actions.append(ma)
                else:
                    cached_actions.append(ma)
            valid_actions = new_valid_actions

        self.valid_actions = valid_actions
        self.rewards.append(reward)

        ob = Observation(read_ind=mapped_action,  # current action id in vocab, where vocab is a list of [functions|variables|special_tokens]
                         write_ind=new_var_id,  # newly created variable id in vocab, e.g. id of '20'
                         valid_action_indices=self.valid_actions,
                         output_features=[self.id_feature_dict[a] for a in valid_actions])  # mask, 1 for prop, 0 for function/entity/result

        # If no valid actions are available, then stop.
        if not self.valid_actions:
            self.done = True
            self.error = True

        # If the program is not finished yet, collect the
        # observation.
        if not self.done:
            # Add the actions that are filtered by cache into the
            # training example because at test time, they will be
            # there (no cache is available).

            # Note that this part is a bit tricky, `self.obs.valid_actions`
            # maintains all valid actions regardless of the cache, while the returned
            # observation `ob` only has valid continuating actions not covered by
            # the cache. `self.obs` shall only be used in training to compute
            # valid action masks for trajectories
            if self.use_cache:
                valid_actions = self.valid_actions + cached_actions

                true_ob = Observation(read_ind=mapped_action, write_ind=new_var_id, valid_action_indices=valid_actions,
                                      output_features=[self.id_feature_dict[a] for a in valid_actions])
                self.obs.append(true_ob)  # observation保存了所有valid actions，不考虑cache
            else:
                self.obs.append(ob)
        elif self.use_cache:
            # If already finished, save it in the cache.
            self.cache.save(self.de_vocab.lookup(self.mapped_actions, reverse=True))
        return ob, reward, self.done, {}
        # 'valid_actions': valid_actions, 'new_var_id': new_var_id}

    def calculate_region_reward(self, selected_region: Region, data_cells):
        """ Reward of matching selected region."""
        if selected_region is None:  # region selection not complete
            return 0
        if len(data_cells) == 0:  # TODO: maybe empty 'quantity_link'
            return 0
        left_ids, top_ids = selected_region.left_ids, selected_region.top_ids
        selected_region_data = [self.kg['kg'].data_region[left_id][top_id].value   # already List[float]
                                                for top_id in top_ids
                                                for left_id in left_ids]
        data_cells_sorted = sorted(data_cells,  key=functools.cmp_to_key(linked_cell_compare))
        data_cells_value = [v[1] for v in data_cells_sorted]  # List[float]
        reward = self.score_fn(selected_region_data, data_cells_value)
        return reward

    def calculate_op_reward(self, selected_ops, aggregations):
        """ Reward of matching operation type(s)."""
        if len(aggregations) == 1 and aggregations[0] == 'none' and len(selected_ops) == 0:
            return 1
        annotated_aggrs = []
        for aggr in aggregations:
            if aggr in AGGR_MAP:
                annotated_aggrs.extend(AGGR_MAP[aggr])
        annotated_aggrs = set(annotated_aggrs)
        if len(selected_ops) > 0 and set(selected_ops).issubset(annotated_aggrs):
            return 1
        else:
            return 0

    def calculate_entity_link_reward(self, selected_headers, entity_link_tokens):
        """ Reward of matching entity link(s)."""
        if set(entity_link_tokens).issubset(set(selected_headers)):
            return 1
        else:
            return 0

    def calculate_answer_reward(self, result, answer):
        """ Reward of matching answer."""
        reward = self.score_fn(result, answer)
        return reward

    def prune_with_trigger_words(self, valid_actions):
        """ Prune valid action candidates with trigger words."""
        question_tokens = self.question_annotation['tmp_tokens']
        if 'pos_tags' in self.question_annotation:
            pos_tags = self.question_annotation['pos_tags']
            tokens = question_tokens + pos_tags
        else:
            tokens = question_tokens
        invalid_functions = []
        for function, trigger_words in self.trigger_words_dict.items():
            # if there is no overlap between `trigger_words` and `tokens`
            if not set(trigger_words) & set(tokens):
                invalid_functions.append(function)
        invalid_actions = self.de_vocab.lookup(invalid_functions)
        new_valid_actions = list(set(valid_actions) - set(invalid_actions))
        return new_valid_actions

    def reset(self):  # only used in __init__()
        self.actions = []
        self.mapped_actions = []
        self.program = []
        self.rewards = []
        self.done = False
        valid_actions = self.de_vocab.lookup(self.interpreter.valid_tokens())
        if self.use_cache:
            new_valid_actions = []
            for ma in valid_actions:
                partial_program = self.de_vocab.lookup(
                    self.mapped_actions + [ma], reverse=True)
                if not self.cache.check(partial_program):
                    new_valid_actions.append(ma)
            valid_actions = new_valid_actions
        self.valid_actions = valid_actions
        self.start_ob = Observation(self.de_vocab.decode_id,
                                    -1,
                                    valid_actions,
                                    [self.id_feature_dict[a] for a in valid_actions])
        self.obs = [self.start_ob]

    def interactive(self):
        self.interpreter.interactive()
        print('reward is: %s' % self.score_fn(self.interpreter))

    def clone(self):
        new_interpreter = self.interpreter.clone()
        new = QAProgrammingEnv(
            question_annotation=self.question_annotation,
            kg=self.kg,
            answer=self.answer,
            score_fn=self.score_fn,
            interpreter=new_interpreter,
            de_vocab=self.de_vocab,
            constants=self.constants,
            init_interp=False,
            context=self.context,
            id_feature_dict=self.id_feature_dict,
            cache=self.cache,
            reset=False,
            alpha_region=self.alpha_region,
            alpha_op=self.alpha_op,
            alpha_entity_link=self.alpha_entity_link,
            trigger_words_dict=self.trigger_words_dict
        )
        new.actions = self.actions[:]
        new.mapped_actions = self.mapped_actions[:]
        new.program = self.program[:]
        new.rewards = self.rewards[:]
        new.obs = self.obs[:]
        new.done = self.done
        new.name = self.name
        # Cache is shared among all copies of this environment.
        new.cache = self.cache
        new.use_cache = self.use_cache
        new.valid_actions = self.valid_actions
        new.start_ob = self.start_ob
        new.error = self.error
        new.id_feature_dict = self.id_feature_dict
        new.punish_extra_work = self.punish_extra_work
        new.trigger_words_dict = self.trigger_words_dict

        return new

    def show(self):
        program = ' '.join(
            self.de_vocab.lookup([o.read_ind for o in self.obs], reverse=True))
        valid_tokens = ' '.join(self.de_vocab.lookup(self.valid_actions, reverse=True))
        return 'program: {}\nvalid tokens: {}'.format(program, valid_tokens)

    def get_human_readable_action_token(self, program_token: str) -> str:
        if program_token.startswith('v'):
            mem_entry = self.interpreter.namespace[program_token]
            if mem_entry['is_constant']:
                if isinstance(mem_entry['value'], list):
                    value = ', '.join(map(str, mem_entry['value']))
                else:
                    value = str(mem_entry['value'])

                token = f"{program_token}:{value}"
            else:
                token = program_token
        else:
            token = program_token

        return token

    def to_human_readable_program(self):
        readable_program = []
        for token in self.program:
            readable_token = self.get_human_readable_action_token(token)
            readable_program.append(readable_token)

        return readable_program

    def get_entity_link_action_indices_and_tokens(self, annotated_entity_link):
        try:
            entity_link_tokens, entity_link_vars = set(), set()
            for direction in ['top', 'left']:
                link_dict = annotated_entity_link[direction]
                for phrase, links in link_dict.items():
                    for coord, literal in links.items():
                        for k, v in self.interpreter.namespace.items():
                            if v['value'] == literal:
                                entity_link_tokens.add(literal)
                                entity_link_vars.add(k)
            entity_link_tokens, entity_link_vars = list(entity_link_tokens), list(entity_link_vars)
            entity_link_actions = self.de_vocab.lookup(entity_link_vars)
            return entity_link_actions, entity_link_tokens

        except Exception as e:
            # print(f"Parse str2dict error: {e}")
            return []

    def get_data_cells(self, annotated_quantity_link):
        links = annotated_quantity_link.get('[ANSWER]', None)
        if not links:
            return []
        data_cells = set()
        for coord, literal in links.items():
            data_cells.add((eval(coord), literal))
        data_cells = list(data_cells)
        return data_cells

    def get_aggregation_indices(self, annotated_aggregation):
        aggregation_tokens = []
        for aggr in annotated_aggregation:
            if aggr in AGGR_MAP:
                aggregation_tokens.extend(AGGR_MAP[aggr])
        aggregation_tokens = list(set(aggregation_tokens))
        aggregation_actions = self.de_vocab.lookup(aggregation_tokens)
        return aggregation_actions


class SearchCache(object):
    def __init__(self, name, size=None, max_elements=1e4, error_rate=1e-8):
        self.name = name
        self.max_elements = max_elements
        self.error_rate = error_rate
        self._set = bloom_filter.BloomFilter(
            max_elements=max_elements, error_rate=error_rate)

    def check(self, tokens):
        return ' '.join(tokens) in self._set

    def save(self, tokens):
        string = ' '.join(tokens)
        self._set.add(string)

    def is_full(self):
        return '(' in self._set

    def reset(self):
        self._set = bloom_filter.BloomFilter(
            max_elements=self.max_elements, error_rate=self.error_rate)


class Sample(object):
    def __init__(self, trajectory: Trajectory, prob: Union[float, torch.Tensor], **kwargs):
        self.trajectory = trajectory
        self.prob = prob

        for field, value in kwargs.items():
            setattr(self, field, value)

    def to(self, device: torch.device):
        for ob in self.trajectory.observations:
            ob.to(device)

        return self

    def __repr__(self):
        return 'Sample({}, prob={})'.format(self.trajectory, self.prob)

    __str__ = __repr__
