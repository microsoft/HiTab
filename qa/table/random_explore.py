import json
import time
import os
import multiprocessing
from argparse import ArgumentParser
from pathlib import Path

from qa.nsm.env import *
from qa.nsm.computer import *
from qa.nsm.execution.executor import *
from qa.table.utils import hmt_score
from qa.datadump.utils import *

np.random.seed(2)


def get_experiment_dir():
    experiment_dir = args.output_dir / args.experiment_name

    return experiment_dir


def random_explore(env,
                   use_cache=True,
                   trigger_dict=None,
                   ):
    env = env.clone()
    env.use_cache = use_cache
    question_tokens = env.question_annotation['tokens']
    if 'pos_tags' in env.question_annotation:
        pos_tags = env.question_annotation['pos_tags']
        tokens = question_tokens + pos_tags
    else:
        tokens = question_tokens
    invalid_functions = []
    if trigger_dict is not None:
        for function, trigger_words in trigger_dict.items():
            # if there is no overlap between `trigger_words` and `tokens`
            if not set(trigger_words) & set(tokens):
                invalid_functions.append(function)
    ob = env.start_ob
    while not env.done:
        invalid_actions = env.de_vocab.lookup(invalid_functions)
        valid_actions = ob.valid_action_indices
        new_valid_actions = list(set(valid_actions) - set(invalid_actions))

        # add manual constraints, turn on/off "not contain" and "union header"
        new_valid_actions = add_manual_constraints(env,
                                                   new_valid_actions,
                                                   allow_not_contain=args.allow_not_contain,
                                                   allow_union_header=args.allow_union_header,
                                                   )
        if args.use_linked_cells:  # add constraint with annotated entity links and aggregation
            new_valid_actions = add_weights_to_action(env,
                                                      new_valid_actions)
            new_valid_actions = add_entity_link_constraints(env,
                                                            new_valid_actions,
                                                            allow_min_max=args.allow_min_max,
                                                            allow_sum_average=args.allow_sum_average)

        # No action available anymore.
        if len(new_valid_actions) <= 0:
            return None
        new_action = np.random.randint(0, len(new_valid_actions))
        action = valid_actions.index(new_valid_actions[new_action])
        ob, _, _, _ = env.step(action)
    if sum(env.rewards) >= args.total_reward_threshold and union_headers_both_in_entity_link(env):
        return env.de_vocab.lookup(env.mapped_actions, reverse=True), \
               env.to_human_readable_program(), \
               sum(env.rewards)
    else:
        return None


def union_headers_both_in_entity_link(env):
    """ Assert if selecting union headers, they have to be both in entity links."""
    if not args.use_linked_cells:
        return True
    i = 0
    while i < len(env.mapped_actions):
        if env.mapped_actions[i] == env.de_vocab.lookup('filter_tree_str_contain'):
            if env.mapped_actions[i+3] != env.de_vocab.lookup(')'):  # union headers are selected
                if env.mapped_actions[i+2] in env.entity_link_actions and env.mapped_actions[i+3] in env.entity_link_actions:
                    i += 5  # after '('
                else:
                    return False
            else:  # after '('
                i += 4
        else:
            i += 1
    return True


def add_weights_to_action(env, valid_actions):
    """ Multiply actions in entity link and aggregation actions."""
    new_valid_actions = []
    num_actions = len(valid_actions)
    W = 6
    for action in valid_actions:
        if action in env.entity_link_actions or action in env.aggregation_actions:  # add weights of annotated tokens
            new_valid_actions.extend([action] * W * num_actions)
        else:
            # TODO: should change the code when new functions are added
            aggr_action_start, aggr_action_end = env.de_vocab.lookup('max'), env.de_vocab.lookup('opposite')
            if not env.aggregation_actions and action in list(
                    range(aggr_action_start, aggr_action_end + 1)):  # no aggregation, remove all operations
                continue
            else:
                new_valid_actions.append(action)
    return new_valid_actions


def add_manual_constraints(env, valid_actions,
                           allow_not_contain=False, allow_union_header=False):
    """ Add constraints only based on config."""
    if len(valid_actions) == 0:
        return valid_actions
    exp_stack = env.interpreter.exp_stack
    exp = exp_stack[-1] if len(exp_stack) > 0 else []
    # whether to allow 'not contain'
    if not allow_not_contain:
        if env.de_vocab.lookup('filter_tree_str_not_contain') in valid_actions:
            valid_actions.remove(env.de_vocab.lookup('filter_tree_str_not_contain'))
    # whether to allow two headers in one 'filter tree'
    if not args.use_linked_cells:
        union_words = {'or', 'and'}  # union trigger words
        if not (union_words & set(env.question_annotation['tokens'])):
            allow_union_header = False
    if not allow_union_header:  # dont allow two headers as input of 'filter_tree'
        if ('filter_tree_str_contain' in exp or 'filter_tree_str_not_contain' in exp) and len(exp) == 3:
            if env.de_vocab.lookup(')') in valid_actions:
                valid_actions = [env.de_vocab.lookup(')')]
    return valid_actions


def add_entity_link_constraints(env, valid_actions,
                                allow_min_max=False, allow_sum_average=False):
    """ Add constraints based on annotated entity links."""
    if len(valid_actions) == 0:
        return valid_actions
    exp_stack = env.interpreter.exp_stack
    exp = exp_stack[-1] if len(exp_stack) > 0 else []
    # whether to allow min/max if not annotated
    if not allow_min_max:
        min_action, max_action = env.de_vocab.lookup('min'), env.de_vocab.lookup('max')
        if min_action in valid_actions and min_action not in env.aggregation_actions:
            valid_actions.remove(min_action)
        if max_action in valid_actions and max_action not in env.aggregation_actions:
            valid_actions.remove(max_action)
    # whether to allow sum/average if not annotated
    if not allow_sum_average:
        sum_action, average_action = env.de_vocab.lookup('sum'), env.de_vocab.lookup('average')
        if sum_action in valid_actions and sum_action not in env.aggregation_actions:
            valid_actions.remove(sum_action)
        if average_action in valid_actions and average_action not in env.aggregation_actions:
            valid_actions.remove(average_action)
    # use entity link to limit union headers
    if ('filter_tree_str_contain' in exp or 'filter_tree_str_not_contain' in exp) and len(exp) == 3:
        if len(set(env.entity_link_actions).intersection(set(valid_actions))) == 0:
            if env.de_vocab.lookup(')') in valid_actions:
                valid_actions = [env.de_vocab.lookup(')')]
    return valid_actions


def run_random_exploration(shard_id):
    experiment_dir = get_experiment_dir()
    experiment_dir.mkdir(exist_ok=True, parents=True)

    if args.trigger_word_file.exists():
        with args.trigger_word_file.open() as f:
            trigger_dict = json.load(f)
            print('use trigger words in {}'.format(args.trigger_word_file))
    else:
        trigger_dict = None

    # Load dataset.
    train_set = []
    train_shard_file = Path(args.train_file_tmpl.format(shard_id))
    print('working on shard {}'.format(train_shard_file))
    with train_shard_file.open() as f:
        for line in f:
            example = json.loads(line)
            train_set.append(example)
    print('{} examples in training set.'.format(len(train_set)))

    table_dict = {}
    with args.table_file.open() as f:
        for line in f:
            table = json.loads(line)
            table_dict[table['name']] = table
    print('{} tables.'.format(len(table_dict)))

    if args.executor == 'hmt':
        score_fn = hmt_score
        process_answer_fn = lambda x: x
        executor_fn = HMTExecutor
    else:
        raise ValueError('Unknown executor {}'.format(args.executor))
    all_envs = []
    t1 = time.time()
    for i, example in enumerate(train_set):
        if i % 100 == 0:
            print('creating environment #{}'.format(i))
        kg_info = table_dict[example['context']]
        kg_info = HMTable.from_dict(kg_info['kg']).build_kg_info()
        executor = executor_fn(kg_info)
        api = executor.get_api()
        type_hierarchy = api['type_hierarchy']
        func_dict = api['func_dict']
        constant_dict = api['constant_dict']
        interpreter = LispInterpreter(
            op_region=kg_info['kg'].get_init_op_region(),
            type_hierarchy=type_hierarchy,
            max_mem=args.max_n_mem,
            max_n_exp=args.max_n_exp,
            hmt=kg_info['kg'],
            assisted=True
        )

        for v in func_dict.values():
            interpreter.add_function(**v)

        de_vocab = interpreter.get_vocab()
        env = QAProgrammingEnv(
            question_annotation=example,
            kg=kg_info,
            answer=process_answer_fn(example['answer']),
            constants=constant_dict.values(),
            interpreter=interpreter,
            score_fn=score_fn,
            max_cache_size=args.n_explore_samples * args.n_epoch * 10,
            name=example['id'],
            alpha_region=args.alpha_region,
            alpha_op=args.alpha_op,
            alpha_entity_link=args.alpha_entity_link
        )
        all_envs.append(env)

    program_dict = dict([(env.name, []) for env in all_envs])
    max_reward_dict = dict([(env.name, 0.) for env in all_envs])
    for i in range(1, args.n_epoch + 1):
        print('iteration {}'.format(i))
        t1 = time.time()
        for env in all_envs:
            if len(program_dict[env.name]) > 20:
                continue
            # update the program dict: (1) replace with programs w/ higher reward; (2) add programs w/ same reward
            for _ in range(args.n_explore_samples):
                result = random_explore(env, trigger_dict=trigger_dict)
                if result is not None:
                    program, program_readable, program_reward = result
                    if program_reward > max_reward_dict[env.name]:
                        max_reward_dict[env.name] = program_reward
                        new_programs = []
                        for p_r in program_dict[env.name]:
                            if p_r[1] == max_reward_dict[env.name]:
                                new_programs.append(p_r)
                        program_dict[env.name] = new_programs
                        program_dict[env.name].append((program, program_reward))
                    elif program_reward == max_reward_dict[env.name]:
                        program_dict[env.name].append((program, program_reward))
                        # program_dict[env.name].append(program_readable)
        t2 = time.time()
        print('{} sec used in iteration {}'.format(t2 - t1, i))

        if i % args.save_every_n == 0 or i >= args.n_epoch:
            print('saving programs and cache in iteration {}'.format(i))
            t1 = time.time()
            with open(os.path.join(
                    get_experiment_dir(), 'program_shard_{}-{}.json'.format(shard_id, i)), 'w') as f:
                program_str_dict = dict([(k, [' '.join(p[0]) for p in v]) for k, v
                                         in program_dict.items()])
                json.dump(program_str_dict, f, sort_keys=True, indent=2)

            # cache_dict = dict([(env.name, list(env.cache._set)) for env in all_envs])
            t2 = time.time()
            print(
                '{} sec used saving programs and cache in iteration {}'.format(
                    t2 - t1, i))

        n = len(all_envs)
        solution_ratio = len([env for env in all_envs if program_dict[env.name]]) * 1.0 / n
        print(
            'At least one solution found ratio: {}'.format(solution_ratio))
        n_programs_per_env = np.array([len(program_dict[env.name]) for env in all_envs])
        print(
            'number of solutions found per example: max {}, min {}, avg {}, std {}'.format(
                n_programs_per_env.max(), n_programs_per_env.min(), n_programs_per_env.mean(),
                n_programs_per_env.std()))

        # Macro average length.
        mean_length = np.mean([np.mean([len(p[0]) for p in program_dict[env.name]]) for env in all_envs
                               if program_dict[env.name]])
        print('macro average program length: {}'.format(
            mean_length))


def collect_programs():
    saved_programs = {}
    for i in range(args.id_start, args.id_end):
        with open(os.path.join(
                get_experiment_dir(),
                'program_shard_{}-{}.json'.format(i, args.n_epoch)), 'r') as f:
            program_shard = json.load(f)
            saved_programs.update(program_shard)
    saved_program_path = get_experiment_dir() / args.saved_programs_file
    num_found, num_programs_searched = 0, 0
    for id in saved_programs:
        if len(saved_programs[id]) > 0:
            num_found += 1
            num_programs_searched += len(saved_programs[id])
    print(f"\nat least one solution found ratio: {num_found / len(saved_programs)}")
    print(f"\naverage {num_programs_searched / num_found} programs for each sample.")
    with saved_program_path.open('w') as f:
        json.dump(saved_programs, f)
    print('saved programs are aggregated in {}'.format(saved_program_path))


def main(unused_argv):
    ps = []
    for idx in range(args.id_start, args.id_end):
        p = multiprocessing.Process(target=run_random_exploration, args=(idx,))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
    # run_random_exploration(0)
    collect_programs()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='output directory')
    parser.add_argument(
        '--experiment_name',
        type=str,
        required=True,
        help='All outputs of this experiment is'
             ' saved under a folder with the same name.')

    parser.add_argument('--table_file', type=Path, required=True, help='table file')
    parser.add_argument('--train_file_tmpl', type=str, required=True, help='training shards')
    parser.add_argument('--saved_programs_file', type=str, required=True, help='saved programs file')
    parser.add_argument('--trigger_word_file', type=Path, required=False, help='trigger word file')

    parser.add_argument('--n_epoch', type=int, default=10)

    parser.add_argument('--max_n_mem', type=int, default=120, help='Max number of memory slots in the "computer".')
    parser.add_argument('--max_n_exp', type=int, default=10, help='Max number of expressions allowed in a program.')
    parser.add_argument('--max_n_valid_indices', type=int, default=100,
                        help='Max number of valid tokens during decoding.')
    parser.add_argument('--executor', type=str, default='wtq', help='Which executor to use, wtq or wikisql.')

    parser.add_argument('--n_explore_samples', type=int, default=50)
    parser.add_argument('--save_every_n', type=int, default=10)
    parser.add_argument('--id_start', type=int, default=0)
    parser.add_argument('--id_end', type=int, default=0)

    # constraints
    parser.add_argument('--allow_not_contain', action='store_true')
    parser.add_argument('--allow_union_header', action='store_true')

    parser.add_argument('--use_linked_cells', action='store_true')
    parser.add_argument('--allow_min_max', action='store_true')
    parser.add_argument('--allow_sum_average', action='store_true')

    # reward coefficients
    parser.add_argument('--alpha_region', type=float, default=0.)
    parser.add_argument('--alpha_op', type=float, default=0.)
    parser.add_argument('--alpha_entity_link', type=float, default=0.)
    parser.add_argument('--total_reward_threshold', type=float, default=1.0)

    args = parser.parse_args()

    main(args)
