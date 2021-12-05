from pathlib import Path
from typing import List, Dict, Iterable, Any, Optional, Union
import multiprocessing
from argparse import ArgumentParser
import sys
import os

from qa.nsm.actor import Actor
from qa.nsm.parser_module.agent import PGAgent
import qa.table.utils as utils
from qa.nsm.data_utils import load_jsonl
from qa.nsm.evaluator import Evaluator, Evaluation
from qa.nsm.learner import Learner
from qa.nsm.program_cache import SharedProgramCache
from qa.nsm.execution.executor import *
from qa.nsm.computer import *
from qa.nsm.env import *

def annotate_example_for_bert(
        example: Dict, table: Dict,
        bert_tokenizer: BertTokenizer,
):
    # sub-tokenize the question
    question_tokens = example['tokens']
    example['original_tokens'] = question_tokens
    token_position_map = OrderedDict()  # map of token index before and after sub-tokenization

    question_feature = example['features']

    cur_idx = 0
    new_question_feature = []
    question_subtokens = []
    for old_idx, token in enumerate(question_tokens):
        if token == '<DECODE>': token = '[MASK]'
        if token == '<START>': token = '[MASK]'

        sub_tokens = bert_tokenizer.tokenize(token)
        question_subtokens.extend(sub_tokens)

        token_new_idx_start = cur_idx
        token_new_idx_end = cur_idx + len(sub_tokens)
        token_position_map[old_idx] = (token_new_idx_start, token_new_idx_end)
        new_question_feature.extend([question_feature[old_idx]] * len(sub_tokens))

        cur_idx = token_new_idx_end

    token_position_map[len(question_tokens)] = (len(question_subtokens), len(question_subtokens))

    example['tokens'] = question_subtokens
    example['features'] = new_question_feature

    for entity in example['entities']:
        old_token_start = entity['token_start']
        old_token_end = entity['token_end']

        new_token_start = token_position_map[old_token_start][0]
        new_token_end = token_position_map[old_token_end][0]

        entity['token_start'] = new_token_start
        entity['token_end'] = new_token_end

    table = table['kg'].tokenize(bert_tokenizer)
    example['table'] = table

    return example


def load_environments(
        example_files: List[str],
        table_file: str,
        example_ids: Iterable = None,
        bert_tokenizer: BertTokenizer = None,
        alpha_region: float = 0.,
        alpha_op: float = 0.,
        alpha_entity_link: float = 0.,
        trigger_words_dict: Dict = None
):
    dataset = []
    if example_ids is not None:
        example_ids = set(example_ids)

    for fn in example_files:
        data = load_jsonl(fn)
        for example in data:
            if example_ids:
                if example['id'] in example_ids:
                    dataset.append(example)
            else:
                dataset.append(example)
    print('{} examples in dataset.'.format(len(dataset)))

    tables = load_jsonl(table_file)
    table_dict = {table['name']: table for table in tables}
    print('{} tables.'.format(len(table_dict)))

    environments = create_environments(
        table_dict, dataset,
        executor_type='hmt',
        bert_tokenizer=bert_tokenizer,
        alpha_region=alpha_region,
        alpha_op=alpha_op,
        alpha_entity_link=alpha_entity_link,
        trigger_words_dict=trigger_words_dict
    )
    print('{} environments in total'.format(len(environments)))

    return environments


def create_environments(
        table_dict, dataset,  # table_dict:{table_name:table}, dataset:[q-t-a]
        executor_type,
        max_n_mem=120, max_n_exp=10,
        bert_tokenizer=None,
        alpha_region: float = 0.,
        alpha_op: float = 0.,
        alpha_entity_link: float = 0.,
        trigger_words_dict: Dict = None
) -> List[QAProgrammingEnv]:
    all_envs = []

    for i, example in enumerate(dataset):
        if i % 100 == 0:
            print('creating environment #{}'.format(i))

        kg_info = table_dict[example['context']]
        env = create_environment(
            example, kg_info,
            executor_type,
            max_n_mem, max_n_exp,
            bert_tokenizer,
            alpha_region,
            alpha_op,
            alpha_entity_link,
            trigger_words_dict
        )

        all_envs.append(env)

    return all_envs


def create_environment(
        example: Dict,
        table_kg: Dict,
        executor_type: str = 'hmt',
        max_n_mem: int = 120,
        max_n_exp: int = 10,
        bert_tokenizer: BertTokenizer = None,
        alpha_region: float = 0.,
        alpha_op: float = 0.,
        alpha_entity_link: float = 0.,
        trigger_words_dict: Dict = None,
) -> 'QAProgrammingEnv':
    if executor_type == 'hmt':
        score_fn = utils.hmt_score
        # process_answer_fn = eval
        process_answer_fn = lambda x: x
        executor_fn = HMTExecutor
    else:
        raise ValueError(f"Unknown executor: {executor_type}")

    if not isinstance(table_kg['kg'], HMTable):
        table_kg['kg'] = HMTable.from_dict(table_kg['kg'])  # 与build_kg_info()等价

    executor = executor_fn(table_kg)
    api = executor.get_api()
    type_hierarchy = api['type_hierarchy']
    func_dict = api['func_dict']
    constant_dict = api['constant_dict']  # rely on table_kg

    interpreter = LispInterpreter(
        op_region=table_kg['kg'].get_init_op_region(),
        type_hierarchy=type_hierarchy,
        max_mem=max_n_mem,
        max_n_exp=max_n_exp,
        hmt=table_kg['kg'],
        assisted=True
    )

    for v in func_dict.values():
        interpreter.add_function(**v)

    if bert_tokenizer:
        if isinstance(bert_tokenizer, BertTokenizer):
            example = annotate_example_for_bert(
                example, table_kg, bert_tokenizer,
            )
        else:
            raise NotImplementedError(f'Unknown tokenizer: {bert_tokenizer}')

    env = QAProgrammingEnv(
        question_annotation=example,
        kg=table_kg,
        answer=process_answer_fn(example['answer']),
        constants=constant_dict.values(),
        interpreter=interpreter,
        score_fn=score_fn,
        name=example['id'],
        alpha_region=alpha_region,
        alpha_op=alpha_op,
        alpha_entity_link=alpha_entity_link,
        trigger_words_dict=trigger_words_dict
    )
    return env


def to_human_readable_program(program, env):
    env = env.clone()
    env.use_cache = False
    ob = env.start_ob

    for tk in program:
        valid_actions = list(ob.valid_action_indices)
        action_id = env.de_vocab.lookup(tk)
        rel_action_id = valid_actions.index(action_id)
        ob, _, _, _ = env.step(rel_action_id)

    readable_program = []
    first_intermediate_var_id = len(
        [v for v, entry in env.interpreter.namespace.items() if v.startswith('v') and entry['is_constant']])
    for tk in program:
        if tk.startswith('v'):
            mem_entry = env.interpreter.namespace[tk]
            if mem_entry['is_constant']:
                if isinstance(mem_entry['value'], list):
                    token = mem_entry['value'][0]
                else:
                    token = mem_entry['value']
            else:
                intermediate_var_relative_id = int(tk[1:]) - first_intermediate_var_id
                token = 'v{}'.format(intermediate_var_relative_id)
        else:
            token = tk

        readable_program.append(token)

    return readable_program


def distributed_train(args):
    use_cuda = args.cuda

    config_file = args.config_file
    print(f'load config file [{config_file}]', file=sys.stderr)
    config = json.load(open(config_file))
    config = AttrDict(config)
    seed = config['seed']
    work_dir = config['work_dir']
    print(f'work dir [{work_dir}]', file=sys.stderr)

    if not os.path.exists(work_dir):
        print(f'creating work dir [{work_dir}]', file=sys.stderr)
        os.makedirs(work_dir)

    json.dump(config, open(os.path.join(work_dir, 'config.json'), 'w'), indent=2)

    actor_devices = []
    if use_cuda:
        print(f'use cuda', file=sys.stderr)
        device_count = torch.cuda.device_count()

        assert device_count >= 2

        # Learner and evaluator are on cuda:0, actors are on cuda:>=2.
        learner_devices = ['cuda:0', 'cuda:0']
        evaluator_device = learner_devices[0]

        for i in range(2, device_count):
            actor_devices.append(f'cuda:{i}')
        else:
            actor_devices.append('cpu')
    else:
        learner_devices = [torch.device('cpu'), torch.device('cpu')]
        evaluator_device = torch.device('cpu')
        actor_devices.append(torch.device('cpu'))

    shared_program_cache = SharedProgramCache()

    config['seed'] = seed
    learner = Learner(
        config=config,
        shared_program_cache=shared_program_cache,
        devices=learner_devices
    )

    print(f'Evaluator uses device {evaluator_device}', file=sys.stderr)
    config['seed'] = seed + 1
    evaluator = Evaluator(
        config=config,
        eval_file=config['dev_file'], device=evaluator_device)
    learner.register_evaluator(evaluator)

    actor_num = config['actor_num']
    print('initializing %d actors' % actor_num, file=sys.stderr)
    actors = []
    train_shard_dir = Path(config['train_shard_dir'])
    shard_start_id = config['shard_start_id']  # 0
    shard_end_id = config['shard_end_id']  # 90
    train_example_ids = []
    for shard_id in range(shard_start_id, shard_end_id):
        shard_data = load_jsonl(train_shard_dir / f"{config['train_shard_prefix']}{shard_id}.jsonl")
        train_example_ids.extend(
            e['id']
            for e
            in shard_data
        )

    per_actor_example_num = len(train_example_ids) // actor_num
    for actor_id in range(actor_num):
        config['seed'] = seed + 2 + actor_id
        actor = Actor(
            actor_id,
            example_ids=train_example_ids[
                        actor_id * per_actor_example_num:
                        ((actor_id + 1) * per_actor_example_num) if actor_id < actor_num - 1 else len(train_example_ids)
                        ],
            shared_program_cache=shared_program_cache,
            device=actor_devices[actor_id % len(actor_devices)],
            config=config, )
        learner.register_actor(actor)

        actors.append(actor)

    print('starting %d actors' % actor_num, file=sys.stderr)
    for actor in actors:
        actor.start()
        pass

    print('starting evaluator', file=sys.stderr)
    evaluator.start()

    print('starting learner', file=sys.stderr)
    learner.start()

    print('Learner process {}, evaluator process {}'.format(learner.pid, evaluator.pid), file=sys.stderr)

    # learner will quit first
    learner.join()
    print('Learner exited', file=sys.stderr)

    for actor in actors:
        actor.terminate()
        actor.join()

    evaluator.terminate()
    evaluator.join()


def test(args):
    use_gpu = args.cuda
    model_path = args.model_file

    print(f'loading model [{model_path}] for evaluation', file=sys.stderr)
    agent = PGAgent.load(model_path, gpu_id=0 if use_gpu else -1).eval()
    config = agent.config

    test_file = config['test_file']
    # test_file = 'data/processed_input/test_samples_processed.jsonl'  # TODO: debug use
    print(f'loading test file [{test_file}]', file=sys.stderr)
    test_envs = load_environments(
        [test_file],
        table_file=config['table_file'],
        bert_tokenizer=agent.encoder.bert_model.tokenizer,
        alpha_region=config['alpha_region'],
        alpha_op=config['alpha_op'],
        alpha_entity_link=config['alpha_entity_link']
    )

    for env in test_envs:
        env.use_cache = False
        env.punish_extra_work = False

    batch_size = config['eval_batch_size']
    beam_size = config['beam_size']
    print(f'batch size {batch_size}, beam size {beam_size}', file=sys.stderr)
    decode_results = agent.decode_examples(test_envs,
                                           beam_size=beam_size,
                                           batch_size=batch_size)
    assert len(test_envs) == len(decode_results)
    eval_results = Evaluation.evaluate_decode_results(test_envs, decode_results)
    print(eval_results, file=sys.stderr)

    save_to = config['save_decode_to']
    if save_to != 'None':
        print(f'save results to [{save_to}]', file=sys.stderr)

        results = to_decode_results_dict(decode_results, test_envs)

        json.dump(results, open(save_to, 'w'), indent=2)


def to_decode_results_dict(decode_results, test_envs):
    results = OrderedDict()

    for env, hyp_list in zip(test_envs, decode_results):

        env_result = {
            'name': env.name,
            'question': ' '.join(str(x) for x in env.context['question_tokens']),
            'hypotheses': None
        }

        hypotheses = []
        for hyp in hyp_list:
            hypotheses.append(OrderedDict(
                program=' '.join(str(x) for x in to_human_readable_program(hyp.trajectory.program, env)),
                is_correct=hyp.trajectory.reward >= 1.,
                prob=hyp.prob
            ))

        env_result['hypotheses'] = hypotheses
        env_result['top_prediction_correct'] = hypotheses and hypotheses[0]['is_correct']
        results[env.name] = env_result

    return results


def main():
    multiprocessing.set_start_method('spawn', force=True)

    # command line args
    parser = ArgumentParser()
    parser.add_argument('--train', action="store_true", help="train phase")
    parser.add_argument('--test', action="store_true", help="test phase")
    parser.add_argument('--cuda', action="store_true", help="use cuda")
    parser.add_argument('--config_file', default='qa/config/config.vanilla_bert.json', help="configuration file")
    # for test
    parser.add_argument('--model_file', default='qa/runs/hmtqa/model.best.bin', help='test model file')
    args = parser.parse_args()

    if args.train:
        distributed_train(args)
    elif args.test:
        test(args)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == '__main__':
    main()
