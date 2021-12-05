import os
import re
import sys
import time
import json
from pathlib import Path

import torch.multiprocessing as torch_mp
import multiprocessing

from qa.nsm import nn_util
from qa.nsm.parser_module import get_parser_agent_by_name
from qa.nsm.parser_module.agent import PGAgent

import torch

from qa.nsm.replay_buffer import ReplayBuffer


class Actor(torch_mp.Process):
    def __init__(self, actor_id, example_ids, shared_program_cache, device, config):
        super(Actor, self).__init__(daemon=True)

        self.config = config
        self.actor_id = f'Actor_{actor_id}'
        self.example_ids = example_ids
        self.device = device

        if not self.example_ids:
            raise RuntimeError(f'empty shard for Actor {self.actor_id}')

        self.model_path = None
        self.checkpoint_queue = None
        self.train_queue = None
        self.shared_program_cache = shared_program_cache

    def run(self):
        # initialize cuda context
        self.device = torch.device(self.device)
        if 'cuda' in self.device.type:
            torch.cuda.set_device(self.device)

        # seed the random number generators
        nn_util.init_random_seed(self.config['seed'], self.device)

        def get_train_shard_path(i):
            return os.path.join(
                self.config['train_shard_dir'], self.config['train_shard_prefix'] + str(i) + '.jsonl')

        # create agent and set it to evaluation mode
        if 'cuda' in str(self.device.type):
            torch.cuda.set_device(self.device)

        agent_name = self.config.get('parser', 'vanilla')
        self.agent = get_parser_agent_by_name(agent_name).build(self.config, master=self.actor_id).to(self.device).eval()  # agent包含encoder/decoder

        # load environments
        self.load_environments(
            [
                get_train_shard_path(i)
                for i
                in range(self.config['shard_start_id'], self.config['shard_end_id'])
            ],
            example_ids=self.example_ids
        )

        self.replay_buffer = ReplayBuffer(self.agent, self.shared_program_cache)

        if self.config['load_saved_programs']:
            self.replay_buffer.load(self.environments, self.config['saved_program_file'])
            print(f'[Actor {self.actor_id}] loaded {self.replay_buffer.size} programs to buffer', file=sys.stderr)

        self.train()

    def train(self):
        config = self.config
        epoch_id = 0
        sample_method = self.config['sample_method']
        method = self.config['method']
        assert sample_method in ('sample', 'beam_search')
        assert method in ('sample', 'mapo', 'mml')

        work_dir = Path(self.config['work_dir'])
        log_dir = work_dir / 'log'
        log_dir.mkdir(exist_ok=True, parents=True)

        debug_file = None
        if self.config.get('save_actor_log', False):
            debug_file = (log_dir / f'debug.actor{self.actor_id}.log').open('w')

        with torch.no_grad():
            while True:
                epoch_id += 1
                epoch_start = time.time()
                batch_iter = nn_util.batch_iter(self.environments, batch_size=self.config['batch_size'], shuffle=True)  # 10envs/batch
                for batch_id, batched_envs in enumerate(batch_iter):
                    try:
                        if method in ['mapo', 'sample']:
                            t1 = time.time()
                            # Algorithm1 in MAPO paper
                            if sample_method == 'sample':
                                explore_samples = self.agent.sample(
                                    batched_envs,
                                    sample_num=config['n_explore_samples'],  # 10
                                    use_cache=config['use_cache'],  # True
                                )
                            else:
                                explore_samples = self.agent.new_beam_search(
                                    batched_envs,
                                    beam_size=config['n_explore_samples'],
                                    use_cache=config['use_cache'],
                                    return_list=True,
                                )
                            t2 = time.time()

                            if debug_file:
                                print('Explored programs:', file=debug_file)
                                for sample in explore_samples:
                                    print(f"[{sample.trajectory.environment_name}] "
                                          f"{' '.join(sample.trajectory.program)} "
                                          f"(prob={sample.prob:.4f}, "
                                          f"correct={sample.trajectory.reward >= self.config['good_sample_threshold']})",
                                          file=debug_file)

                            print(
                                f'[Actor {self.actor_id}] '
                                f'epoch {epoch_id} batch {batch_id}, '
                                f'sampled {len(explore_samples)} trajectories (took {t2 - t1}s)', file=sys.stderr
                            )

                            # retain samples with high reward
                            good_explore_samples = [sample for sample in explore_samples
                                                    if sample.trajectory.reward >= self.config['good_sample_threshold']]
                            self.replay_buffer.save_samples(good_explore_samples)  # save high-reward programs to replay-buffer

                        # sample replay examples from the replay buffer
                        t1 = time.time()
                        # Algorithm2 (after Algorithm1 is applied) in MAPO paper
                        replay_samples = self.replay_buffer.replay(  # sample high-reward from replay_buffer, i.e. sample a+ from B in paper
                            batched_envs,
                            n_samples=config['n_replay_samples'],
                            use_top_k=config['use_top_k_replay_samples'],
                            replace=config['replay_sample_with_replacement'],
                            truncate_at_n=config.get('sample_replay_from_topk', 0),
                            debug_file=debug_file
                        )
                        t2 = time.time()
                        print(f'[Actor {self.actor_id}] epoch {epoch_id} batch {batch_id}, got {len(replay_samples)} replay samples (took {t2 - t1}s)',
                              file=sys.stderr)

                        samples_info = dict()
                        if method == 'mapo':
                            train_examples = []
                            for sample in replay_samples:  # high-reward into training batch D
                                sample_weight = self.replay_buffer.env_program_prob_sum_dict.get(sample.trajectory.environment_name, 0.)
                                sample_weight = max(sample_weight, self.config['min_replay_samples_weight'])

                                sample.weight = sample_weight * 1. / config['n_replay_samples']  # 0.1/1=0.1
                                train_examples.append(sample)

                            on_policy_samples = self.agent.sample(batched_envs,  # sample non-high-reward, not from B
                                                                  sample_num=config['n_policy_samples'],
                                                                  use_cache=False)
                            non_replay_samples = [sample for sample in on_policy_samples
                                                  if sample.trajectory.reward >= self.config['good_sample_threshold']
                                                  and not self.replay_buffer.contains(sample.trajectory)]
                            self.replay_buffer.save_samples(non_replay_samples)

                            for sample in non_replay_samples:  # non-high-reward into training batch D

                                replay_samples_prob = self.replay_buffer.env_program_prob_sum_dict.get(sample.trajectory.environment_name, 0.)
                                if replay_samples_prob > 0.:
                                    # clip the sum of probabilities for replay samples if the replay buffer is not empty
                                    replay_samples_prob = max(replay_samples_prob, self.config['min_replay_samples_weight'])

                                sample_weight = 1. - replay_samples_prob

                                sample.weight = sample_weight * 1. / config['n_policy_samples']
                                train_examples.append(sample)

                            n_clip = 0
                            for env in batched_envs:
                                name = env.name
                                if (name in self.replay_buffer.env_program_prob_dict and
                                        self.replay_buffer.env_program_prob_sum_dict.get(name, 0.) < self.config['min_replay_samples_weight']):
                                    n_clip += 1
                            clip_frac = n_clip / len(batched_envs)

                            train_examples = train_examples
                            samples_info['clip_frac'] = clip_frac
                        elif method == 'mml':  # only train on high-reward programs
                            for sample in replay_samples:
                                sample.weight = sample.prob / self.replay_buffer.env_program_prob_sum_dict[sample.trajectory.environment_name]
                            train_examples = replay_samples
                        elif method == 'sample':  # reinforce
                            if self.get_global_step() > config['warm_up_steps']:
                                train_examples = good_explore_samples
                            else:
                                train_examples = replay_samples

                            for sample in train_examples:
                                sample.weight = max(sample.prob, config['min_replay_samples_weight'])
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            msg = (
                                    f'[Actor {self.actor_id}] WARNING: ran out of memory with exception: '
                                    + '{};'.format(e)
                                    + '\n Skipping batch'
                            )
                            print(msg, file=sys.stderr)
                            sys.stderr.flush()

                            continue
                        else:
                            raise e

                    if train_examples:  # save into train_queue, shared with 'learner'
                        self.train_queue.put((train_examples, samples_info))
                    else:
                        continue

                    self.check_and_load_new_model()
                    if debug_file:
                        debug_file.flush()

                    if self.device.type == 'cuda':
                        mem_cached_mb = torch.cuda.memory_cached() / 1000000
                        if mem_cached_mb > 8000:
                            print(f'Actor {self.actor_id} empty cached memory [{mem_cached_mb} MB]', file=sys.stderr)
                            torch.cuda.empty_cache()

                epoch_end = time.time()
                print(f"[Actor {self.actor_id}] epoch {epoch_id} finished, took {epoch_end - epoch_start}s", file=sys.stderr)

    def load_environments(self, file_paths, example_ids=None):
        from qa.table.experiments import load_environments
        trigger_words_dict = None
        if os.path.exists(self.config['trigger_words_file']):
            with open(self.config['trigger_words_file']) as f:
                trigger_words_dict = json.load(f)
                print('use trigger words in {}'.format(self.config['trigger_words_file']))
        envs = load_environments(file_paths,
                                 example_ids=example_ids,
                                 table_file=self.config['table_file'],
                                 bert_tokenizer=self.agent.encoder.bert_model.tokenizer,
                                 alpha_region=self.config['alpha_region'],
                                 alpha_op=self.config['alpha_op'],
                                 alpha_entity_link=self.config['alpha_entity_link'],
                                 trigger_words_dict=trigger_words_dict
                                 )

        setattr(self, 'environments', envs)

    def check_and_load_new_model(self):
        t1 = time.time()
        while True:
            new_model_path = self.checkpoint_queue.get()

            if new_model_path == self.model_path or os.path.exists(new_model_path):
                break
        print(f'[Actor {self.actor_id}] {time.time() - t1}s used to wait for new checkpoint', file=sys.stderr)

        if new_model_path != self.model_path:
            t1 = time.time()

            state_dict = torch.load(new_model_path, map_location=lambda storage, loc: storage)
            self.agent.load_state_dict(state_dict, strict=False)
            self.model_path = new_model_path

            t2 = time.time()
            print('[Actor %s] loaded new model [%s] (took %.2f s)' % (self.actor_id, new_model_path, t2 - t1), file=sys.stderr)

            return True
        else:
            return False

    def get_global_step(self):
        if not self.model_path:
            return 0

        model_name = self.model_path.split('/')[-1]
        train_iter = re.search('iter(\d+)?', model_name).group(1)

        return int(train_iter)
