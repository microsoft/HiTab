import heapq
import json
import math
import random
import sys
from typing import List

import numpy as np

from qa.nsm.env import Environment, Trajectory, Sample


def normalize_probs(p_list):
    smoothing = 1.e-8
    p_list = np.array(p_list) + smoothing

    return p_list / p_list.sum()


class ReplayBuffer(object):
    def __init__(self, agent, shared_program_cache, discount_factor=1.0, debug=False):
        self.trajectory_buffer = dict()
        self.discount_factor = discount_factor
        self.agent = agent
        self.shared_program_cache = shared_program_cache
        self.env_program_prob_dict = dict()
        self.env_program_prob_sum_dict = dict()

    def load(self, envs: List[Environment], saved_programs_file_path: str):
        programs = json.load(open(saved_programs_file_path))

        trajectories = []
        n = 0
        total_env = 0
        n_found = 0
        for env in envs:
            total_env += 1
            found = False
            if env.name in programs:
                program_str_list = programs[env.name]
                n += len(program_str_list)
                env.cache._set = set(program_str_list)
                for program_str in program_str_list:
                    program = program_str.split()
                    try:
                        traj = Trajectory.from_program(env, program)
                    except ValueError:
                        print(f'Error loading program {program} for env {env.name}', file=sys.stderr)
                        continue

                    if traj is not None and traj.reward >= 1.:
                        trajectories.append(traj)
                        found = True
                        n_found += 1
                    else:
                        print('Fail ', env.name, traj.reward, program)

        print('@' * 100, file=sys.stderr)
        print('loading programs from file {}'.format(saved_programs_file_path), file=sys.stderr)
        print('at least 1 solution found fraction: {}'.format(
            float(n_found) / total_env), file=sys.stderr)

        self.save_trajectories(trajectories)
        print('{} programs in the file'.format(n), file=sys.stderr)
        print('{} programs extracted'.format(len(trajectories)), file=sys.stderr)
        print('{} programs in the buffer'.format(self.program_num), file=sys.stderr)
        print('@' * 100, file=sys.stderr)

    def has_found_solution(self, env_name):
        return env_name in self.trajectory_buffer and self.trajectory_buffer[env_name]

    def contains(self, traj: Trajectory):
        env_name = traj.environment_name
        if env_name not in self.trajectory_buffer:
            return False

        program = traj.program
        program_str = ' '.join(program)

        if program_str in self.env_program_prob_dict[env_name]:
            return True
        else:
            return False

    @property
    def size(self):
        n = 0
        for _, v in self.trajectory_buffer.items():
            n += len(v)
        return n

    @property
    def program_num(self):
        return sum(len(v) for v in self.env_program_prob_dict.values())

    def update_program_prob(self, env_name, program: List[str], prob: float):
        self.env_program_prob_dict[env_name][' '.join(program)] = prob
        self.shared_program_cache.update_hypothesis_prob(env_name, program, prob)

    def add_trajectory(self, trajectory: Trajectory, prob=None):
        program = trajectory.program

        self.shared_program_cache.add_trajectory(trajectory, prob)
        self.env_program_prob_dict.setdefault(trajectory.environment_name, dict())[' '.join(program)] = prob

        self.trajectory_buffer.setdefault(trajectory.environment_name, []).append(trajectory)

    def save_trajectories(self, trajectories):
        for trajectory in trajectories:
            if not self.contains(trajectory):
                self.add_trajectory(trajectory)

    def save_samples(self, samples: List[Sample], log=True):
        for sample in samples:
            if not self.contains(sample.trajectory):
                prob = math.exp(sample.prob) if log else sample.prob
                self.add_trajectory(sample.trajectory, prob=prob)  # prob ~ [0, 1]

    def all_samples(self, agent=None):
        samples = dict()
        for env_name, trajs in self.trajectory_buffer.items():
            samples[env_name] = [Sample(traj, prob=self.env_program_prob_dict[env_name][' '.join(traj.program)]) for traj in trajs]

        return samples

    def replay(self, environments, n_samples=1, use_top_k=False, truncate_at_n=0, replace=True, debug_file=None):
        select_env_names = set([e.name for e in environments])
        trajs = []

        # Collect all the trajs for the selected environments.
        for env_name in select_env_names:
            if env_name in self.trajectory_buffer:
                trajs += self.trajectory_buffer[env_name]

        if len(trajs) == 0:
            return []

        # chunk the trajectories, in case there are so many
        chunk_size = 64
        trajectory_probs = []
        for i in range(0, len(trajs), chunk_size):
            trajs_chunk = trajs[i: i + chunk_size]
            traj_chunk_probs = self.agent.compute_trajectory_prob(trajs_chunk, log=False)
            trajectory_probs.extend(traj_chunk_probs)

        # Put the samples into an dictionary keyed by env names.
        samples = [Sample(trajectory=t, prob=p) for t, p in zip(trajs, trajectory_probs)]
        env_sample_dict = dict()
        for sample in samples:
            env_name = sample.trajectory.environment_name
            env_sample_dict.setdefault(env_name, []).append(sample)

        replay_samples = []
        for env_name, samples in env_sample_dict.items():
            n = len(samples)

            # Compute the sum of prob of replays in the buffer.
            self.env_program_prob_sum_dict[env_name] = sum([sample.prob for sample in samples])

            for sample in samples:
                self.update_program_prob(env_name, sample.trajectory.program, sample.prob)

            # Truncated the number of samples in the selected
            # samples and in the buffer.
            if 0 < truncate_at_n < n:
                # Randomize the samples before truncation in case
                # when no prob information is provided and the trajs
                # need to be truncated randomly.
                random.shuffle(samples)
                samples = heapq.nlargest(
                    truncate_at_n, samples, key=lambda s: s.prob)

            if use_top_k:
                # Select the top k samples weighted by their probs.
                selected_samples = heapq.nlargest(
                    n_samples, samples, key=lambda s: s.prob)
                # replay_samples += normalize_probs(selected_samples)
            else:
                # Randomly samples according to their probs.
                p_samples = normalize_probs([sample.prob for sample in samples])

                if replace:
                    if self.agent.config['method'] in ['mml', 'sample']:
                        selected_sample_indices = np.random.choice(
                            len(samples),
                            size=len(samples),
                            p=p_samples,
                            replace=False)
                    else:
                        selected_sample_indices = np.random.choice(
                            len(samples),
                            size=n_samples,
                            p=p_samples)
                else:
                    sample_num = min(len(samples), n_samples)
                    selected_sample_indices = np.random.choice(len(samples), sample_num, p=p_samples, replace=False)

                selected_samples = [samples[i] for i in selected_sample_indices]

            selected_samples = [Sample(trajectory=sample.trajectory, prob=sample.prob) for sample in selected_samples]
            replay_samples += selected_samples

        return replay_samples