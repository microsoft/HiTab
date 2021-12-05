from multiprocessing import Manager, Value
from typing import List, Any

from qa.nsm.env import Trajectory


class SharedProgramCache(object):
    def __init__(self):
        self.program_cache = Manager().dict()
        self.total_entry_count = Value('i')

    def add_hypothesis(self, env_name: str, program: List[Any], prob: float, human_readable_program: List[Any] = None):
        if env_name not in self.program_cache:
            self.program_cache[env_name] = dict()

        hypotheses = self.program_cache[env_name]
        hypotheses[' '.join(program)] = {
            'program': program,
            'human_readable_program': human_readable_program,
            'prob': prob,
        }
        self.program_cache[env_name] = hypotheses
        with self.total_entry_count.get_lock():
            self.total_entry_count.value += 1

    def add_trajectory(self, trajectory: Trajectory, prob: float):
        self.add_hypothesis(
            trajectory.environment_name,
            trajectory.program,
            prob,
            human_readable_program=trajectory.human_readable_program
        )

    def update_hypothesis_prob(self, env_name: str, program: List[Any], prob: float):
        hypotheses = self.program_cache[env_name]

        entry = hypotheses[' '.join(program)]
        entry['prob'] = prob
        hypotheses[' '.join(program)] = entry

        self.program_cache[env_name] = hypotheses

    def contains_env(self, env_name):
        return env_name in self.program_cache

    def get_hypotheses(self, env_name):
        if not self.contains_env(env_name):
            return []

        result = self.program_cache[env_name]
        result = [x for x in result.values() if x['prob'] is not None]
        result = sorted(result, key=lambda x: -x['prob'])

        return result

    def stat(self):
        num_envs = len(self.program_cache)
        num_entries = self.total_entry_count.value

        return {'num_envs': num_envs, 'num_entries': num_entries}

    def all_programs(self):
        programs = dict()
        for env_name, entries in self.program_cache.items():
            programs[env_name] = list(entries.values())

        return programs
