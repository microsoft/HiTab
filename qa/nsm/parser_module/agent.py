import collections
import math
import sys
from collections import OrderedDict
from typing import Dict, List

import torch
from torch import nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from qa.nsm import nn_util
from qa.nsm.env import Trajectory, Observation, Sample, QAProgrammingEnv
from qa.nsm.parser_module.bert_decoder import BertDecoder
from qa.nsm.parser_module.bert_encoder import BertEncoder
from qa.nsm.parser_module.decoder import DecoderBase, Hypothesis, DecoderState
from qa.nsm.parser_module.encoder import EncoderBase


class PGAgent(nn.Module):
    "Agent trained by policy gradient."

    def __init__(
        self,
        encoder: EncoderBase, decoder: DecoderBase,
        config: Dict
    ):
        super(PGAgent, self).__init__()

        self.config = config

        self.encoder = encoder
        self.decoder = decoder

    @property
    def memory_size(self):
        return self.decoder.memory_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def sufficient_context_encoding_entries(self):
        return ['question_encoding', 'question_mask', 'question_encoding_att_linear']

    def encode(self, env_context):
        return self.encoder.encode(env_context)

    def compute_trajectory_actions_prob(self, trajectories: List[Trajectory], return_info=False) -> torch.Tensor:
        contexts = [traj.context for traj in trajectories]
        context_encoding = self.encoder.encode(contexts)
        state_tm1 = init_state = self.decoder.get_initial_state(context_encoding)

        batched_observation_seq, tgt_actions_info = Trajectory.to_batched_sequence_tensors(trajectories,
                                                                                           self.memory_size)

        # moved to device
        batched_observation_seq.to(self.device)

        # tgt_action_id (batch_size, max_action_len)
        # tgt_action_mask (batch_size, max_action_len)
        tgt_action_id, tgt_action_mask = tgt_actions_info['tgt_action_ids'], tgt_actions_info['tgt_action_mask']
        tgt_action_id = tgt_action_id.to(self.device)
        tgt_action_mask = tgt_action_mask.to(self.device)

        max_time_step = batched_observation_seq.read_ind.size(1)
        action_logits = []
        for t in range(max_time_step):
            obs_slice_t = batched_observation_seq.slice(t)

            # mem_logits: (batch_size, memory_size)
            mem_logits, state_t = self.decoder.step(obs_slice_t, state_tm1, context_encoding)

            action_logits.append(mem_logits)
            state_tm1 = state_t

        # (max_action_len, batch_size, memory_size)
        action_logits = torch.stack(action_logits, dim=0).permute(1, 0, 2)

        # (batch_size, max_action_len, memory_size)
        action_log_probs = nn_util.masked_log_softmax(action_logits, batched_observation_seq.valid_action_mask)

        # (batch_size, max_action_len)
        tgt_action_log_probs = torch.gather(action_log_probs, dim=-1, index=tgt_action_id.unsqueeze(-1)).squeeze(
            -1) * tgt_action_mask

        # (batch_size)
        traj_log_prob = tgt_action_log_probs.sum(dim=-1)

        if return_info:
            info = dict(
                action_log_probs=action_log_probs,
                tgt_action_id=tgt_action_id,
                tgt_action_mask=tgt_action_mask,
                action_logits=action_logits,
                valid_action_mask=batched_observation_seq.valid_action_mask,
                context_encoding=context_encoding
            )

            return traj_log_prob, info

        return traj_log_prob

    def compute_trajectory_prob(self, trajectories: List[Trajectory], log=True) -> torch.Tensor:
        with torch.no_grad():
            traj_log_prob = self.forward(trajectories)

            if not log:
                traj_log_prob = traj_log_prob.exp()

            return traj_log_prob.tolist()

    def forward(self, trajectories: List[Trajectory], entropy=False, return_info=False):
        # (batch_size, max_action_len)
        traj_log_prob, meta_info = self.compute_trajectory_actions_prob(trajectories, return_info=True)

        # compute entropy
        if entropy:
            # (batch_size, max_action_len, memory_size)
            logits = meta_info['action_logits']
            action_log_probs = meta_info['action_log_probs']
            # (batch_size, max_action_len, memory_size)
            valid_action_mask = meta_info['valid_action_mask']
            # (batch_size, max_action_len)
            tgt_action_mask = meta_info['tgt_action_mask']

            # masked_logits = logits * tgt_action_mask + (1. - tgt_action_mask) * -1.e30  # mask logits with a very negative number

            # max_z, pos = torch.max(masked_logits, dim=-1, keepdim=True)
            # z = masked_logits - max_z
            # exp_z = torch.util(z)
            # (batch_size, max_action_len)
            # sum_exp_z = torch.sum(exp_z, dim=-1, keepdim=True)

            p_action = nn_util.masked_softmax(logits, mask=valid_action_mask)
            # neg_log_action = torch.log(sum_exp_z) - z

            H = - p_action * action_log_probs * valid_action_mask
            # H = p_action * neg_log_action
            H = torch.sum(H, dim=-1).sum(dim=-1) / tgt_action_mask.sum(-1)

            return traj_log_prob, H

        if return_info:
            return traj_log_prob, meta_info

        return traj_log_prob

    def sample(
        self, environments, sample_num, use_cache=False,
    ):
        if sample_num == 0:  # 10
            return []

        if use_cache:  # True
            # if already explored everything, then don't explore this environment anymore.
            environments = [env for env in environments if not env.cache.is_full()]

        duplicated_envs = []
        for env in environments:  # won't change env in batch_env of actor
            for i in range(sample_num):
                duplicated_envs.append(env.clone())

        environments = duplicated_envs
        for env in environments:
            env.use_cache = use_cache

        completed_envs = []
        active_envs = environments

        env_context = [env.get_context() for env in environments]
        context_encoding = self.encode(env_context)  # bert encoding

        observations_tm1 = [env.start_ob for env in environments]
        state_tm1 = self.decoder.get_initial_state(context_encoding)
        sample_probs = torch.zeros(len(environments), device=self.device)

        while True:
            batched_ob_tm1 = Observation.to_batched_input(observations_tm1, memory_size=self.memory_size).to(
                self.device)
            mem_logits, state_t = self.decoder.step(observations_tm1, state_tm1, context_encoding=context_encoding)

            # (batch_size)
            sampled_action_t_id, sampled_action_t_prob = self.sample_action(mem_logits,
                                                                            batched_ob_tm1.valid_action_mask,
                                                                            return_log_prob=True)

            sample_probs = sample_probs + sampled_action_t_prob

            observations_t = []
            new_active_env_pos = []
            new_active_envs = []
            has_completed_sample = False
            for env_id, (env, action_t) in enumerate(zip(active_envs, sampled_action_t_id.tolist())):
                action_rel_id = env.valid_actions.index(action_t)
                ob_t, _, _, info = env.step(action_rel_id)
                if env.done:
                    completed_envs.append((env, sample_probs[env_id].item()))
                    has_completed_sample = True
                else:
                    if ob_t.valid_action_indices:
                        observations_t.append(ob_t)
                        new_active_env_pos.append(env_id)
                        new_active_envs.append(env)
                    else:
                        # force recomputing source context encodings since this environment
                        # is finished
                        has_completed_sample = True

            if not new_active_env_pos:  # end condition
                break

            if has_completed_sample:
                # need to perform slicing
                for key in self.sufficient_context_encoding_entries:
                    context_encoding[key] = context_encoding[key][new_active_env_pos]

                state_tm1 = state_t[new_active_env_pos]
                sample_probs = sample_probs[new_active_env_pos]
            else:
                state_tm1 = state_t

            observations_tm1 = observations_t
            active_envs = new_active_envs

        samples = []
        for env_id, (env, prob) in enumerate(completed_envs):
            if not env.error:
                traj = Trajectory.from_environment(env)
                samples.append(Sample(trajectory=traj, prob=prob))

        return samples

    def new_beam_search(self, environments, beam_size, use_cache=False, return_list=False):
        # if already explored everything, then don't explore this environment anymore.
        if use_cache:
            # if already explored everything, then don't explore this environment anymore.
            environments = [env for env in environments if not env.cache.is_full()]

        for env in environments:
            env.use_cache = use_cache

        CandidateHyp = collections.namedtuple('CandidateHyp',
                                              ['prev_hyp_env', 'action_id', 'rel_action_id', 'score',
                                               'prev_hyp_abs_pos'])

        batch_size = len(environments)
        # max_live_hyp_num = 1
        # live_beam_names = [env.name for env in environments]

        beams = OrderedDict((env.name, [Hypothesis(env=env, score=0.)]) for env in environments)
        completed_hyps = OrderedDict((env.name, []) for env in environments)
        # empty_hyp = dict(env=None, score=float('-inf'), ob=Observation.empty(), parent_beam_abs_pos=0)

        # (env_num, ...)
        env_context = [env.get_context() for env in environments]
        context_encoding_expanded = context_encoding = self.encode(env_context)  # a dict of embedding

        observations_tm1 = [env.start_ob for env in environments]
        state_tm1 = self.decoder.get_initial_state(context_encoding)
        hyp_scores_tm1 = torch.zeros(batch_size, device=self.device)

        # collect input tables for each example
        env_logging_info = {
            env.name: {
                'input_table': context_encoding['table_bert_encoding']['input_tables'][env_idx]
            }
            for env_idx, env
            in enumerate(environments)
        }

        while beams:
            batched_ob_tm1 = Observation.to_batched_input(observations_tm1, memory_size=self.memory_size).to(
                self.device)

            # (env_num, memory_size)
            action_probs_t, state_t = self.decoder.step_and_get_action_scores_t(batched_ob_tm1, state_tm1,
                                                                                context_encoding=context_encoding_expanded)
            action_probs_t[(1 - batched_ob_tm1.valid_action_mask).bool()] = float('-inf')

            # (env_num, memory_size)
            cont_cand_hyp_scores = action_probs_t + hyp_scores_tm1.unsqueeze(-1)
            cont_cand_hyp_scores = cont_cand_hyp_scores

            # collect hypotheses
            beam_start = 0
            continuing_candidates = OrderedDict()
            new_beams = OrderedDict()

            observations_t = []
            new_hyp_parent_abs_pos_list = []
            new_hyp_scores = []
            for env_name, beam in beams.items():
                live_beam_size = len(beam)
                beam_end = beam_start + live_beam_size
                # (beam_size, memory_size)
                beam_new_cont_scores = cont_cand_hyp_scores[beam_start: beam_end]
                continuing_candidates[env_name] = []

                for prev_hyp_id, prev_hyp in enumerate(beam):
                    _cont_action_scores = beam_new_cont_scores[prev_hyp_id][
                        prev_hyp.env.obs[-1].valid_action_indices].cpu()
                    for rel_action_id, new_hyp_score in enumerate(_cont_action_scores):
                        abs_action_id = prev_hyp.env.obs[-1].valid_action_indices[rel_action_id]
                        new_hyp_score = new_hyp_score.item()
                        if not math.isinf(new_hyp_score):
                            candidate_hyp = CandidateHyp(
                                prev_hyp_env=prev_hyp.env,
                                rel_action_id=rel_action_id,
                                action_id=abs_action_id,
                                score=new_hyp_score,
                                prev_hyp_abs_pos=beam_start + prev_hyp_id
                            )

                            is_compatible = True

                            if is_compatible:
                                continuing_candidates[env_name].append(candidate_hyp)

                # rank all hypotheses together with completed ones
                all_candidates = completed_hyps[env_name] + continuing_candidates[env_name]
                all_candidates.sort(key=lambda hyp: hyp.score, reverse=True)

                # top_k_candidates = heapq.nlargest(beam_size, all_candidates, key=lambda x: x.score)
                completed_hyps[env_name] = []

                def _add_hypothesis_to_new_beam(_hyp):

                    if isinstance(_hyp, Hypothesis):
                        completed_hyps[env_name].append(_hyp)
                    else:
                        new_hyp_env = _hyp.prev_hyp_env.clone()

                        ob_t, _, _, info = new_hyp_env.step(_hyp.rel_action_id)

                        if new_hyp_env.done:
                            if not new_hyp_env.error:
                                new_hyp = Hypothesis(env=new_hyp_env, score=_hyp.score)
                                completed_hyps[new_hyp_env.name].append(new_hyp)
                        else:
                            new_hyp = Hypothesis(env=new_hyp_env, score=_hyp.score)
                            new_beams.setdefault(env_name, []).append(new_hyp)

                            new_hyp_parent_abs_pos_list.append(_hyp.prev_hyp_abs_pos)
                            observations_t.append(ob_t)
                            new_hyp_scores.append(_hyp.score)

                new_beam_size = 0

                for cand_hyp in all_candidates:
                    if new_beam_size < beam_size:
                        _add_hypothesis_to_new_beam(cand_hyp)

                    new_beam_size += 1

                beam_start = beam_end

            if len(new_beams) == 0:
                break

            new_hyp_state_t = [(s[0][new_hyp_parent_abs_pos_list], s[1][new_hyp_parent_abs_pos_list]) for s in
                               state_t.state]
            new_hyp_memory_t = state_t.memory[new_hyp_parent_abs_pos_list]

            state_tm1 = DecoderState(state=new_hyp_state_t, memory=new_hyp_memory_t)
            observations_tm1 = observations_t
            hyp_scores_tm1 = torch.tensor(new_hyp_scores, device=self.device)

            for key in self.sufficient_context_encoding_entries:
                context_encoding_expanded[key] = context_encoding_expanded[key][new_hyp_parent_abs_pos_list]

            beams = new_beams

        if not return_list:
            # rank completed hypothesis
            for env_name in completed_hyps.keys():
                sorted_hyps = sorted(completed_hyps[env_name], key=lambda hyp: hyp.score, reverse=True)[:beam_size]
                completed_hyps[env_name] = [
                    Sample(
                        trajectory=Trajectory.from_environment(hyp.env),
                        prob=hyp.score,
                        logging_info=env_logging_info[env_name]
                    ) for hyp in sorted_hyps
                ]

            return completed_hyps
        else:
            samples_list = []
            for env_name, _hyps in completed_hyps.items():
                samples = [
                    Sample(
                        trajectory=Trajectory.from_environment(hyp.env),
                        prob=hyp.score,
                        logging_info=env_logging_info[env_name]
                    ) for hyp in _hyps
                ]
                samples_list.extend(samples)
            return samples_list

    def decode_examples(self, environments: List[QAProgrammingEnv], beam_size, batch_size=2):
        decode_results = []

        with torch.no_grad():
            batch_iter = nn_util.batch_iter(environments, batch_size, shuffle=False)
            for batched_envs in tqdm(batch_iter, total=len(environments) // batch_size, file=sys.stdout):

                batch_decode_result = self.new_beam_search(
                    batched_envs,
                    beam_size=beam_size,
                )
                # print(f"batch_decode_result: {batch_decode_result}")
                batch_decode_result = list(batch_decode_result.values())
                decode_results.extend(batch_decode_result)

        return decode_results

    def sample_action(self, logits, valid_action_mask, return_log_prob=False):
        """
        logits: (batch_size, action_num)
        valid_action_mask: (batch_size, action_num)
        """

        # p_actions = nn_util.masked_softmax(logits, mask=valid_action_mask)
        logits.masked_fill_((1 - valid_action_mask).bool(), -math.inf)
        p_actions = F.softmax(logits, dim=-1)
        # (batch_size, 1)
        sampled_actions = torch.multinomial(p_actions, num_samples=1)

        if return_log_prob:
            log_p_actions = nn_util.masked_log_softmax(logits, mask=valid_action_mask)
            log_prob = torch.gather(log_p_actions, dim=1, index=sampled_actions).squeeze(-1)

            return sampled_actions.squeeze(-1), log_prob

        return sampled_actions.squeeze(-1)

    @classmethod
    def build(cls, config, params=None, master=None):
        encoder = BertEncoder.build(config, master=master)
        decoder = BertDecoder.build(config, encoder, master=master)

        return cls(
            encoder, decoder,
            config=config
        )

    def save(self, model_path, kwargs=None):
        ddp = None
        if isinstance(self.encoder.bert_model, nn.DataParallel):
            ddp = self.encoder.bert_model
            self.encoder.bert_model = ddp.module

        params = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'kwargs': kwargs
        }

        if ddp:
            self.encoder.bert_model = ddp

        torch.save(params, model_path)

    @classmethod
    def load(cls, model_path, gpu_id=-1, default_values_handle=None, **kwargs):
        device = torch.device("cuda:%d" % gpu_id if gpu_id >= 0 else "cpu")
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        config = params['config']

        if default_values_handle:
            default_values_handle(config)
        config.update(kwargs)
        kwargs = params['kwargs'] if params['kwargs'] is not None else dict()

        model_cls = cls

        model = model_cls.build(config)
        incompatible_keys = model.load_state_dict(params['state_dict'], strict=False)
        if incompatible_keys.missing_keys:
            print('Loading agent, got missing keys {}'.format(incompatible_keys.missing_keys), file=sys.stderr)
        if incompatible_keys.unexpected_keys:
            print('Loading agent, got unexpected keys {}'.format(incompatible_keys.unexpected_keys), file=sys.stderr)

        model = model.to(device)
        model.eval()

        return model
