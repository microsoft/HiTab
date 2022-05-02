"""Run Test. 
* Beam 
* BeamSearch 
"""


import os
import time
import torch
from typing import List 

from . import config 
from .model.model import Model
from .data import Vocab, Batch, Batcher, BOS_TOKEN, EOS_TOKEN 
from .utils import outputids2words, get_input_from_batch 


class Beam(object):
    """A beam searched with probabilities and states. """ 

    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage
    
    def extend(self, token, log_prob, state, context, coverage):
        return Beam(
            tokens=self.tokens + [token], 
            log_probs=self.log_probs + [log_prob], 
            state=state,
            context=context,
            coverage=coverage
        )
    
    @property
    def latest_token(self):
        return self.tokens[-1]
    
    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)



# %% Beam Search

from datasets import load_metric
bleu_scorer = load_metric('bleu')
from ..utils.metrics import parent


class BeamSearch(object):
    """Beam search with loaded model to generate texts. """
    
    def __init__(self, args, model_path: str, file_path: str): 
        """Initialize an instance to perform beam search. 
        args: 
            model_path: str, path of model to load from 
            file_path: str, path of the test dataset (parsed) 
        """
        model_name = os.path.basename(model_path)
        self._test_dir = os.path.join(args.run_dir, f'decode_{model_name}')
        if not os.path.exists(self._test_dir): os.makedirs(self._test_dir)
        self._test_metrics_path = os.path.join(self._test_dir, 'metrics')
    
        self.vocab = Vocab(args.vocab_path, args.vocab_size)
        self.batcher = Batcher(
            vocab=self.vocab, 
            data_path=file_path, 
            batch_size=config.beam_size, 
            single_pass=True, 
            mode='decode',     
        )
        time.sleep(15)

        self.model = Model(
            config=config, 
            model_path=model_path, 
            is_eval=True, 
            is_transformer=False, 
        )
        
        self.use_cuda = config.use_gpu and torch.cuda.is_available()
        self.config = config

    def sort_beams(self, beams: List[Beam]) -> List[Beam]:
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def beam_search(self, batch: Batch):
        # single example repeated across the batch
        (
            enc_batch, enc_lens, enc_pos, enc_padding_mask, 
            enc_batch_extend_vocab, extra_zeros, c_t, coverage
        ) = get_input_from_batch(batch, self.use_cuda, self.config)
        enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_lens)
        s_t = self.model.reduce_state(enc_h)

        dec_h, dec_c = s_t     # b x hidden_dim
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, 
        # it has beam_size example initially everything is repeated
        beams = [
            Beam(
                tokens=[self.vocab.word2id(BOS_TOKEN)], 
                log_probs=[0.0], 
                state=(dec_h[0], dec_c[0]), 
                context=c_t[0], 
                coverage=(coverage[0] if self.config.is_coverage else None)
            )
            for _ in range(self.config.beam_size)
        ]

        steps = 0
        results = []
        while steps < self.config.max_dec_steps and len(results) < self.config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [
                t if (t < self.vocab.size()) 
                else self.vocab.word2id(self.config.UNK_TOKEN)
                for t in latest_tokens
            ]
            y_t = torch.autograd.Variable(torch.LongTensor(latest_tokens))
            if self.use_cuda: y_t = y_t.cuda()
            all_state_h = [h.state[0] for h in beams]
            all_state_c = [h.state[1] for h in beams]
            all_context = [h.context for h in beams]

            s_t = (
                torch.stack(all_state_h, 0).unsqueeze(0), 
                torch.stack(all_state_c, 0).unsqueeze(0)
            )
            c_t = torch.stack(all_context, 0)

            coverage_t = None
            if self.config.is_coverage:
                all_coverage = [h.coverage for h in beams]
                coverage_t = torch.stack(all_coverage, 0)
            
            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(
                y_t, s_t, enc_out, enc_fea, enc_padding_mask, c_t, 
                extra_zeros, enc_batch_extend_vocab, coverage_t, steps
            )
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, self.config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            # On the first step, we only had one original hypothesis (the initial hypothesis). 
            # On subsequent steps, all original hypotheses are distinct.
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage[i] if self.config.is_coverage else None)

                # for each of the top 2*beam_size hyps:
                for j in range(self.config.beam_size * 2):  
                    new_beam = h.extend(
                        token=topk_ids[i, j].item(), 
                        log_prob=topk_log_probs[i, j].item(), 
                        state=state_i, 
                        context=context_i, 
                        coverage=coverage_i
                    )
                    all_beams.append(new_beam)
                
            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(EOS_TOKEN):
                    if steps >= self.config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == self.config.beam_size or len(results) == self.config.beam_size:
                    break
            
            steps += 1

        if len(results) == 0:
            results = beams
        
        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]      # best_summary

    def run(self, interval: int = 1000): 
        """Run beam-search on each test sample. 
        interval: number of batch steps for logging info. 
        """
        counter = 0
        start = time.time()

        all_pred_tokens, all_ref_tokens = [], []

        batch = self.batcher.next_batch()
        while batch:  # not None or not Empty
            # run beam search to get best Hypothesis
            try: best_summary = self.beam_search(batch)
            except: break    # RuntimeError: Cannot pack empty tensors.

            # extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            article_oovs = batch.art_oovs[0] if self.config.pointer_gen else None
            decoded_words = outputids2words(
                id_list=output_ids, 
                vocab=self.vocab, 
                article_oovs=article_oovs, 
            )

            # remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(EOS_TOKEN)
                decoded_words = decoded_words[: fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            
            all_pred_tokens.append(decoded_words)
            all_ref_tokens.append([
                tgt for tgt in batch.original_targets  # [: 1]
            ])

            counter += 1
            if counter % interval == 0: 
                print(f'{counter:d} example in {int(time.time()-start):d} sec')
                start = time.time()
                print(f'ORG: {all_ref_tokens[-1]}')
                print(f'GEN: {decoded_words}')
            
            batch = self.batcher.next_batch()
        
        print(f'Decoder has finished reading dataset for single_pass.')
        print(f'Now starting BLEU eval...')
        results_dict = bleu_scorer.compute(
            predictions=all_pred_tokens, 
            references=all_ref_tokens, 
        )
        print(f"BLEU: {results_dict['bleu']:.4f}")

        with open(self._test_metrics_path, 'w') as fw:
            mline = f"bleu-4\t{results_dict['bleu']:.4f}\n"
            fw.write(mline)
            for idx, (pred, ref) in enumerate(zip(all_pred_tokens, all_ref_tokens)):
                iscore = bleu_scorer.compute(
                    predictions=[pred], 
                    references=[ref]
                ) 
                fw.write(f"#{idx}: {iscore['bleu']:.4f}\n")
                fw.write(f'{pred}\n{ref[0]}\n\n')
         
    def eval_parent(self, interval=1000):
        """Run beam-search on each test sample."""
        counter = 0
        start = time.time()

        all_pred_tokens, all_ref_tokens = [], []
        all_table_tokens = []

        batch = self.batcher.next_batch()
        while batch:  # not None or not Empty
            # run beam search to get best Hypothesis
            try: best_summary = self.beam_search(batch)
            except: break    # RuntimeError: Cannot pack empty tensors.

            # extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            article_oovs = batch.art_oovs[0] if self.config.pointer_gen else None
            decoded_words = outputids2words(
                id_list=output_ids, 
                vocab=self.vocab, 
                article_oovs=article_oovs, 
            )

            # remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(EOS_TOKEN)
                decoded_words = decoded_words[: fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            
            all_pred_tokens.append(decoded_words)
            all_ref_tokens.append([
                tgt for tgt in batch.original_targets[: 1]
            ])
            all_table_tokens.append(batch.original_table_parents[0])   # @zhiruow

            counter += 1
            if counter % interval == 0: 
                print(f'{counter:d} example in {int(time.time()-start):d} sec')
                start = time.time()
                print(f'TGT: {all_ref_tokens[-1]}')
                print(f'GEN: {decoded_words}')
                # print(f'TAB: {all_table_tokens[-1]}')
            
            batch = self.batcher.next_batch()
        
        print(f'Decoder has finished reading dataset for single_pass.')
        print(f'Now starting PARENT eval...')
        results_dict = parent(
            predictions=all_pred_tokens, 
            references=all_ref_tokens, 
            tables=all_table_tokens, 
            return_dict=True, 
        )
        print(f"PARENT: {results_dict['average_f1']:.4f}")