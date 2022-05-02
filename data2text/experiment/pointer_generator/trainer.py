"""Trainer for the Pointer-Generator Network. """

import os
import time
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import tensorflow.compat.v1 as tf

from . import config, utils
from .data import Batch, Vocab, Batcher
from .model.model import Model


class Trainer(object):
    """Epoch-wise train and validation."""

    def __init__(self, args):
        """Initialize the trainer with config.
        (vocab_path, vocab_size, train_data_path, batch_size, log_root)
        """
        self.args = args
        self.use_cuda = config.use_gpu and torch.cuda.is_available()

        self.vocab = Vocab(args.vocab_path, args.vocab_size)
        print(
            f'model load data in batch ({args.per_device_train_batch_size}) ', 
            f'from path: {args.train_data_path}'
        )
        self.batcher = Batcher(
            vocab=self.vocab, 
            data_path=args.train_data_path, 
            batch_size=args.per_device_train_batch_size, 
            single_pass=True, 
            mode='train', 
        )
        time.sleep(args.train_sleep_time)

        train_dir = os.path.join(args.run_dir, 'train')   # f'train_{int(time.time())}'
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        self.model_dir = os.path.join(train_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        with tf.Graph().as_default():
            self.summary_writer = tf.summary.FileWriter(train_dir)
        
        self.running_avg_loss = None

    def save_model(self, running_avg_loss, iepoch: int) -> None: 
        """Save the model state in path. 
        Includes (encoder/decoder/reduce-state), optimizer dict, and current loss. 
        """
        state = {
            'iepoch': iepoch,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, f'model_{iepoch:d}.bin')
        torch.save(state, model_save_path)

    def setup_train(self, model_path: str = None) -> float:
        """Set-up the starting iteration index and loss before activating the training process."""
        self.model = Model(config, model_path)
        # initial_lr = config.lr_coverage if config.is_coverage else config.lr
        initial_lr = self.args.learning_rate

        params = list(self.model.encoder.parameters()) + \
            list(self.model.decoder.parameters()) + \
            list(self.model.reduce_state.parameters())
        total_params = sum([param[0].nelement() for param in params])
        print(f'The Number of params of model: {total_params/1e3:.3f} k')
        self.optimizer = optim.Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
        print(f'trainer.optimizer with lr: {initial_lr:.6f}, acc_val: {config.adagrad_init_acc:.6f}')

        start_loss = 0.

        if model_path is not None:
            state = torch.load(model_path, map_location=lambda storage, location: storage)
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if self.use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_loss

    def train_one_batch(self, batch: Batch):
        """Execute one training step with a batch of data."""
        (
            enc_batch, enc_lens, enc_pos, enc_padding_mask, 
            enc_batch_extend_vocab, extra_zeros, c_t, coverage
        ) = utils.get_input_from_batch(batch, self.use_cuda, config)
        dec_batch, dec_lens, dec_pos, dec_padding_mask, max_dec_len, tgt_batch = \
            utils.get_output_from_batch(batch, self.use_cuda, config)

        self.optimizer.zero_grad()

        if not config.tran:
            enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_lens)
        else:
            enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_pos)

        s_t = self.model.reduce_state(enc_h)

        step_losses, cove_losses = [], []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t, c_t, attn_dist, p_gen, next_coverage = \
                self.model.decoder(
                    y_t, s_t, enc_out, enc_fea, enc_padding_mask, 
                    c_t, extra_zeros, enc_batch_extend_vocab, coverage, di
                )
            tgt = tgt_batch[:, di]
            step_mask = dec_padding_mask[:, di]
            gold_probs = torch.gather(final_dist, 1, tgt.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                cove_losses.append(step_coverage_loss * step_mask)
                coverage = next_coverage

            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        if config.is_coverage:
            cove_losses = torch.sum(torch.stack(cove_losses, 1), 1)
            batch_cove_loss = cove_losses / dec_lens
            batch_cove_loss = torch.mean(batch_cove_loss)
            if loss.item() == float('nan') or batch_cove_loss.item() == float('nan'):
                print('nan')
            return loss.item(), batch_cove_loss.item()

        return loss.item(), 0.

    def run_one_epoch(
        self, iepoch: int, 
        model_path: str = None, 
        interval: int = 1000, 
        save_model: bool = True, 
    ):
        if (iepoch == 0) or (self.running_avg_loss is None):
            self.running_avg_loss = self.setup_train(model_path)
            print(f'no.epoch {iepoch}, self avg loss: {self.running_avg_loss}')
            print(f'setup training loss {self.running_avg_loss} from model path: {model_path}')
        
        self.batcher = Batcher(
            vocab=self.vocab, 
            data_path=self.args.train_data_path, 
            batch_size=self.args.per_device_train_batch_size, 
            single_pass=True, 
            mode='train', 
        )
        time.sleep(self.args.train_sleep_time)

        start = time.time()
        
        i_iter = 0
        while True:
            batch = self.batcher.next_batch()
            if batch is None: break
            loss, cove_loss = self.train_one_batch(batch)
            self.running_avg_loss = utils.calc_running_avg_loss(
                loss, self.running_avg_loss, self.summary_writer, i_iter)
            if self.running_avg_loss == float('nan'):
                print(f'get NaN')
                break
            i_iter += 1

            if i_iter % interval == 0:
                self.summary_writer.flush()
                time_period = time.time() - start
                print(f'step: {i_iter:d}, second: {time_period:.2f}, '
                    f'loss: {loss:.3f}, cover_loss: {cove_loss:.3f}')
                start = time.time()
        
        if save_model == True:
            self.save_model(self.running_avg_loss, iepoch)
        
    def run(self, num_epochs):
        for iepoch in range(num_epochs):
            self.run_one_epoch(iepoch)