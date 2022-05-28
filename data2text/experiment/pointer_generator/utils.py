"""Utility functions for process and load data."""

import torch
import numpy as np 
import tensorflow.compat.v1 as tf

from torch.autograd import Variable

# %% get i/o from batch

def get_input_from_batch(batch, use_cuda, config):
    extra_zeros = None
    enc_lens = batch.enc_lens
    max_enc_len = np.max(enc_lens)
    enc_batch_extend_vocab = None
    batch_size = len(batch.enc_lens)
    enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()

    if config.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        # max_art_oovs is the max over all the article oov list in the batch
        if batch.max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

    c_t = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

    coverage = None
    if config.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    enc_pos = np.zeros((batch_size, max_enc_len))
    for i, inst in enumerate(batch.enc_batch):
        for j, w_i in enumerate(inst):
            if w_i != config.PAD:
                enc_pos[i, j] = (j + 1)
            else:
                break
    enc_pos = Variable(torch.from_numpy(enc_pos).long())

    if use_cuda:
        c_t = c_t.cuda()
        enc_pos = enc_pos.cuda()
        enc_batch = enc_batch.cuda()
        enc_padding_mask = enc_padding_mask.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()

    return enc_batch, enc_lens, enc_pos, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, c_t, coverage


def get_output_from_batch(batch, use_cuda, config):
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    batch_size = len(batch.dec_lens)
    dec_lens = Variable(torch.from_numpy(dec_lens)).float()
    tgt_batch = Variable(torch.from_numpy(batch.tgt_batch)).long()
    dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
    dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()

    dec_pos = np.zeros((batch_size, config.max_dec_steps))
    for i, inst in enumerate(batch.dec_batch):
        for j, w_i in enumerate(inst):
            if w_i != config.PAD:
                dec_pos[i, j] = (j + 1)
            else:
                break
    dec_pos = Variable(torch.from_numpy(dec_pos).long())

    if use_cuda:
        dec_lens = dec_lens.cuda()
        tgt_batch = tgt_batch.cuda()
        dec_batch = dec_batch.cuda()
        dec_padding_mask = dec_padding_mask.cuda()
        dec_pos = dec_pos.cuda()

    return dec_batch, dec_lens, dec_pos, dec_padding_mask, max_dec_len, tgt_batch


# calc
def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    if int(running_avg_loss) == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip

    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)

    return running_avg_loss


# %% article / abstract <=> id

def article2ids(article_words, vocab, config):
    ids = []
    oov = []
    unk_id = vocab.word2id(config.UNK_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oov:  # Add to list of OOVs
                oov.append(w)
            oov_num = oov.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oov


def abstract2ids(abstract_words, vocab, article_oovs, config):
    ids = []
    unk_id = vocab.word2id(config.UNK_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


# %% decode

def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, \
                ("Error: models produced a word ID that isn't in the vocabulary. "
                 "This should not happen in baseline (no pointer-generator) mode. ")
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: models produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                        i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words