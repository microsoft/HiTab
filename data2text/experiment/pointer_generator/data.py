"""Classes for data. 
* Vocab: vocabulary instance initiated from the voab file 
* Example: an source-target(-table-parent) example 
* Batch: a list of batched and masked examples 
* Batcher: load examples and batch them for model input 
"""


# %% Vocabulary

import csv 

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_STA = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
BOS_TOKEN = '[BOS]'  # This has a vocab id, which is used at the start of every decoder input sequence
EOS_TOKEN = '[EOS]'  # This has a vocab id, which is used at the end of untruncated target sequences
# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):
    """Vocabulary class. """

    def __init__(self, file: str, max_size: int):
        self.word2idx = {}
        self.idx2word = {}
        self.count = 0     # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [BOS] and [EOS] get the ids 0,1,2,3.
        for w in [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
            self.word2idx[w] = self.count
            self.idx2word[self.count] = w
            self.count += 1

        # Read the vocab file and add words up to max_size
        with open(file, 'r') as fin:
            for line in fin:
                items = line.split()
                if len(items) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s' % line.strip())
                    continue
                w = items[0]
                if w in [SENTENCE_STA, SENTENCE_END, UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [BOS] and [EOS] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self.word2idx:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self.word2idx[w] = self.count
                self.idx2word[self.count] = w
                self.count += 1
                if max_size != 0 and self.count >= max_size:
                    break
        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (
          self.count, self.idx2word[self.count - 1]))

    def word2id(self, word):
        if word not in self.word2idx:
            return self.word2idx[UNK_TOKEN]
        return self.word2idx[word]

    def id2word(self, word_id):
        if word_id not in self.idx2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.idx2word[word_id]

    def size(self):
        return self.count

    def write_metadata(self, path):
        print( "Writing word embedding metadata file to %s..." % (path))
        with open(path, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({"word": self.idx2word[i]})



# %% Example

from typing import List 
from experiment.pointer_generator import config, utils


class Example(object):
    """A hmt-table example. """

    def __init__(
        self, 
        source: List[List[str]], 
        target: List[List[str]], 
        table_parent: List, 
        vocab: Vocab, 
        config=config, 
    ):
        """Initialize the example with source and target texts.
        args:
            source: List[List[str]], list of parsed tokens/words-list.
            target: List[str], a single list of parsed target text. 
            table_parent: List[List[List[str], List[str]]], parsed attributes and fields. 
        """
        # Get ids of special tokens
        bos_decoding = vocab.word2id(BOS_TOKEN)
        eos_decoding = vocab.word2id(EOS_TOKEN)

        # process the source input
        src_words = [sword for sitem in source for sword in sitem]
        if len(src_words) > config.max_enc_steps:
            src_words = src_words[: config.max_enc_steps]
        self.enc_len = len(src_words)
        self.enc_inp = [vocab.word2id(w) for w in src_words]

        # process the target text
        tgt_words = target
        tgt_ids = [vocab.word2id(w) for w in tgt_words]
        # get the decoder input dequence and target sequence
        self.dec_inp, self.tgt = self.get_dec_seq(tgt_ids, 
            config.max_dec_steps, bos_decoding, eos_decoding)
        self.dec_len = len(self.dec_inp)

        # if using pg mode, need to store some extra info
        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id;
            # also store the in-article OOVs words themselves
            self.enc_inp_extend_vocab, self.article_oovs = utils.article2ids(src_words, vocab, config)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = utils.abstract2ids(tgt_words, vocab, self.article_oovs, config)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.tgt = self.get_dec_seq(abs_ids_extend_vocab, config.max_dec_steps, bos_decoding, eos_decoding)

        # store the original strings
        self.original_source = source
        self.original_target = target
        self.original_table_parent = table_parent
    
    def get_dec_seq(self, sequence: List[int], max_len: int, start_id: int, stop_id: int): 
        """Perform decoding seuqence processing, add special tokens and do truncation. """
        src = [start_id] + sequence[:]
        tgt = sequence[:]
        if len(src) > max_len:    # truncate
            src = src[: max_len]
            tgt = tgt[: max_len]  # no end_token
        else:  # no truncation
            tgt.append(stop_id)   # end token
        assert len(src) == len(tgt)
        return src, tgt

    def pad_enc_seq(self, max_len: int, pad_id: int) -> None: 
        """Pad the encoding sequence to config-specified max length. """
        while len(self.enc_inp) < max_len:
            self.enc_inp.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_inp_extend_vocab) < max_len:
                self.enc_inp_extend_vocab.append(pad_id)

    def pad_dec_seq(self, max_len: int, pad_id: int) -> None: 
        """Pad the decoding sequence to config-specified max length. """ 
        while len(self.dec_inp) < max_len:
            self.dec_inp.append(pad_id)
        while len(self.tgt) < max_len:
            self.tgt.append(pad_id)



# %% Batch 

import numpy as np


class Batch(object):
    def __init__(self, example_list: List[Example], vocab: Vocab, batch_size: int):
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list)  # initialize the input to the encoder
        self.init_decoder_seq(example_list)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list: List[Example]): 
        """Create self enc_batch/enc_lens/enc_padding_mask from the list of examples. """ 

        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_enc_seq(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch 
        # because we use dynamic_rnn for the encoder. 
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_inp[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if config.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_inp_extend_vocab[:]

    def init_decoder_seq(self, example_list: List[Example]): 
        """Create self dec_batch/tgt_batch/dec_lens/dec_padding_mask from the list of examples. """ 
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_dec_seq(config.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.tgt_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_inp[:]
            self.tgt_batch[i, :] = ex.tgt[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list: List[Example]): 
        self.original_sources = [ex.original_source for ex in example_list]  # list of lists
        self.original_targets = [ex.original_target for ex in example_list]  # list of lists
        self.original_table_parents = [ex.original_table_parent for ex in example_list]



# %% Batcher

import glob
import json
import time
import queue
import random
from threading import Thread
import tensorflow.compat.v1 as tf


class Batcher(object):
    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(
        self, vocab: Vocab, data_path: str,   # hidden-intend, naming with starting '_' 
        batch_size: int, single_pass: bool, mode: str, 
    ):
        self._vocab = vocab
        self._data_path = data_path 

        self.batch_size = batch_size
        self.single_pass = single_pass
        self.mode = mode

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1    # just one thread to batch examples
            self._bucketing_cache_size = 1   # only load one batch's worth of examples before bucketing
            self._finished_reading = False   # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 1  # num threads to fill example queue
            self._num_batch_q_threads = 1    # num threads to fill batch queue
            self._bucketing_cache_size = 1   # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:                   # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.logging.warning(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())
            if self.single_pass and self._finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def pair_generator(self, data_path: str, single_pass: bool, verbose: bool = False):
        """Generate hmt text pairs to construct examples. 
        Yield (source text, target text, and table parent list) for each turn. 
        """
        if verbose: print(f'[pair-generator] from data-path [{data_path}]')

        while True:
            filelist = glob.glob(data_path)
            assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty
            if single_pass: filelist = sorted(filelist)
            else: random.shuffle(filelist)

            for f in filelist:
                print(f'[pair-gen] reading from file: {f}')
                reader = open(f, 'r')
                for line in reader:
                    tabdict = json.loads(line.strip())   
                    if verbose: print(f"\n[pair-gen] got sample: \n{tabdict['source']}\n{tabdict['target']}")
                    yield (tabdict['source'], tabdict['target'], tabdict['table_parent'])
            if single_pass:
                print("example_generator completed reading all datafiles. No more data.")
                break
    
    def fill_example_queue(self):
        input_generator = self.pair_generator(self._data_path, self.single_pass)

        while True:
            try:
                (source, target, table_parent) = input_generator.__next__()
            except StopIteration:  # if there are no more examples:
                tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
                if self.single_pass:
                    tf.logging.info(
                        "single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")

            example = Example(source, target, table_parent, self._vocab)
            self._example_queue.put(example)

    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
                # beam search decode mode single example repeated in the batch
                ex = self._example_queue.get()
                b = [ex for _ in range(self.batch_size)]
                self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
            else:
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True)  # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])
                if not self.single_pass:
                    random.shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    batch = Batch(b, self._vocab, self.batch_size)
                    self._batch_queue.put(batch)

    def watch_threads(self):
        while True:
            tf.logging.info(
                'Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())

            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()