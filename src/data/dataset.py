from __future__ import absolute_import, division, print_function

import io
import math
from logging import getLogger
import numpy as np
import torch

from .vocab import Vocabulary

logger = getLogger()


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new['source']) + 1)
    max_tgt_in_batch = max(max_tgt_in_batch, len(new['target']) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def load_and_batch_input_data(input_file, params):
    assert hasattr(params, 'src_vocab')
    vocab = params.src_vocab
    examples = []
    for line in open(input_file, 'r'):
        example = {}
        tokens = line.strip().split()
        example['source'] = [vocab[tok] for tok in tokens] + [vocab.eos_index]
        examples.append(example)

        if len(examples) >= params.decode_batch_size:
            src_max_len = max(len(ex['source']) for ex in examples[:params.decode_batch_size])
            src_padded, src_lengths = [], []
            for ex in examples[:params.decode_batch_size]:
                src_seq = ex['source']
                src_padded.append(src_seq[:src_max_len] + [params.pad_index] * max(0, src_max_len - len(src_seq)))
                src_lengths.append(len(src_padded[-1]) - max(0, src_max_len - len(src_seq)))
            src_padded = torch.tensor(src_padded, dtype=torch.long)
            src_lengths = torch.tensor(src_lengths, dtype=torch.long)
            yield {'source': src_padded, 'source_length': src_lengths}
            examples = examples[params.decode_batch_size:]
    if len(examples) > 0:
        src_max_len = max(len(ex['source']) for ex in examples)
        src_padded, src_lengths = [], []
        for ex in examples:
            src_seq = ex['source']
            src_padded.append(src_seq[:src_max_len] + [params.pad_index] * max(0, src_max_len - len(src_seq)))
            src_lengths.append(len(src_padded[-1]) - max(0, src_max_len - len(src_seq)))
        src_padded = torch.tensor(src_padded, dtype=torch.long)
        src_lengths = torch.tensor(src_lengths, dtype=torch.long)
        yield {'source': src_padded, 'source_length': src_lengths}


def create_example(data, fields):
    example = {}
    assert len(data) == len(fields)
    for (field_name, vocab), tokens in zip(fields, data):
        example[field_name] = [vocab[tok] for tok in tokens] + [vocab.eos_index]
    return example


def create_batch(example_list, pad_index=0):
    src_max_len = max(len(example['source']) for example in example_list)
    tgt_max_len = max(len(example['target']) for example in example_list)
    src_padded, src_lengths, tgt_padded, tgt_lengths = [], [], [], []
    for example in example_list:
        src_seq = example['source']
        tgt_seq = example['target']
        src_padded.append(src_seq[:src_max_len] + [pad_index] * max(0, src_max_len - len(src_seq)))
        src_lengths.append(len(src_padded[-1]) - max(0, src_max_len - len(src_seq)))

        tgt_padded.append(tgt_seq[:tgt_max_len] + [pad_index] * max(0, tgt_max_len - len(tgt_seq)))
        tgt_lengths.append(len(tgt_padded[-1]) - max(0, tgt_max_len - len(tgt_seq)))
    src_padded = torch.tensor(src_padded, dtype=torch.long)
    src_lengths = torch.tensor(src_lengths, dtype=torch.long)
    tgt_padded = torch.tensor(tgt_padded, dtype=torch.long)
    tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.long)
    return {"source": src_padded, "source_length": src_lengths,
            "target": tgt_padded, "target_length": tgt_lengths}


class DataIterator(object):
    def __init__(self, dataset, params, train=True, repeat=None):
        self.dataset = dataset
        self.params = params
        self.train = train
        self.repeat = repeat if repeat is not None else train

        self.batch_size = params.batch_size
        self.constant = params.constant_batch_size

        self.iterations = 0
        self.epoches = 0
        self.epoch_size = 0

        # For state loading/saving only
        self.iterations_this_epoch = 0
        self._random_state_this_epoch = None
        self._restored_from_state = False

    def init_epoch(self):
        """Set up the batch generator for a new epoch."""
        self.epoches += 1
        self.iterations_this_epoch = 0
        if not self.repeat:
            self.iterations = 0

    @property
    def epoch(self):
        return self.epoches

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        if self.train:
            for idx, batch in enumerate(self.batch_examples(self.dataset, self.batch_size, constant=self.constant)):
                # fast-forward if loaded from state
                if self.iterations_this_epoch > idx:
                    logger.info("Passing batch.")
                    continue
                self.iterations += 1
                self.iterations_this_epoch += 1
                if self.iterations_this_epoch > self.epoch_size:
                    self.epoch_size = self.iterations_this_epoch
                yield batch
        else:
            for batch in self.batch_examples_keep_order(self.dataset, self.batch_size, constant=self.constant):
                yield batch

    def batch_examples(self, data, batch_size, constant=False, mantissa_bits=2):
        max_length = self.params.max_sequence_size or batch_size
        min_length = 8
        mantissa_bits = mantissa_bits

        x = min_length
        boundaries = []

        while x < max_length:
            boundaries.append(x)
            x += 2 ** max(0, int(math.log(x, 2)) - mantissa_bits)
        if not constant:
            batch_sizes = [max(1, batch_size // length) for length in boundaries + [max_length]]
            bucket_capacities = [2 * b for b in batch_sizes]
        else:
            batch_sizes = [batch_size for _ in boundaries + [max_length]]
            bucket_capacities = [2 * b for b in batch_sizes]
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]

        buckets = [[] for _ in range(len(buckets_min))]
        assert len(batch_sizes) == len(buckets)

        # The queue to bucket on will be chosen based on maximum length
        while True:
            self.init_epoch()
            for example in data:
                example_length = self._example_length_fn(example)
                bucket_index = self._which_bucket_fn(example_length, buckets_min, buckets_max)
                buckets[bucket_index].append(example)
                if len(buckets[bucket_index]) >= bucket_capacities[bucket_index]:
                    buckets[bucket_index] = sorted(buckets[bucket_index], key=lambda x: self._example_length_fn(x))
                    bucket_batch_size = batch_sizes[bucket_index]
                    yield create_batch(buckets[bucket_index][:bucket_batch_size], self.params.pad_index)
                    buckets[bucket_index] = buckets[bucket_index][bucket_batch_size:]
            if not self.repeat:
                for bucket_index in range(len(buckets)):
                    buckets[bucket_index] = sorted(buckets[bucket_index], key=lambda x: self._example_length_fn(x))
                    bucket_batch_size = batch_sizes[bucket_index]
                    if len(buckets[bucket_index]) > bucket_batch_size:
                        yield create_batch(buckets[bucket_index][:bucket_batch_size], self.params.pad_index)
                        yield create_batch(buckets[bucket_index][bucket_batch_size:], self.params.pad_index)
                    elif len(buckets[bucket_index]) > 0:
                        yield create_batch(buckets[bucket_index], self.params.pad_index)

                    buckets[bucket_index] = []
                return

    def _example_length_fn(self, example):
        return max(len(example['source']), len(example['target']))

    def _which_bucket_fn(self, length, buckets_min, buckets_max):
        assert len(buckets_min) == len(buckets_max)
        for bucket_index in range(len(buckets_min)):
            if buckets_min[bucket_index] <= length \
                    and length < buckets_max[bucket_index]:
                return bucket_index

    def batch_examples_keep_order(self, data, batch_size, constant=False):
        examples = []
        size_sofar = 0
        for example in data:
            if constant:
                examples.append(example)
                if len(examples) >= batch_size:
                    yield create_batch(examples[:batch_size], self.params.pad_index)
                    examples = examples[batch_size:]
            else:
                examples.append(example)
                size_sofar = batch_size_fn(example, len(examples), size_sofar)
                if size_sofar == batch_size:
                    yield create_batch(examples, self.params.pad_index)
                    examples, size_sofar = [], 0
                elif size_sofar > batch_size:
                    yield create_batch(examples[:-1], self.params.pad_index)
                    examples, size_sofar = examples[-1:], batch_size_fn(example, 1, size_sofar)
        if len(examples) > 0:
            yield create_batch(examples, self.params.pad_index)


class TranslationDataset(object):
    """
    A sentence pair
    """

    def __init__(self, src_filename, tgt_filename, src_vocab_file, tgt_vocab_file, params):
        """
        """
        self.params = params
        self.on_memory = params.on_memory
        self.src_filename = src_filename
        self.tgt_filename = tgt_filename
        self.src_vocab_file = src_vocab_file
        self.tgt_vocab_file = tgt_vocab_file

        self.src_vocab = Vocabulary(src_vocab_file)
        self.tgt_vocab = Vocabulary(tgt_vocab_file)

        self.fields = [('source', self.src_vocab), ('target', self.tgt_vocab)]

        if hasattr(params, 'src_vocab') or hasattr(params, 'tgt_vocab'):
            assert params.src_vocab == self.src_vocab
            assert params.tgt_vocab == self.tgt_vocab
        else:
            params.src_vocab = self.src_vocab
            params.tgt_vocab = self.tgt_vocab
            params.src_vocab_size = len(self.src_vocab)
            params.tgt_vocab_size = len(self.tgt_vocab)
            params.pad_index = self.tgt_vocab.pad_index
            params.eos_index = self.tgt_vocab.eos_index
            params.unk_index = self.tgt_vocab.unk_index

        self.data_size = -1

        if params.on_memory:
            self.examples = []
            with io.open(src_filename, mode='r', encoding='utf-8') as src_file, \
                    io.open(tgt_filename, mode='r', encoding='utf-8') as tgt_file:
                for src_line, tgt_line in zip(src_file, tgt_file):
                    src_tokens = src_line.strip().split()
                    tgt_tokens = tgt_line.strip().split()
                    if len(src_tokens) == 0 or len(tgt_tokens) == 0:
                        continue
                    if len(src_tokens) + 1 > params.max_sequence_size:
                        src_tokens = src_tokens[:params.max_sequence_size - 1]
                    if len(tgt_tokens) + 1 > params.max_sequence_size:
                        tgt_tokens = tgt_tokens[:params.max_sequence_size - 1]

                    self.examples.append(create_example([src_tokens, tgt_tokens], self.fields))
            self.data_size = len(self.examples)
            logger.info("Loaded %d examples on memory" % self.data_size)
        else:
            src_size = 0
            with io.open(src_filename, mode='r', encoding='utf-8') as src_file:
                for _ in src_file:
                    src_size += 1
            tgt_size = 0
            with io.open(tgt_filename, mode='r', encoding='utf-8') as tgt_file:
                for _ in tgt_file:
                    tgt_size += 1
            assert src_size == tgt_size
            self.data_size = src_size
            logger.info("Data size: %d" % self.data_size)

    def __len__(self):
        return self.data_size

    def __iter__(self):
        if self.on_memory:
            for x in self.examples:
                yield x
        else:
            with io.open(self.src_filename, mode='r', encoding='utf-8') as src_file, \
                    io.open(self.tgt_filename, mode='r', encoding='utf-8') as tgt_file:
                for src_line, tgt_line in zip(src_file, tgt_file):
                    src_tokens = src_line.strip().split()
                    tgt_tokens = tgt_line.strip().split()
                    if len(src_tokens) == 0 or len(src_tokens) + 1 > self.params.max_sequence_size or \
                            len(tgt_tokens) == 0 or len(tgt_tokens) + 1 > self.params.max_sequence_size:
                        continue
                    yield create_example([src_tokens, tgt_tokens], self.fields)


class Table2TextDataset(object):
    """
    table to text dataset.
    """

    def __init__(self, table_file, summary_file, table_vocab_file, summary_vocab_file, params):
        """
        """
        self.params = params
        self.table_file = table_file
        self.summary_file = summary_file
        self.table_vocab_file = table_vocab_file
        self.summary_vocab_file = summary_vocab_file

        self.src_vocab = Vocabulary(table_vocab_file)
        self.tgt_vocab = Vocabulary(summary_vocab_file)

        self.fields = [('source', self.table_vocab), ('target', self.summary_vocab)]

        if hasattr(params, 'src_vocab') or hasattr(params, 'tgt_vocab'):
            assert params.src_vocab == self.src_vocab
            assert params.tgt_vocab == self.tgt_vocab
        else:
            params.src_vocab = self.src_vocab
            params.tgt_vocab = self.tgt_vocab
            params.src_vocab_size = len(self.src_vocab)
            params.tgt_vocab_size = len(self.tgt_vocab)
            params.pad_index = params.tgt_vocab_size.pad_index
            params.eos_index = params.tgt_vocab_size.eos_index
            params.unk_index = params.tgt_vocab_size.unk_index

        self.data_size = -1

        # here we only implement the on-memory mode
        self.examples = []
