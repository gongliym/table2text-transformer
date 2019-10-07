from __future__ import absolute_import, division, print_function

import io
import math
import random
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
    max_tgt_in_batch = max(max_tgt_in_batch, len(new['summary']) + 1)
    tgt_elements = count * max_tgt_in_batch
    return tgt_elements

def create_batch(example_list, pad_index=0):
    # source sides are fixed length table data
    assert 1 == len(set([len(example['table_entity']) for example in example_list]))
    assert 1 == len(set([len(example['table_type']) for example in example_list]))
    assert 1 == len(set([len(example['table_value']) for example in example_list]))
    assert 1 == len(set([len(example['table_feature']) for example in example_list]))
    assert 1 == len(set([len(example['table_label']) for example in example_list]))
    for example in example_list:
        assert len(example['table_entity']) == len(example['table_type']) == len(example['table_value'])
        assert len(example['table_value']) == len(example['table_feature']) == len(example['table_label'])
    max_len = max(len(example['summary']) for example in example_list)
    table_entities = []
    table_typies   = []
    table_values   = []
    table_features = []
    table_labels   = []
    table_lengths  = []
    summaries      = []
    summary_lengths= []
    for example in example_list:
        table_entities.append(example['table_entity'])
        table_typies.append(example['table_type'])
        table_values.append(example['table_value'])
        table_features.append(example['table_feature'])
        table_labels.append(example['table_label'])
        table_lengths.append(len(example['table_label']))
        summaries.append(example['summary'][:max_len] + [pad_index] * max(0, max_len - len(example['summary'])))
        summary_lengths.append(len(summaries[-1]) - max(0, max_len - len(example['summary'])))

    table_entities = torch.tensor(table_entities, dtype=torch.long)
    table_typies = torch.tensor(table_typies, dtype=torch.long)
    table_values = torch.tensor(table_values, dtype=torch.long)
    table_features = torch.tensor(table_features, dtype=torch.long)
    table_labels = torch.tensor(table_labels, dtype=torch.long)
    table_lengths = torch.tensor(table_lengths, dtype=torch.long)
    summaries = torch.tensor(summaries, dtype=torch.long)
    summary_lengths = torch.tensor(summary_lengths, dtype=torch.long)
    return {
        'table_entity': table_entities,
        'table_type': table_typies,
        'table_value': table_values,
        'table_feature': table_features,
        'table_label': table_labels,
        'table_length': table_lengths, 
        'summary': summaries,
        'summary_lengths': summary_lengths
    }
    
def create_example(table_seq, table_label_seq, summary_seq, table_vocab, summary_vocab):
    assert len(table_seq) == len(table_label_seq)
    table_entity = []
    table_type = []
    table_value = []
    table_feature = []
    table_label = []
    
    for item, label in zip(table_seq, table_label_seq):
        fields = item.split('|')
        assert len(fields) == 4
        table_entity.append(table_vocab[fields[0]])
        table_type.append(table_vocab[fields[1]])
        table_value.append(table_vocab[fields[2]])
        table_feature.append(table_vocab[fields[3]])
        table_label.append(int(label))

    summary = [summary_vocab[tok] for tok in summary_seq] + [summary_vocab.eos_index]
    
    example = {
        'table_entity': table_entity,
        'table_type': table_type,
        'table_value': table_value,
        'table_feature': table_feature,
        'table_label': table_label,
        'summary': summary
    }
        
    return example


class Table2TextDataIterator(object):
    """
    """

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
            for idx, batch in enumerate(self.batch_examples(self.dataset, self.batch_size, constant=self.constant,
                                        shuffle=True, repeat=self.repeat)):
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
            for batch in self.batch_examples(self.dataset, self.batch_size, constant=self.constant,
                                             shuffle=False, repeat=False):
                yield batch
            
    def batch_examples(self, data, batch_size, constant=False, shuffle=True, repeat=True):
        examples = []
        size_sofar = 0
        while True:
            self.init_epoch()
            example_indexes = list(range(len(data)))
            if shuffle:
                random.shuffle(example_indexes)
            for idx in example_indexes:
                example = data[idx]
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
            if not repeat:
                if len(examples) > 0:
                    yield create_batch(examples, self.params.pad_index)
                return


class Table2TextDataset(object):
    """
    table to text dataset.
    """

    def __init__(self, table_file, table_label_file, summary_file, table_vocab_file, summary_vocab_file, params):
        """
        """
        self.params = params
        self.table_file = table_file
        self.table_file = table_file
        self.summary_file = summary_file
        self.table_vocab_file = table_vocab_file
        self.summary_vocab_file = summary_vocab_file

        self.src_vocab = Vocabulary(table_vocab_file)
        self.tgt_vocab = Vocabulary(summary_vocab_file)

        if hasattr(params, 'src_vocab') or hasattr(params, 'tgt_vocab'):
            assert params.src_vocab == self.src_vocab
            assert params.tgt_vocab == self.tgt_vocab
        else:
            params.src_vocab = self.src_vocab
            params.tgt_vocab = self.tgt_vocab
            params.src_vocab_size = len(self.src_vocab)
            params.tgt_vocab_size = len(self.tgt_vocab)
            params.pad_index = params.tgt_vocab.pad_index
            params.eos_index = params.tgt_vocab.eos_index
            params.unk_index = params.tgt_vocab.unk_index

        self.data_size = -1

        # here we only implement the on-memory mode
        table_length = -1
        self.examples = []
        with io.open(table_file, mode='r', encoding='utf-8') as table_inf, \
            io.open(table_label_file, mode='r', encoding='utf-8') as table_label_inf, \
            io.open(summary_file, mode='r', encoding='utf-8') as summary_inf:
            for table_line, table_label_line, summary_line in zip(table_inf, table_label_inf, summary_inf):
                src_tokens = table_line.strip().split()
                src_labels = table_label_line.strip().split()
                tgt_tokens = summary_line.strip().split()
                assert len(src_tokens) == len(src_labels)
                if len(src_tokens) == 0 or len(tgt_tokens) == 0:
                    continue
                if len(tgt_tokens) + 1 > params.max_sequence_size:
                    tgt_tokens = tgt_tokens[:params.max_sequence_size - 1]

                # sanity check
                if table_length == -1:
                    table_length = len(src_tokens)
                else:
                    assert len(src_tokens) == table_length

                self.examples.append(create_example(src_tokens, src_labels, tgt_tokens, params.src_vocab, params.tgt_vocab))
            self.data_size = len(self.examples)
            logger.info("Loaded %d examples on memory" % self.data_size)

    def __len__(self):
        return self.data_size

    def __getitem__(self, key):
        return self.examples[key]
        

                
