from __future__ import absolute_import, division, print_function

import io
import math
import random
from logging import getLogger
import numpy as np
import torch

from .vocab import Vocabulary

logger = getLogger()


global max_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_in_batch
    if count == 1:
        max_in_batch = 0
    max_in_batch = max(max_in_batch, len(new['sentence']))
    elements = count * max_in_batch
    return elements

def load_and_batch_data(input_file, params):
    assert hasattr(params, 'src_vocab')
    vocab = params.src_vocab
    examples = []
    for line in open(input_file, 'r'):
        fields = line.strip().split('\t')
        assert len(fields) >= 3
        sentence, entity_index, value_index = fields[:3]
        token_ids = [params.src_vocab[token] for token in sentence.split()]

        example = {'sentence': token_ids, 'sentence_length': len(token_ids), 
                   'entity_index': int(entity_index), 'value_index': int(value_index)}
        examples.append(example)

        if len(examples) >= params.decode_batch_size:
            max_len = max(len(example['sentence']) for example in examples[:params.decode_batch_size])
            sentence_padded, lengths, entity_indexes, value_indexes = [], [], [], []
            for example in examples[:params.decode_batch_size]:
                sentence_padded.append(example['sentence'] + 
                                       [params.pad_index] * max(0, max_len - len(example['sentence'])))
                lengths.append(example['sentence_length'])
                entity_indexes.append(example['entity_index'])
                value_indexes.append(example['value_index'])

            sentence_tensor = torch.tensor(sentence_padded, dtype=torch.long)
            lengths_tensor = torch.tensor(lengths, dtype=torch.long)
            entity_index_tensor = torch.tensor(entity_indexes, dtype=torch.long)
            value_index_tensor  = torch.tensor(value_indexes,  dtype=torch.long)
            yield {
                'sentence': sentence_tensor,
                'sentence_length': lengths_tensor,
                'entity_index': entity_index_tensor,
                'value_index' : value_index_tensor,
            }
            examples = examples[params.decode_batch_size:]

    if len(examples) > 0:
        max_len = max(len(example['sentence']) for example in examples)
        sentence_padded, lengths, entity_indexes, value_indexes = [], [], [], []
        for example in examples:
            sentence_padded.append(example['sentence'] + 
                                   [params.pad_index] * max(0, max_len - len(example['sentence'])))
            lengths.append(example['sentence_length'])
            entity_indexes.append(example['entity_index'])
            value_indexes.append(example['value_index'])

        sentence_tensor = torch.tensor(sentence_padded, dtype=torch.long)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        entity_index_tensor = torch.tensor(entity_indexes, dtype=torch.long)
        value_index_tensor  = torch.tensor(value_indexes,  dtype=torch.long)
        yield {
            'sentence': sentence_tensor,
            'sentence_length': lengths_tensor,
            'entity_index': entity_index_tensor,
            'value_index' : value_index_tensor,
        }

def create_example(sentence, entity_index, value_index, label, params):
    sentence_tokens = sentence.strip().split()
    assert entity_index >= 0 and value_index >= 0
    assert entity_index < len(sentence_tokens) and value_index < len(sentence_tokens)
    assert label in params.tgt_label
    
    token_ids = [params.src_vocab[token] for token in sentence_tokens]
    label_id = params.tgt_label[label]

    example = {
        'sentence': token_ids,
        'sentence_length': len(token_ids),
        'entity_index': entity_index,
        'value_index': value_index,
        'label': label_id
    }
    return example

def create_batch(example_list, pad_index=0):
    max_len = max(len(example['sentence']) for example in example_list)
    sentence_padded, lengths, entity_indexes, value_indexes, labels = [], [], [], [], []
    for example in example_list:
        sentence_padded.append(example['sentence'] + [pad_index] * max(0, max_len - len(example['sentence'])))
        lengths.append(example['sentence_length'])
        entity_indexes.append(example['entity_index'])
        value_indexes.append(example['value_index'])
        labels.append(example['label'])

    sentence_tensor = torch.tensor(sentence_padded, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    entity_index_tensor = torch.tensor(entity_indexes, dtype=torch.long)
    value_index_tensor  = torch.tensor(value_indexes,  dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    
    return {
        'sentence': sentence_tensor,
        'sentence_length': lengths_tensor,
        'entity_index': entity_index_tensor,
        'value_index' : value_index_tensor,
        'label': label_tensor,
        'target_length': lengths_tensor # for computing training speed
    }

class RelationDataIterator(object):
    """ """
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
        logger.info("Starting data epoch: %d" % self.epoches)

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


class RelationDataset(object):
    """
    """
    
    def __init__(self, data_file, text_vocab_file, target_label_file, params):
        """"""
        self.params = params
        self.data_file = data_file

        self.src_vocab = Vocabulary(text_vocab_file)
        self.tgt_label = {}
        for idx, line in enumerate(open(target_label_file, 'r')):
            key  = line.strip()
            assert key not in self.tgt_label
            self.tgt_label[key] = idx

        params.src_vocab = self.src_vocab
        params.src_vocab_size = len(params.src_vocab)
        params.tgt_label = self.tgt_label
        params.tgt_label_size = len(params.tgt_label)
        params.pad_index = params.src_vocab.pad_index
        params.eos_index = params.src_vocab.eos_index
        params.unk_index = params.src_vocab.unk_index

        self.examples = []

        with io.open(data_file, mode='r', encoding='utf-8') as data_inf:
            for line in data_inf:
                fields = line.strip().split('\t')
                assert len(fields) == 4
                sent, entity_idx, value_idx, label = fields
                entity_idx = int(entity_idx)
                value_idx = int(value_idx)
                self.examples.append(create_example(sent, entity_idx, value_idx, label, params))

            self.data_size = len(self.examples)
            logger.info("Loaded %d examples on memory" % self.data_size)

    def __len__(self):
        return self.data_size

    def __getitem__(self, key):
        return self.examples[key]
