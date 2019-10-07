from __future__ import absolute_import, division, print_function

from logging import getLogger

from .dataset import TranslationDataset, DataIterator
from .table2text_dataset import Table2TextDataset, Table2TextDataIterator

logger = getLogger()

def load_data(input_files, params, train=False, repeat=False):
    """
    Load parallel data.
    """

    train_dataset = TranslationDataset(params.train_files[0], params.train_files[1],
                                       params.vocab_files[0], params.vocab_files[1],
                                       params=params)

    train_data_iter = DataIterator(train_dataset, params=params, train=train, repeat=repeat)

    return train_data_iter

def load_data2text_data(input_files, params, train=False, repeat=False):
    """
    Load parallel data.
    """

    train_dataset = Table2TextDataset(params.train_files[0], params.train_files[1], params.train_files[2],
                                       params.vocab_files[0], params.vocab_files[1],
                                       params=params)

    train_data_iter = Table2TextDataIterator(train_dataset, params=params, train=train, repeat=repeat)

    return train_data_iter

