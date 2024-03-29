from __future__ import absolute_import, division, print_function

from logging import getLogger

from .translation_dataset import TranslationDataset, TranslationDataIterator
from .table2text_dataset import Table2TextDataset, Table2TextDataIterator
from .relation_dataset import RelationDataset, RelationDataIterator

logger = getLogger()

def load_data(input_files, params, train=False, repeat=False, model="nmt"):
    """
    Load parallel data.
    """
    if model == "nmt":
        assert 2 == len(params.train_files)
        train_dataset = TranslationDataset(params.train_files[0], params.train_files[1],
                                           params.vocab_files[0], params.vocab_files[1],
                                           params=params)

        train_data_iter = TranslationDataIterator(train_dataset, params=params, train=train, repeat=repeat)
    elif model == "nlg":
        assert 3 == len(params.train_files)
        train_dataset = Table2TextDataset(params.train_files[0], params.train_files[1], params.train_files[2],
                                           params.vocab_files[0], params.vocab_files[1],
                                           params=params)

        train_data_iter = Table2TextDataIterator(train_dataset, params=params, train=train, repeat=repeat)
    elif model == "ie":
        assert 1 == len(params.train_files)
        train_dataset = RelationDataset(params.train_files[0], params.vocab_files[0], params.vocab_files[1],
                                           params=params)

        train_data_iter = RelationDataIterator(train_dataset, params=params, train=train, repeat=repeat)
    else:
        raise Exception("Unkown model name. %s" % model)

    return train_data_iter

