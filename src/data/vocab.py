from __future__ import absolute_import, division, print_function

import io
from logging import getLogger
from collections import defaultdict

logger = getLogger()

PAD_TOKEN = '<pad>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'


class Vocabulary(object):
    def __init__(self, filename):
        self.str2int = defaultdict(self._default_unk_index)
        self.int2str = []
        self.specials = [PAD_TOKEN, EOS_TOKEN, UNK_TOKEN]

        with io.open(filename, mode='r', encoding='utf-8') as vocab_file:
            for line in vocab_file:
                word = line.strip()
                if word in self.str2int:
                    raise Exception("Double occurrances in vocab: %s" % word)
                    exit()
                self.str2int[word] = len(self.str2int)
                self.int2str.append(word)

        assert PAD_TOKEN in self.str2int
        self.pad_index = self.str2int[PAD_TOKEN]
        assert EOS_TOKEN in self.str2int
        self.eos_index = self.str2int[EOS_TOKEN]
        assert UNK_TOKEN in self.str2int
        self.unk_index = self.str2int[UNK_TOKEN]

        self.vocab_size = len(self.int2str)
        logger.info("Vocab size: %d (%s)" % (self.vocab_size, filename))

    def stoi(self, token):
        return self.str2int.get(token, self.unk_idx)

    def itos(self, idx):
        if idx >= self.vocab_size:
            raise ValueError("Word index out of range: %d" % idx)
        return self.int2str[idx]

    def __len__(self):
        return self.vocab_size

    def _default_unk_index(self):
        return self.unk_index

    def __getitem__(self, token):
        return self.str2int.get(token, self.unk_index)

    def __eq__(self, other):
        if self.vocab_size != other.vocab_size:
            return False
        if self.str2int != other.str2int:
            return False
        if self.int2str != other.int2str:
            return False
        return True

