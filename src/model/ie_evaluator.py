from __future__ import absolute_import, division, print_function

from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerDecoder
from .layers import Embedding, Linear, TransformerFFN
from .layers import add_timing_signal, smoothed_softmax_cross_entropy_with_logits
from .layers import get_masks

logger = getLogger()


class IEEvaluator(nn.Module):
    """
    Information Extraction Evaluator
    """

    def __init__(self, params):
        super(IEEvaluator, self).__init__()
        self.params = params
        self.max_len = params.max_sequence_size
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index

        self.n_words = params.src_vocab_size
        self.n_labels = params.tgt_label_size

        self.emb_dim = params.model_dim
        self.pos_emb_dim = params.model_dim // 2

        self.hidden_size = 500

        self.token_embedding = Embedding(self.n_words, self.emb_dim)
        self.ent_pos_embedding = Embedding(2*self.max_len, self.pos_emb_dim)
        self.val_pos_embedding = Embedding(2*self.max_len, self.pos_emb_dim)
        self.blstm = nn.LSTM(2*self.emb_dim, self.hidden_size, 
                             dropout=params.relu_dropout, 
                             bidirectional=True) # bs, slen, 1000)
        self.ffn = TransformerFFN(2*self.hidden_size, 700, self.n_labels, params.relu_dropout, params.gelu_activation)

    def forward(self, features, mode='train', step=-1):
        """
        """
        input_sentence = features['sentence']
        input_lengths = features['sentence_length']
        entity_indexes = features['entity_index']
        value_indexes  = features['value_index']
        bs, slen = input_sentence.size()
        word_embedding = self.token_embedding(input_sentence)

        positions = torch.arange(slen, dtype=torch.long, device=input_lengths.device).unsqueeze(0).expand_as(input_sentence)
        ent_relative_positions = positions - entity_indexes.unsqueeze(-1) + self.max_len
        val_relative_positions = positions - value_indexes.unsqueeze(-1) + self.max_len

        ent_position_embedding = self.ent_pos_embedding(ent_relative_positions)
        val_position_embedding = self.val_pos_embedding(val_relative_positions)

        x = torch.cat((word_embedding, ent_position_embedding, val_position_embedding), -1)
        x, (hn, cn) = self.blstm(x)
        x, indices = torch.max(x, dim=1)
        scores = self.ffn(x)

        if mode == 'train' or mode == 'valid':
            assert 'label' in features
            targets = features['label']
            loss = F.cross_entropy(scores, targets)
            return loss
        elif mode == 'test':
            _, predicted = torch.max(scores, dim=1)
            return predicted






            
        
       
        
