from __future__ import absolute_import, division, print_function

from logging import getLogger
import itertools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = getLogger()


# def add_timing_signal(x):
#    batch, length, dim = x.size()
#    position_enc = np.array([
#        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
#        for pos in range(length)
#    ])
#    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
#    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
#    position_enc = torch.tensor(position_enc, device=x.device).float()
#    position_enc = position_enc.view((1, length, dim))
#    return x + position_enc

def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4, name=None):
    batch, length, channels = x.size()
    position = torch.arange(length, dtype=torch.float, device=x.device)
    num_timescales = channels // 2

    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float, device=x.device) * -log_timescale_increment)

    scaled_time = (torch.unsqueeze(position, 1) * torch.unsqueeze(inv_timescales, 0))
    signal = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), dim=1)
    signal = F.pad(signal, (0, np.mod(channels, 2), 0, 0))
    signal = signal.view((1, length, channels))
    return x + signal


def Embedding(num_embeddings, embedding_dim):
    m = nn.Embedding(num_embeddings, embedding_dim)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight, gain=1.0)
    return m

def smoothed_softmax_cross_entropy_with_logits(logits, labels, smoothing=0.0):
    if not smoothing:
        loss = F.cross_entropy(logits, labels, reduction='mean')
        return loss

    log_prb = F.log_softmax(logits, dim=1)
    return  F.nll_loss(log_prb, labels) * (1 - smoothing) - log_prb.mean() * smoothing

def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class PredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, params):
        super().__init__()
        self.n_words = params.tgt_vocab_size
        self.pad_index = params.pad_index
        dim = params.model_dim
        self.proj = Linear(dim, self.n_words, bias=False)
        self.label_smoothing = params.label_smoothing

    def forward(self, x, y, get_scores=False):
        """
        Compute the loss, and optionally the scores.
        """
        assert (y == self.pad_index).sum().item() == 0

        scores = self.proj(x).view(-1, self.n_words)
        loss = smoothed_softmax_cross_entropy_with_logits(scores, y, self.label_smoothing)
        # loss = F.cross_entropy(scores, y)

        if get_scores:
            return scores, loss
        else:
            return loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert x.dim() == 2
        return self.proj(x)


class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, num_heads, dim, dropout):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        assert self.dim % self.num_heads == 0

        self.q_lin = Linear(dim, dim)
        self.k_lin = Linear(dim, dim)
        self.v_lin = Linear(dim, dim)
        self.out_lin = Linear(dim, dim)

    def forward(self, input, mask, kv=None, cache=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        num_heads = self.num_heads
        dim_per_head = dim // num_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.num_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * dim_per_head)

        q = shape(self.q_lin(input))  # (bs, num_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, num_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, num_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, num_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, num_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)  # (bs, num_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # (bs, num_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(dim_per_head)  # (bs, num_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, num_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)  # (bs, num_heads, qlen, klen)
        scores.masked_fill_(mask, -float('inf'))  # (bs, num_heads, qlen, klen)

        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, num_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, num_heads, qlen, klen)
        context = torch.matmul(weights, v)  # (bs, num_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        return self.out_lin(context)


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, dropout, gelu_activation):
        super().__init__()
        self.dropout = dropout
        self.lin1 = Linear(in_dim, dim_hidden)
        self.lin2 = Linear(dim_hidden, out_dim)
        self.act = gelu if gelu_activation else F.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, params):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()

        self.n_words = params.src_vocab_size

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index

        # model parameters
        self.dim = params.model_dim  # 512 by default
        self.hidden_dim = params.hidden_dim  # 2048 by default
        self.num_heads = params.num_heads  # 8 by default
        assert self.dim % self.num_heads == 0, 'transformer dim must be a multiple of num_heads'

        self.n_layers = params.num_encoder_layers
        self.residual_dropout = params.residual_dropout
        self.attention_dropout = params.attention_dropout
        self.relu_dropout = params.relu_dropout
        self.max_sequence_size = params.max_sequence_size

        # embeddings
        self.embeddings = Embedding(self.n_words, self.dim)
        # don't know why, THUMT is doing this
        self.bias = nn.Parameter(torch.zeros(self.dim))

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.num_heads,
                                                      self.dim,
                                                      dropout=self.attention_dropout))
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-6))
            self.ffns.append(TransformerFFN(self.dim,
                                            self.hidden_dim,
                                            self.dim,
                                            dropout=self.relu_dropout,
                                            gelu_activation=params.gelu_activation))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-6))

    def forward(self, src_seq, src_len):
        """
        Inputs:
            `src_seq` LongTensor(bs, slen), containing word indices
            `src_len` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
        """
        bs, slen = src_seq.size()
        assert src_len.size(0) == bs
        assert src_len.max().item() <= slen

        # generate masks
        mask, attn_mask = get_masks(slen, src_len, causal=False)

        # positions
        positions = src_seq.new(slen).long()
        positions = torch.arange(slen, out=positions).unsqueeze(0)

        # embeddings
        tensor = self.embeddings(src_seq)
        tensor = tensor * (self.dim ** 0.5)
        tensor = tensor + self.bias
        tensor = add_timing_signal(tensor)
        tensor = F.dropout(tensor, p=self.residual_dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        for i in range(self.n_layers):
            # self attention
            attn = self.attentions[i](tensor, attn_mask)
            attn = F.dropout(attn, p=self.residual_dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # FFN
            ffn_out = self.ffns[i](tensor)
            ffn_out = F.dropout(ffn_out, p=self.residual_dropout, training=self.training)
            tensor = tensor + ffn_out
            tensor = self.layer_norm2[i](tensor)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        return tensor


class TransformerDecoder(nn.Module):

    def __init__(self, params):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()

        self.n_words = params.tgt_vocab_size

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index

        # model parameters
        self.dim = params.model_dim  # 512 by default
        self.hidden_dim = params.hidden_dim  # 2048 by default
        self.num_heads = params.num_heads  # 8 by default
        assert self.dim % self.num_heads == 0, 'transformer dim must be a multiple of num_heads'

        self.n_layers = params.num_decoder_layers
        self.residual_dropout = params.residual_dropout
        self.attention_dropout = params.attention_dropout
        self.relu_dropout = params.relu_dropout
        self.max_sequence_size = params.max_sequence_size

        # embeddings
        self.embeddings = Embedding(self.n_words, self.dim)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        self.layer_norm15 = nn.ModuleList()
        self.encoder_attn = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.num_heads, self.dim, dropout=self.attention_dropout))
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-6))
            self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-6))
            self.encoder_attn.append(MultiHeadAttention(self.num_heads, self.dim, dropout=self.attention_dropout))
            self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, dropout=self.relu_dropout,
                                            gelu_activation=params.gelu_activation))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-6))

    def forward(self, tgt_seq, tgt_len, src_enc=None, src_len=None, cache=None):
        """
        Inputs:
            `tgt_seq` LongTensor(bs, slen), containing word indices
            `tgt_len` LongTensor(bs), containing the length of each sentence
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
        """
        bs, slen = tgt_seq.size()
        assert tgt_len.size(0) == bs
        assert tgt_len.max().item() <= slen
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert src_enc.size(0) == bs

        # generate masks
        mask, attn_mask = get_masks(slen, tgt_len, causal=True)
        if src_enc is not None:
            src_mask = torch.arange(src_len.max(), dtype=torch.long, device=tgt_len.device) < src_len[:, None]

        # positions
        positions = tgt_seq.new(slen).long()
        positions = torch.arange(slen, out=positions).unsqueeze(0)

        # do not recompute cached elements
        if cache is not None:
            _slen = slen - cache['slen']
            tgt_seq = tgt_seq[:, -_slen:]
            positions = positions[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        targets = self.embeddings(tgt_seq)
        targets = targets * (self.dim ** 0.5)
        targets *= mask.unsqueeze(-1).to(targets.dtype)

        tensor = F.pad(targets, (0, 0, 1, 0))[:, :-1, :]
        tensor = add_timing_signal(tensor)
        tensor = F.dropout(tensor, p=self.residual_dropout, training=self.training)

        # transformer layers
        for i in range(self.n_layers):
            # self attention
            attn = self.attentions[i](tensor, attn_mask, cache=cache)
            attn = F.dropout(attn, p=self.residual_dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            if src_enc is not None:
                attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
                attn = F.dropout(attn, p=self.residual_dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)

            # FFN
            ffn_out = self.ffns[i](tensor)
            ffn_out = F.dropout(ffn_out, p=self.residual_dropout, training=self.training)
            tensor = tensor + ffn_out
            tensor = self.layer_norm2[i](tensor)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # update cache length
        if cache is not None:
            cache['slen'] += tensor.size(1)

        return tensor


class Transformer(nn.Module):
    """
    """

    def __init__(self, params):
        super(Transformer, self).__init__()
        self.params = params
        self.encoder = TransformerEncoder(params)
        self.decoder = TransformerDecoder(params)
        self.pred_layer = PredLayer(params)

        if params.share_source_target_embedding:
            self.encoder.embeddings.weight = self.decoder.embeddings.weight

        if params.share_embedding_and_softmax_weights:
            self.pred_layer.proj.weight = self.decoder.embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'train':
            return self.fwd(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(self, src_seq, src_len, tgt_seq, tgt_len):
        encoder_output = self.encoder(src_seq, src_len)
        decoder_output = self.decoder(tgt_seq, tgt_len, src_enc=encoder_output, src_len=src_len)

        pred_mask = torch.arange(tgt_len.max(), dtype=torch.long, device=tgt_len.device) < tgt_len[:, None]

        # masked_decoder_output = decoder_output[pred_mask.unsqueeze(-1).expand_as(decoder_output)].view(-1, self.params.model_dim)
        masked_decoder_output = decoder_output[pred_mask]
        masked_y = tgt_seq.masked_select(pred_mask)

        loss = self.pred_layer(x=masked_decoder_output, y=masked_y, get_scores=False)

        return loss
