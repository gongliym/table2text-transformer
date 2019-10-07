from __future__ import absolute_import, division, print_function

from logging import getLogger
import itertools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = getLogger()

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

def smoothed_softmax_cross_entropy_with_logits(logits, labels, smoothing=0.0):
    if not smoothing:
        loss = F.cross_entropy(logits, labels, reduction='mean')
        return loss

    log_prb = F.log_softmax(logits, dim=1)

    _, vocab_size = log_prb.size()

    n = vocab_size - 1.0
    p = 1.0 - smoothing
    q = smoothing / n

    # Normalizing constant is the best cross-entropy value with soft
    # targets. We subtract it just for readability, makes no difference on
    # learning
    normalizing = -(p * np.log(p) + n * q * np.log(q + 1e-20))
    xentropy = F.nll_loss(log_prb, labels) * (1 - smoothing) - log_prb.mean() * smoothing
    return xentropy - normalizing

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


def Embedding(num_embeddings, embedding_dim):
    m = nn.Embedding(num_embeddings, embedding_dim)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight, gain=1.0)
    return m


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
