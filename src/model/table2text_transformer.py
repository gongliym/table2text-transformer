from __future__ import absolute_import, division, print_function

from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerDecoder
from .layers import Embedding, Linear, MultiHeadAttention, TransformerFFN
from .layers import add_timing_signal, smoothed_softmax_cross_entropy_with_logits
from .layers import get_masks

logger = getLogger()


class CSPredLayer(nn.Module):
    """
    binary classification layer
    """
    def __init__(self, params):
        super().__init__()
        self.pad_index = params.pad_index
        dim = params.model_dim
        self.proj = Linear(dim, 1, bias=True)
        self.proj_act = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, x, y, get_scores=False):
        """
        Compute the loss, and optionally the scores.
        """
        scores = self.proj(x)
        scores = self.proj_act(scores)
        y = y.view_as(scores).float()
        loss = self.criterion(scores, y)
        if get_scores:
            return scores, loss
        else:
            return loss


class SMPredLayer(nn.Module):
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

        if get_scores:
            return scores, loss
        else:
            return loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert x.dim() == 2
        return F.log_softmax(self.proj(x), dim=1)


class TableEncoder(nn.Module):

    def __init__(self, params):
        """
        Table2Text Transformer model (encoder).
        """
        super().__init__()

        self.n_words = params.src_vocab_size

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index

        # model parameters
        self.dim = params.model_dim  # 512 by default
        self.emb_dim = params.model_dim // 4  # 512 by default
        self.hidden_dim = params.hidden_dim  # 2048 by default
        self.num_heads = params.num_heads  # 8 by default
        assert self.dim % self.num_heads == 0, 'transformer dim must be a multiple of num_heads'

        self.n_layers = params.num_encoder_layers
        self.residual_dropout = params.residual_dropout
        self.attention_dropout = params.attention_dropout
        self.relu_dropout = params.relu_dropout
        self.max_sequence_size = params.max_sequence_size

        # embeddings
        self.embeddings = Embedding(self.n_words, self.emb_dim)
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
        
    def forward(self, x1, x2, x3, x4, lengths):
        """
        Inputs:
            `src_seq` LongTensor(bs, slen), containing word indices
        """
        bs, slen = x1.size()
        assert x1.size() == x2.size() == x3.size() == x4.size()

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal=False)

        # embeddings
        x1 = self.embeddings(x1)
        x2 = self.embeddings(x2)
        x3 = self.embeddings(x3)
        x4 = self.embeddings(x4)
        tensor = torch.cat((x1, x2, x3, x4), dim=-1)

        tensor = tensor * (self.dim ** 0.5)
        tensor = tensor + self.bias
        tensor = add_timing_signal(tensor)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        tensor = F.dropout(tensor, p=self.residual_dropout, training=self.training)

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


class Data2TextTransformer(nn.Module):
    """
    """

    def __init__(self, params):
        super(Data2TextTransformer, self).__init__()
        self.params = params
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index

        self.tgt_n_words = params.tgt_vocab_size
        self.src_n_words = params.src_vocab_size

        # model parameters
        self.dim = params.model_dim  # 512 by default
        self.hidden_dim = params.hidden_dim  # 2048 by default
        self.num_heads = params.num_heads  # 8 by default
        assert self.dim % self.num_heads == 0, 'transformer dim must be a multiple of num_heads'

        self.encoder = TableEncoder(params)
        self.decoder = TransformerDecoder(params)
        self.cs_pred_layer = CSPredLayer(params)
        self.sm_pred_layer = SMPredLayer(params)

        self.lambda_cs = min(1.0, max(0, params.lambda_cs))

        if params.share_embedding_and_softmax_weights:
            self.sm_pred_layer.proj.weight = self.decoder.embeddings.weight

        if params.share_source_target_embedding:
            assert self.src_n_words == self.tgt_n_words
            self.encoder.embeddings.weight = self.decoder.embeddings.weight

    def forward(self, features, mode='train'):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        x1 = features['table_entity']
        x2 = features['table_type']
        x3 = features['table_value']
        x4 = features['table_feature']
        cs_label = features['table_label']
        src_len = features['table_length']
        if mode == 'train' or mode == 'valid':
            assert features['summary'] is not None and features['summary_length'] is not None
            tgt_seq = features['summary']
            tgt_len = features['summary_length']
            encoder_output = self.encoder(x1, x2, x3, x4, src_len)
            cs_pred_mask = torch.arange(src_len.max(), dtype=torch.long, device=src_len.device) < src_len[:, None]
            cs_label = cs_label[cs_pred_mask]
            masked_encoder_output = encoder_output[cs_pred_mask]
            cs_loss = self.cs_pred_layer(masked_encoder_output, cs_label)

            decoder_output = self.decoder(tgt_seq, tgt_len, src_enc=encoder_output, src_len=src_len)
            sm_pred_mask = torch.arange(tgt_len.max(), dtype=torch.long, device=tgt_len.device) < tgt_len[:, None]
            masked_decoder_output = decoder_output[sm_pred_mask]
            masked_y = tgt_seq.masked_select(sm_pred_mask)
            sm_loss = self.sm_pred_layer(x=masked_decoder_output, y=masked_y, get_scores=False)

            self.params.tensorboard_writer.add_scalar('Training_NLG/cs_loss', cs_loss)
            self.params.tensorboard_writer.add_scalar('Training_NLG/sm_loss', sm_loss)
            return self.lambda_cs * cs_loss + (1-self.lambda_cs) * sm_loss
        elif mode == 'test':
            encoder_output = self.encoder(x1, x2, x3, x4, src_len)
            max_len = max(src_len) + 50
            if self.params.beam_size > 1:
                output, out_len = self.generate_beam(encoder_output,
                                                     src_len,
                                                     self.params.beam_size,
                                                     self.params.length_penalty,
                                                     self.params.early_stopping,
                                                     max_len=max_len)
            else:
                output, out_len = self.generate(encoder_output, src_len, max_len=max_len)

            return output, out_len
        else:
            raise Exception("Unknown mode: %s" % mode)

    def generate(self, enc_out, src_len, max_len=200, sample_temperature=None):
        """
        Decode a sentence given initial start (without beam search)
        :param enc_out: Encoder output (bs, seq_len, dim)
        :param src_len: (bs, )
        :param max_len: int
        :param sample_temperature:
        :return:
        """
        bs = len(src_len)
        assert enc_out.size(0) == bs

        #generated sentences
        generated = src_len.new(bs, max_len)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[:,0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)

        # cache compute states
        cache = {'slen': 0}

        while cur_len < max_len:
            # compute word socres
            tensor = self.decoder(tgt_seq=F.pad(generated[:, 1:cur_len], (0, 1)),
                                  tgt_len=gen_len,
                                  src_enc=enc_out,
                                  src_len=src_len,
                                  cache=cache
                                  )
            assert tensor.size() == (bs, 1, self.dim)
            tensor = tensor.data[:, -1, :] # (bs, dim)
            scores = self.sm_pred_layer.get_scores(tensor) # (bs, vocab_size)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(F.softmax(scores / sample_temperature, dim=1), 1).squeeze(1)
            assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            generated[:, cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[:, -1].masked_fill_(unfinished_sents.byte(), self.eos_index)

        # sanity check
        assert (generated == self.eos_index).sum() == 2 * bs

        return generated[:cur_len], gen_len

    def generate_beam(self, enc_out, src_len, beam_size, length_penalty, early_stopping, max_len=200):
        """
        Decode a sentence by beam search
        :param enc_out: Encoder output (bs, seq_len, dim)
        :param src_len: (bs, )
        :param beam_size:
        :param length_penalty:
        :param early_stopping:
        :param max_len:
        :return:
        """
        # check inputs
        assert enc_out.size(0) == src_len.size(0)
        assert beam_size > 1

        # batch size / number of words
        bs = len(src_len)
        n_words = self.tgt_n_words

        # expand to beam search the source latent representation / source lengths
        enc_out = enc_out.unsqueeze(1).expand((bs, beam_size) + enc_out.shape[1:]).contiguous().view((bs * beam_size,) + enc_out.shape[1:]) # (bs*beam_size, seq_len, dim)
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1) # (bs*beam_size, )

        # generated sentences
        generated = src_len.new(bs*beam_size, max_len) # upcoming output
        generated.fill_(self.pad_index) # fill upcoming output with <pad>
        generated[:,0].fill_(self.eos_index) # we use <eos> for <bos> everywhere

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]

        # scores for each sentence in the beam
        beam_scores = enc_out.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1) # (bs*beam_size, )

        #current position
        cur_len = 1
        cache = {'slen': 0}
        done = [False for _ in range(bs)]

        while cur_len < max_len:
            tensor = self.decoder(tgt_seq=F.pad(generated[:, 1:cur_len], (0, 1)),
                                  tgt_len=src_len.new(bs*beam_size).fill_(cur_len),
                                  src_enc=enc_out,
                                  src_len=src_len,
                                  cache=cache
                                  )
            assert tensor.size() == (bs*beam_size, 1, self.dim)
            tensor = tensor.data[:, -1, :] # (bs * beam_size, dim)
            scores = self.sm_pred_layer.get_scores(tensor) # (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores) # (bs*beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words) # (bs, beam_size*n_words)

            next_scores, next_words = torch.topk(_scores, 2*beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2*beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, beam index in the batch)
            next_batch_beam = []

            for sent_id in range(bs):
                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size) # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[sent_id*beam_size+beam_id, :cur_len].clone(), value.item())
                        # self.print_hyp(generated[sent_id*beam_size+beam_id, :cur_len], value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id*beam_size+beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_index, 0)] * beam_size # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[beam_idx, :]
            generated[:, cur_len] = beam_words
            for k in cache.keys():
                if k != 'slen':
                    cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_score, best_hyp = max(hypotheses.hyp, key=lambda x: x[0])
            tgt_len[i] = len(best_hyp) + 1 # +1 for the <eos> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(bs, tgt_len.max().item()).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[i, :tgt_len[i]-1] = hypo
            decoded[i, tgt_len[i]-1] = self.eos_index

        #sanity check
        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len

    def print_hyp(self, generated, score):
        print(score, ' '.join([self.params.tgt_vocab.itos(x.item()) for x in generated]))


class BeamHypotheses(object):

    def __init__(self, n_hype, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses
        :param n_hype:
        :param max_len:
        :param length_penalty:
        :param early_stopping:
        """
        self.max_len = max_len - 1 # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hype
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list
        :return:
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis in the list
        :param hyp:
        :param sum_logprobs:
        :return:
        """
        length_penalty = ((5.0 + len(hyp)) / 6.0) ** self.length_penalty
        score = sum_logprobs / length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that non of the hypotheses being generated can
        become better than the worst one in the heap, then we are done with this sentence
        :param best_sum_logprobs:
        :return:
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            length_penalty = ((5.0 + self.max_len) / 6.0) ** self.length_penalty
            return self.worst_score >= best_sum_logprobs / length_penalty

