from logging import getLogger
import os
import subprocess
from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F

from src.utils import to_cuda, restore_segmentation, concat_batches
from src.data.dataset import load_and_batch_input_data


BLEU_SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH)

test_list = ['valid']

logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer, params):
        """
        Initialize evaluator
        :param trainer:
        :param params:
        """
        self.trainer = trainer
        self.params = params

        params.hyp_path = os.path.join(params.model_path, 'hypotheses')
        subprocess.Popen('mkdir -p %s' % params.hyp_path, shell=True).wait()

    def run_all_evals(self, trainer):
        """
        Run all evaluation
        :param trainer:
        :return:
        """
        scores = OrderedDict({'iter': trainer.n_total_iter})
        with torch.no_grad():
            self.evaluate_nmt(scores)
        return scores

class TransformerEvaluator(Evaluator):

    def __init__(self, trainer, params):
        """
        Build encoder / decoder evaluator
        :param trainer:
        :param params:
        """
        self.model = trainer.model
        self.src_vocab = params.src_vocab
        self.tgt_vocab = params.tgt_vocab

        super().__init__(trainer, params)

    def evaluate_nmt(self, scores):
        """
        Evaluate perplexity and next word prediction accuracy
        :param scores:
        :param data_set:
        :param eval_bleu:
        :return:
        """

        params = self.params
        step_num = self.trainer.n_total_iter

        hypotheses = []
        for batch in load_and_batch_input_data(params.valid_files[0], params):
            output, out_len = self.model(mode='test',
                                         src_seq=batch['source'],
                                         src_len=batch['source_length'])

            for j in range(output.size(0)):
                sent = output[j,:]
                delimiters = (sent == params.eos_index).nonzero().view(-1)
                assert len(delimiters) >= 1 and delimiters[0].item() == 0
                sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]
                target = ' '.join([params.tgt_vocab.itos(sent[idx].item()) for idx in range(len(sent))])
                hypotheses.append(target)

        # hypothesis / reference paths
        hyp_name = 'eval{0}.valid.txt'.format(step_num)
        hyp_path = os.path.join(params.hyp_path, hyp_name)
        ref_path = params.valid_files[1]

        # export sentences to hypothesis file / restore BPE segmentation
        with open(hyp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(hypotheses) + '\n')
        restore_segmentation(hyp_path)

        # evaluate BLEU score
        bleu_info = eval_moses_bleu(ref_path, hyp_path)
        bleu_score = float(bleu_info[7:bleu_info.index(',')])
        logger.info("BLEU %s %s : %s" % (hyp_path, ref_path, bleu_info))
        scores['nmt_bleu'] = bleu_score

def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    :param ref:
    :param hyp:
    :return:
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref+'0')
    assert os.path.isfile(BLEU_SCRIPT_PATH)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode('utf-8')
    return result
