import os
import math
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .utils import get_optimizer, to_cuda
from .utils import parse_lambda_config, update_lambdas
from torch.utils.tensorboard import SummaryWriter

logger = getLogger()
writer = SummaryWriter()


class Trainer(object):

    def __init__(self, data, params):
        """
        Initialize trainer.
        """

        # stopping criterion used for early stopping
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_total_iter = 0
        self.n_sentences = 0
        self.stats = OrderedDict([('processed_s', 0), ('processed_w', 0), ('loss', [])])
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params)

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # zero grad
        self.optimizer.zero_grad()

        # backward
        loss.backward()

        # clip gradients
        if self.params.clip_grad_norm > 0:
            clip_grad_norm_(self.model.parameters(), self.params.clip_grad_norm)

        # optimization step
        self.optimizer.step()

    def iter(self):
        """
        End of iteration.
        """
        self.n_total_iter += 1
        update_lambdas(self.params, self.n_total_iter)
        self.print_stats()
        # save periodic
        # do evaluation

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 10 != 0:
            return

        s_iter = "Batch %i - Step %i - " % (self.epoch, self.n_total_iter)
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # transformer learning rate
        lr = self.optimizer.param_groups[0]['lr']
        n_step = self.optimizer.param_groups[0]['num_updates']
        writer.add_scalar('learning_rate', lr, n_step)

        s_lr = " - LR = {:.4e}".format(lr)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
            self.stats['processed_s'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_s'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def save_model(self, name):
        """
        Save the model.
        """
        path = os.path.join(self.params.model_path, '%s.pth' % name)
        logger.info('Saving models to %s ...' % path)
        data = {}
        if self.params.multi_gpu:
            data['model'] = self.model.module.state_dict()
        else:
            data['model'] = self.model.state_dict()

        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)

    def save_checkpoint(self):
        """
        Checkpoint the experiment.
        """
        data = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
        }

        data['model'] = self.model.state_dict()
        data['model' + '_optimizer'] = self.optimizer.state_dict()

        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        checkpoint_path = os.path.join(self.params.model_path, 'checkpoint.pth')
        logger.info("Saving checkpoint to %s ..." % checkpoint_path)
        torch.save(data, checkpoint_path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.model_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        logger.warning('Reloading checkpoint from %s ...' % checkpoint_path)
        if self.params.device == 'cpu':
            data = torch.load(checkpoint_path, map_location=lambda storage, loc: 'cpu')
        else:
            data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(self.params.local_rank))

        # reload model parameters and optimizers
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['model' + '_optimizer'])

        # reload main metrics
        self.epoch = data['epoch'] + 1
        self.n_total_iter = data['n_total_iter']

        assert self.params.src_vocab == data['params'].src_vocab
        assert self.params.tgt_vocab == data['params'].tgt_vocab
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)

    def save_periodic(self):
        """
        Save the models periodically.
        """
        # if not self.params.is_master:
        #     return
        self.save_model('periodic-%i' % self.n_total_iter)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        # TODO
        # if not self.params.is_master:
        #     return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_model('best-%s' % metric)

    def end_evaluation(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if scores is not None and \
            self.stopping_criterion is not None:
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                if self.params.multi_gpu and 'SLURM_JOB_ID' in os.environ:
                    os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                exit()
        self.save_checkpoint()
        self.epoch += 1

        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                writer.add_histogram(name, parameter.data)


class EncDecTrainer(Trainer):

    def __init__(self, model, data, params):

        # model / data / params
        self.model = model
        self.data = iter(data)
        self.params = params

        # optimizers
        # self.optimizer = get_optimizer(self.model.parameters(), self.params.optimizer)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        super().__init__(data, params)

    def mt_step(self, lambda_coeff):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        self.model.train()

        batch = next(self.data)
        src_seq = batch['source']
        src_len = batch['source_length']
        tgt_seq = batch['target']
        tgt_len = batch['target_length']

        if params.device.type == 'cuda':
            src_seq, src_len, tgt_seq, tgt_len = to_cuda(src_seq, src_len, tgt_seq, tgt_len)

        # encode source sentence
        loss = self.model(src_seq=src_seq,
                          src_len=src_len,
                          tgt_seq=tgt_seq,
                          tgt_len=tgt_len,
                          mode='train')
        self.stats['loss'].append(loss.item())

        # Tensorboard
        writer.add_scalar('transformer/loss', loss.item(), self.n_total_iter)

        loss = lambda_coeff * loss
        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.stats['processed_s'] += tgt_len.size(0)
        self.stats['processed_w'] += tgt_len.sum().item()

