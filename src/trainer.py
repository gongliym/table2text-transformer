import os
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .optim import get_optimizer
from .utils import to_cuda, parse_lambda_config, update_lambdas

logger = getLogger()


class Trainer(object):

    def __init__(self, data, params):
        """
        Initialize trainer.
        """
        self.tensorboard_writer = params.tensorboard_writer
        
        if params.eval_metric is not None:
            if params.eval_metric[0] == '_':
                self.eval_metric = (params.eval_metric[1:], False)
            else:
                self.eval_metric = (params.eval_metric, True)

            metric_name, biggest = self.eval_metric
            self.best_eval_score = {metric_name: (-1e12 if biggest else 1e12)}
        else:
            self.eval_metric = None

        # training statistics
        self.n_total_iter = 0
        self.stats = OrderedDict([('processed_s', 0), ('processed_w', 0), ('loss', [])])
        self.last_time = time.time()
        self.decrease_count = 0

        # reload potential checkpoints
        self.reload_checkpoint()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params)


    def set_optimizer(self):
        """
        Set optimizer
        :return:
        """
        params = self.params
        # model optimizer (excluding memory values)
        self.optimizer = get_optimizer(self.model.parameters(), params.optimizer)

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

        s_iter = "Step %i - " % (self.n_total_iter)
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # transformer learning rate
        lr = self.optimizer.param_groups[0]['lr']
        #n_step = self.optimizer.param_groups[0]['num_updates']
        self.tensorboard_writer.add_scalar('Training/lr', lr, self.n_total_iter)

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
        data['model'] = self.model.state_dict()
        #if self.params.multi_gpu:
        #    data['model'] = self.model.module.state_dict()
        #else:
        #    data['model'] = self.model.state_dict()

        data['params'] = {k: v for k, v in self.params.__dict__.items() if k!="tensorboard_writer"}

        torch.save(data, path)

    def save_checkpoint(self):
        """
        Checkpoint the experiment.
        """
        data = {
            'n_total_iter': self.n_total_iter,
            'best_eval_score': self.best_eval_score,
            'decrease_count': self.decrease_count,
        }

        data['model'] = self.model.state_dict()
        data['model' + '_optimizer'] = self.optimizer.state_dict()

        data['params'] = {k: v for k, v in self.params.__dict__.items() if k!="tensorboard_writer"}

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
        self.n_total_iter = data['n_total_iter']
        self.best_eval_score = data['best_eval_score']
        self.decrease_count = data['decrease_count']

        assert self.params.src_vocab == data['params'].src_vocab
        assert self.params.tgt_vocab == data['params'].tgt_vocab
        logger.warning('Checkpoint reloaded. Resuming at step %i ...' % self.n_total_iter)

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
        metric, biggest = self.eval_metric
        if metric not in scores:
            logger.warning("Metric \"%s\" not found in scores!" % metric)
            return

        factor = 1 if biggest else -1
        if factor * scores[metric] > factor * self.best_eval_score[metric]:
            self.best_eval_score[metric] = scores[metric]
            logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
            self.save_model('best-%s' % metric)

    def end_evaluation(self, scores):
        """
        End the evaluation.
        """
        self.save_checkpoint()
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.tensorboard_writer.add_histogram(name, parameter.data)

        # stop if the stopping criterion has not improved after a certain number of epochs
        metric, biggest = self.eval_metric
        if metric not in scores:
            logger.warning("Metric \"%s\" not found in scores!" % metric)
            return

        factor = 1 if biggest else -1
        if factor * scores[metric] > factor * self.best_eval_score[metric]:
            self.best_eval_score[metric] = scores[metric]
            self.decrease_count = 0
            logger.info('New best validation score for %s: %.6f' % (metric, scores[metric]))
            self.save_model('best-%s' % metric)
        else:
            self.decrease_count += 1
            logger.info("Not a better validation score (%i / %i)."
                        % (self.decrease_count, self.stopping_criterion))

        if self.decrease_count >= self.stopping_criterion:
            logger.info("Stopping criterion has been below its best value for more "
                        "than %i epochs. Ending the experiment..." % self.stopping_criterion)
            exit()


class EncDecTrainer(Trainer):

    def __init__(self, model, data, params):

        # model / data / params
        self.model = model
        self.data = iter(data)
        self.params = params

        # optimizers
        self.set_optimizer()
        # self.optimizer = get_optimizer(self.model.parameters(), self.params.optimizer)

        super().__init__(data, params)

    def mt_step(self):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """
        params = self.params
        self.model.train()

        batch = next(self.data)

        if params.use_cuda:
            for each in batch:
                batch[each] = batch[each].cuda()

        # encode source sentence
        loss = self.model(batch, mode='train')
        loss = loss.mean()
        self.stats['loss'].append(loss.item())

        # Tensorboard
        self.tensorboard_writer.add_scalar('Training/loss', loss.item(), self.n_total_iter)

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.stats['processed_s'] += batch['target_length'].size(0)
        self.stats['processed_w'] += batch['target_length'].sum().item()

    def sm_step(self):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """
        params = self.params
        self.model.train()

        batch = next(self.data)

        if params.use_cuda:
            for each in batch:
                batch[each] = batch[each].cuda()

        # encode source sentence
        loss = self.model(batch, mode='train', step=self.n_total_iter)
        loss = loss.mean()
        self.stats['loss'].append(loss.item())

        # Tensorboard
        self.tensorboard_writer.add_scalar('Training/loss', loss.item(), self.n_total_iter)

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.stats['processed_s'] += batch['summary_length'].size(0)
        self.stats['processed_w'] += batch['summary_length'].sum().item()

    def step(self):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """
        params = self.params
        self.model.train()

        batch = next(self.data)

        if params.use_cuda:
            for each in batch:
                batch[each] = batch[each].cuda()

        # encode source sentence
        loss = self.model(batch, mode='train', step=self.n_total_iter)
        loss = loss.mean()
        self.stats['loss'].append(loss.item())

        # Tensorboard
        self.tensorboard_writer.add_scalar('Training/loss', loss.item(), self.n_total_iter)

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.stats['processed_s'] += batch['target_length'].size(0)
        self.stats['processed_w'] += batch['target_length'].sum().item()

