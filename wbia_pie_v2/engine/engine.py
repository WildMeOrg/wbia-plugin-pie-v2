# -*- coding: utf-8 -*-
"""
Adapted from: https://github.com/KaiyangZhou/deep-person-reid
"""
from __future__ import division, print_function, absolute_import
import time
import os.path as osp
import datetime
from collections import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
# from ray import tune
import optuna

from utils import (
    MetricMeter,
    AverageMeter,
    open_all_layers,
    save_checkpoint,
    open_specified_layers,
    visualize_batch,
)
from losses import DeepSupervision


class Engine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of datamanager.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, datamanager, use_gpu=True):
        self.datamanager = datamanager
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.use_gpu = torch.cuda.is_available() and use_gpu
        self.writer = None
        self.epoch = 0

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()

    def register_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('_models') is None:
            raise AttributeError('Cannot assign model before super().__init__() call')

        if self.__dict__.get('_optims') is None:
            raise AttributeError('Cannot assign optim before super().__init__() call')

        if self.__dict__.get('_scheds') is None:
            raise AttributeError('Cannot assign sched before super().__init__() call')

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, rank1, save_dir, is_best=False):
        names = self.get_model_names()

        for name in names:
            save_checkpoint(
                {
                    'state_dict': self._models[name].state_dict(),
                    'epoch': epoch + 1,
                    'rank1': rank1,
                    'optimizer': self._optims[name].state_dict(),
                    'scheduler': self._scheds[name].state_dict(),
                },
                osp.join(save_dir, name),
                is_best=is_best,
            )

    def set_model_mode(self, mode='train', names=None):
        assert mode in ['train', 'eval', 'test']
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self._models[name].train()
            else:
                self._models[name].eval()

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[-1]['lr']

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def run(
        self,
        save_dir='log',
        tb_dir='tb_log',
        max_epoch=0,
        start_epoch=0,
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None,
        start_eval=0,
        eval_freq=-1,
        test_only=False,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        ranks=[1, 5, 10, 20],
        rerank=False,
        vis_train_data=True,
        visrank_resize=True,
        optuna_trial=None,
        # report_to_tune=False,
    ):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
        """

        if visrank and not test_only:
            raise ValueError('visrank can be set to True only if test_only=True')

        if test_only:
            self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                ranks=ranks,
                rerank=rerank,
                visrank_resize=visrank_resize,
            )
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=tb_dir)

        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.save_dir = save_dir
        self.vis_train_data = vis_train_data

        print('=> Start training')
        is_best = False
        best_rank1 = 0.0

        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.train(
                print_freq=print_freq,
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers,
            )

            if (
                (self.epoch + 1) >= start_eval
                and eval_freq > 0
                and (self.epoch + 1) % eval_freq == 0
                and (self.epoch + 1) != self.max_epoch
            ):
                rank1 = self.test(
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    ranks=ranks,
                )
                if rank1 > best_rank1:
                    best_rank1 = rank1
                    is_best = True
                else:
                    is_best = False
                self.save_model(self.epoch, rank1, save_dir, is_best=is_best)
                if optuna_trial != None:
                    optuna_trial.report(rank1, self.epoch)
                    if optuna_trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                # if report_to_tune:
                #     tune.report(rank1)

        if self.max_epoch > 0:
            print('=> Final test')
            rank1 = self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                ranks=ranks,
            )
            self.save_model(self.epoch, rank1, save_dir, is_best=is_best)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is not None:
            self.writer.close()

        # for hyperparam optimization
        return best_rank1

    def train(self, print_freq=10, fixbase_epoch=0, open_layers=None):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.set_model_mode('train')

        self.two_stepped_transfer_learning(self.epoch, fixbase_epoch, open_layers)

        self.num_batches = len(self.train_loader) * self.datamanager.num_copies
        end = time.time()
        for self.batch_idx, data in enumerate(self.train_loader):
            if self.vis_train_data and self.epoch == 0:
                # Visualise training batch at first epoch as a sanity check
                visualize_batch(
                    batch=data['img'],
                    labels=data['pid'],
                    save_dir=self.save_dir,
                    figname='train_batch',
                )
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(data)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % print_freq == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (self.max_epoch - (self.epoch + 1)) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch + nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch: [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr:.6f}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta_str,
                        losses=losses,
                        lr=self.get_current_lr(),
                    )
                )

            if self.writer is not None:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/data', data_time.avg, n_iter)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar('Train/' + name, meter.avg, n_iter)
                self.writer.add_scalar('Train/lr', self.get_current_lr(), n_iter)

            end = time.time()

        self.update_lr()

    def forward_backward(self, data):
        raise NotImplementedError

    def compute_loss(self, criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss

    def extract_features(self, input):
        return self.model(input)

    def parse_data_for_train(self, data):
        imgs = data['img']
        pids = data['pid']
        return imgs, pids

    def parse_data_for_eval(self, data):
        imgs = data['img']
        pids = data['pid']
        return imgs, pids

    def two_stepped_transfer_learning(
        self, epoch, fixbase_epoch, open_layers, model=None
    ):
        """Two-stepped transfer learning.

        The idea is to freeze base layers for a certain number of epochs
        and then open all layers for training.

        Reference: https://arxiv.org/abs/1611.05244
        """
        model = self.model if model is None else model
        if model is None:
            return

        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                '* Only train {} (epoch: {}/{})'.format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            open_specified_layers(model, open_layers)
        else:
            open_all_layers(model)
