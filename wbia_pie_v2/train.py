# -*- coding: utf-8 -*-
import sys
import time
import os
import os.path as osp
import argparse
import torch
import torch.nn as nn

from utils import (
    Logger,
    check_isfile,
    set_random_seed,
    collect_env_info,
    resume_from_checkpoint,
    load_pretrained_weights,
    compute_model_complexity,
)

from default_config import (
    imagedata_kwargs,
    optimizer_kwargs,
    engine_run_kwargs,
    get_default_config,
    lr_scheduler_kwargs,
)

import optim
from engine import TripletPIEEngine, CirclePIEEngine
from models import build_model
from datasets.datamanager import AnimalImageDataManager


def train(args, optuna_trial=None):

    # if passed in via argparser
    if type(args) == argparse.Namespace:
        args = vars(args)

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()

    print(f'top of train with args {args}')

    if 'cfg' in args:
        cfg.merge_from_file(args['cfg'])
    if 'opts' in args:
        print(f'PIE training found opt: {args["opts"]}')
        cfg.merge_from_list(args['opts'])


    set_random_seed(cfg.train.seed)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S')
    log_name += timestamp
    config_name = '_'.join([osp.splitext(osp.basename(args['cfg']))[0], cfg.data.version])
    save_dir = osp.join(cfg.data.save_dir, config_name)
    tb_dir = osp.join(cfg.data.tb_dir, '_'.join([config_name, timestamp]))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    sys.stdout = Logger(osp.join(save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    print('Making datamanger:')
    datamanager_args = imagedata_kwargs(cfg)
    datamanager_args['config_fpath'] = args['cfg']
    datamanager = AnimalImageDataManager(**datamanager_args)

    print('Building model: {}'.format(cfg.model.name))
    model = build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu,
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print('Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type))
    if cfg.loss.name == 'circle':
        print('CIRCLE LOSS HERE WE GO BABY')
        engine = CirclePIEEngine(
            datamanager,
            model,
            optimizer=optimizer,
            margin=cfg.loss.circle.margin,
            gamma=cfg.loss.circle.gamma,
            weight_t=cfg.loss.triplet.weight_t,
            weight_x=cfg.loss.triplet.weight_x,
            scheduler=scheduler,
            use_gpu=cfg.use_gpu,
            label_smooth=cfg.loss.softmax.label_smooth,
        )
    elif cfg.loss.name == 'triplet':
        engine = TripletPIEEngine(
            datamanager,
            model,
            optimizer=optimizer,
            margin=cfg.loss.triplet.margin,
            weight_t=cfg.loss.triplet.weight_t,
            weight_x=cfg.loss.triplet.weight_x,
            scheduler=scheduler,
            use_gpu=cfg.use_gpu,
            label_smooth=cfg.loss.softmax.label_smooth,
        )

    best_rank1 = engine.run(**engine_run_kwargs(cfg), save_dir=save_dir, tb_dir=tb_dir, optuna_trial=optuna_trial)
    return best_rank1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--cfg', type=str, default='', help='path to config file')

    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line',
    )
    args = parser.parse_args()

    train(args)
