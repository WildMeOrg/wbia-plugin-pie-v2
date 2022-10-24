import subprocess
import logging
import argparse

import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

from train import train

# from ray import tune
# from ray.tune.schedulers import ASHAScheduler
import optuna
from optuna.pruners import BasePruner
from optuna.trial._state import TrialState

from scipy import stats
from statistics import median
import math

from optuna import study
MAXIMIZE = study.StudyDirection.MAXIMIZE
MINIMIZE = study.StudyDirection.MINIMIZE




STATUS_COMPLETE = 'complete'
EPOCHS_PER_REPORTING_PERIOD = 10

def _train(trial, model, lr, transforms_train, cfg='configs/graywhale_coco.yaml'):
    args = {
        'opts': ['model.name', model, 'train.lr', lr, 'data.transforms_train', transforms_train],
        'cfg': cfg,
    }
    accuracy = train(args, optuna_trial=trial)
    return accuracy


class OverfitPlusMedianPruner(BasePruner):


    # note steps below are in epochs, and we report to optuna every 10 epochs
    def __init__(self, min_trials_for_median=3, median_warmup_steps=30, slope_steps=50, slope_tolerance=0.3):
        self.min_trials_for_median = min_trials_for_median
        self.median_warmup_steps = median_warmup_steps
        self.slope_steps = slope_steps
        self.slope_steps_in_index = math.floor(slope_steps / EPOCHS_PER_REPORTING_PERIOD)
        self.slope_tolerance = slope_tolerance

    def prune(self, study, trial):
        step = trial.last_step

        if not step: # trial.last_step == None when no scores have been reported yet
            return False

        # if objective slope is going in the wrong direction, prune
        if step >= self.slope_steps:
            # intermediate_values() is a dictionary of the form {step: value}
            step_nos = list(trial.intermediate_values.keys())
            step_nos.sort()
            step_nos = step_nos[-self.slope_steps_in_index:]
            objective_vals = [trial.intermediate_values[step] for step in step_nos]
            # linregress returns 5 things; we only want the first
            slope, _, _, _, _ = stats.linregress(step_nos, objective_vals)
            print(f'pruning check at step {step}: slope {slope}')
            if study.direction == MAXIMIZE and slope < -1 * self.slope_tolerance:
                print('Pruning this run because of negative slope.')
                return True
            if study.direction == MINIMIZE and slope > self.slope_tolerance:
                print('Pruning this run because of positive slope.')
                return True
        
        # now classic median pruning
        if trial.number >= self.min_trials_for_median and step >= self.median_warmup_steps:
            completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            other_scores = [
                t.intermediate_values[step]
                for t in completed_trials
                if step in t.intermediate_values
            ]
            if len(other_scores) < self.min_trials_for_median:
                print(f"skipping median computation because {len(other_scores)} trial{'s' if len(other_scores) != 1 else ''} have reported scores at this step (min_trials_for_median={self.min_trials_for_median})")
            else:
                # TODO: remove try/except if I haven't seen it.
                try:
                    median_score = median(other_scores)
                    if trial.intermediate_values[step] < median_score and study.direction == MAXIMIZE:
                        return True
                    if trial.intermediate_values[step] > median_score and study.direction == MINIMIZE:
                        return True
                except Exception as e:
                    print(f'error computing median: {e}')
                    from IPython import embed; embed()

        return False


def train_tune(arg=None, cfg='configs/graywhale_coco.yaml', n_trials=10):

    def objective(trial):
        model = trial.suggest_categorical('model', ['resnet50_fc512', 'resnext101_32x8d', 'efficientnet_b4'])
        lr = trial.suggest_float('lr', 0.00001, 0.1, log=True)
        transforms_train = trial.suggest_categorical('transforms_train', [
            ['resize', 'random_affine', 'color_jitter', 'random_grayscale'],
            ['resize', 'random_affine', 'color_jitter', 'random_grayscale', 'blur','center_crop'],
            ['resize', 'random_affine', 'blur'],
        ])

        accuracy = _train(trial, model, lr, transforms_train, cfg=cfg)
        return accuracy

    pruner = OverfitPlusMedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    return study







    # config = {

    # # }

    # search_space = {
    #     'lr': [0.0001, 0.001, 0.01, 0.1],
    #     'batch_size': tune.grid_search([32, 64, 128, 256]),
    # }
    # if 'cfg' in args:
    #     search_space['cfg'] = args['cfg']
    # analysis = tune.run(train, config=search_space)







# args = {
#     'cfg': 'configs/05_graywhale_resnet50.yaml',
# }
# train_tune(args)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     # parser.add_argument(
#     #     'opts',
#     #     default=None,
#     #     nargs=argparse.REMAINDER,
#     #     help='Modify config options using the command-line',
#     # )
#     args = parser.parse_args()

#     train_tune(args)
