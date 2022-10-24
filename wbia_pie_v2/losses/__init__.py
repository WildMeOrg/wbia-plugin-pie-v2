# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from .cross_entropy_loss import CrossEntropyLoss  # noqa: F401
from .hard_mine_triplet_loss import TripletLoss  # noqa: F401
from .circle_loss import CircleLoss  # noqa: F401

def DeepSupervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.0
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
