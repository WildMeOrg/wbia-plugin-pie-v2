# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
import torch
import torch.nn as nn


class CircleLoss(nn.Module):
    """Circle loss

    Reference:
        Circle Loss: A Unified Perspective of Pair Similarity Optimization. https://arxiv.org/abs/2002.10857.

    Adapted from `<https://github.com/TinyZeaMays/CircleLoss>`.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    def __init__(self, margin=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.margin = margin
        type(self.margin)
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp, sn):
        ap = torch.clamp_min(- sp.detach() + 1 + self.margin, min=0.)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


