# -*- coding: utf-8 -*-
"""
Adapted from source: https://github.com/KaiyangZhou/deep-person-reid
"""
from __future__ import absolute_import
from .resnet import resnet50_fc512, resnext101_32x8d, resnet152
from .efficientnet import efficientnet_b4

__model_factory = {
    'resnet50_fc512': resnet50_fc512,
    'resnext101_32x8d': resnext101_32x8d,
    'efficientnet_b4': efficientnet_b4,
    'resnet152': resnet152,
}


def show_avai_models():
    """Displays available models."""
    print(list(__model_factory.keys()))


def build_model(name, num_classes, loss='softmax', pretrained=True, use_gpu=True):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained
            weights. Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    return __model_factory[name](
        num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu
    )
