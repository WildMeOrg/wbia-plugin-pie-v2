# -*- coding: utf-8 -*-
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)

import logging
import torch.nn as nn
from torchvision import models as torchmodels  # NOQA
from efficientnet_pytorch import EfficientNet

logger = logging.getLogger(__name__)


NAME_EMBEDDING_SIZE = {
    'efficientnet-b0': 1280,
    'efficientnet-b1': 1280,
    'efficientnet-b2': 1408,
    'efficientnet-b3': 1536,
    'efficientnet-b4': 1792,
    'efficientnet-b5': 2048,
    'efficientnet-b6': 2304,
    'efficientnet-b7': 2560,
}


class EfficientNetReid(nn.Module):
    """Re-id model with EfficientNet as a convolutional feature extractor.
    Input:
        core_name (string): name of core model, class from torchvision.models
    """

    def __init__(self, core_name, num_classes, fc_dims, dropout_p, loss):
        super(EfficientNetReid, self).__init__()
        self.loss = loss

        self.core_model = EfficientNet.from_pretrained(core_name)

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self._construct_fc_layer(
            fc_dims, NAME_EMBEDDING_SIZE[core_name], dropout_p
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def featuremaps(self, x):
        return self.core_model.extract_features(x)

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

        if 'softmax' in self.loss:
            return y
        elif 'triplet' or 'circle' in self.loss:
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def efficientnet_b4(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = EfficientNetReid(
        core_name='efficientnet-b4',
        num_classes=num_classes,
        loss=loss,
        fc_dims=[512],
        dropout_p=None,
    )

    return model
