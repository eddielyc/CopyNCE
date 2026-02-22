from functools import partial

from loguru import logger

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict


__loss_factory__ = {}


def register_loss(loss_name=""):
    def wrapper(cls):
        if loss_name in __loss_factory__:
            logger.warning(
                f'Overwriting {loss_name} in registry with {cls.__name__}. This is because the name being '
                'registered conflicts with an existing name. Please check if this is not expected.'
            )
        __loss_factory__[loss_name] = cls
        return cls

    return wrapper


class Criterion(nn.Module):
    def __init__(self, cfg, model, key):
        super().__init__()
        self.criterions = {
            loss_cfg.type: __loss_factory__[loss_cfg.type](loss_cfg, model)
            for loss_cfg in cfg
        }
        self.key = key

        logger.info(self)

    def __repr__(self):
        return f"Build {self.key}Criterion, including {', '.join(loss_type for loss_type in self.criterions)}."

    def forward(self, outputs, inputs):
        losses = OrderedDict()
        for loss_type, criterion in self.criterions.items():
            loss_outputs = criterion(outputs, inputs)
            for loss_name, loss in loss_outputs.items():
                losses[f"{self.key}.{loss_name}"] = loss
        return losses


def build_loss(cfg, model):
    criterions = []
    if hasattr(cfg.loss, "segmentation_loss") and cfg.loss.segmentation_loss is not None:
        criterions.append(Criterion(cfg.loss.segmentation_loss, model, key='seg'))
    if hasattr(cfg.loss, "classification_loss") and cfg.loss.classification_loss is not None:
        criterions.append(Criterion(cfg.loss.classification_loss, model, key='cls'))
    if hasattr(cfg.loss, "auxiliary_loss") and cfg.loss.auxiliary_loss is not None:
        criterions.append(Criterion(cfg.loss.auxiliary_loss, model, key='aux'))

    def wrapper(*criterions):
        def caller(outputs, inputs):
            loss_dict = OrderedDict()
            for criterion in criterions:
                if criterion:
                    loss_dict.update(criterion(outputs, inputs))
            return loss_dict
        return caller

    return wrapper(*criterions)
