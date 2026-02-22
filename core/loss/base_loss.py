from collections import OrderedDict
from typing import Dict

import torch
from torch import nn as nn


class BaseLoss(nn.Module):
    def __init__(self, cfg, model=None, *args, **kwargs):
        super().__init__()
        self.weight = cfg.weight

    def calculate(self, outputs, inputs):
        raise NotImplementedError

    def forward(self, outputs, inputs):
        loss = self.calculate(outputs, inputs)
        if isinstance(loss, torch.Tensor):
            return OrderedDict(
                [
                    (f"{self.__class__.__name__}", self.weight * loss)
                ]
            )
        elif isinstance(loss, Dict):
            return OrderedDict(
                [
                    (f"{self.__class__.__name__}.{name}", self.weight * _loss)
                    for name, _loss in loss.items()
                ]
            )
        else:
            raise TypeError(f"Loss type: {type(loss)} is not supported.")


