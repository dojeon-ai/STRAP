from abc import *
import torch.nn as nn
import torch


class BasePredictor(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.name

    def forward(self, state):
        # [param] state: (B, T, D)
        pass