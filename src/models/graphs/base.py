from abc import *
import torch.nn as nn
import torch


class BaseGraph(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.name

    def get_output_dim(self):
        pass

    def forward(self, x, edge):
        pass