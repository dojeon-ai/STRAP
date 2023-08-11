from abc import *
import torch.nn as nn
import torch
EPS = 1e-5

class BaseBackbone(nn.Module, metaclass=ABCMeta):
    def __init__(self, categorical_ids):
        super().__init__()
        self.categorical_ids = categorical_ids

    @classmethod
    def get_name(cls):
        return cls.name
    
    @abstractmethod
    def get_output_dim(self):
        pass

    @abstractmethod
    def forward(self, batch):
        pass