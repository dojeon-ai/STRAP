from abc import *
import torch.nn as nn
from .base import BaseGraph


class IdentityGraph(BaseGraph):
    name = 'identity'
    def __init__(self, in_dim):
        super().__init__()
        self.output_dim = in_dim

    def get_output_dim(self):
        return self.output_dim

    def forward(self, x, edge):
        state = x['state']
        return state

