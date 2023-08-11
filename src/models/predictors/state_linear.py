from abc import *
import torch.nn as nn
from .base import BasePredictor


class StateLinearPredictor(BasePredictor):
    name = 'state_linear'
    def __init__(self, in_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1)
        )
                
    def forward(self, state):

        x = self.head(state).squeeze(-1)

        return x


