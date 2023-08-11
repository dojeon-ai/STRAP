import torch
import torch.nn as nn
import ipdb
import time

class Model(nn.Module):
    def __init__(self, backbone, graph, predictor):
        super().__init__()
        self.backbone = backbone
        self.graph = graph
        self.predictor = predictor

    def forward(self, x):
        edges = {}
        if 'res_edge' in x:
            edges['res'] = x['res_edge']
        edges['poi'] = x['poi_edge']
        edges['trp'] = x['trp_edge']

        # state: (B, T, D) [s_0, s_1, ..., s_T]
        # price: (B, T, D  [p_0, p_1, ..., p_T]

        x = self.backbone(x)    # torch.Size([1480, 50, 256])
        x = self.graph(x, edges)    # torch.Size([256, 1, 256])
        
        # prediction: (B, T)
        x = self.predictor(x)   # torch.Size([256, 1])

        return x
