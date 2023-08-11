from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from .base import BaseGraph


# why use graphsage rather than graphconv?
# 1.GraphSage uses different weights for each iteration of the neighborhood aggregation.
#  -> GraphSage can handle different numbers of neighbors for different nodes.
# 2.GraphSage aggregates information of sampled neighbors, not the all connected nodes.
#  -> we have sampled the neighbors at the dataloading stage.
class GraphSage(BaseGraph):
    name = 'graphsage'
    def __init__(self,
                batch_size,
                in_dim,
                hid_dim,
                use_resident,
                use_poi,
                use_trp):
        super().__init__()

        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        self.use_resident = use_resident
        self.use_poi = use_poi
        self.use_trp = use_trp

        self.res_conv = SAGEConv(in_dim, hid_dim)
        self.poi_conv = SAGEConv(in_dim, hid_dim)
        self.trp_conv = SAGEConv(in_dim, hid_dim)

        self.norm = nn.LayerNorm(hid_dim)

    def get_output_dim(self):
        return self.hid_dim

    def forward(self, x, edge):
        B = self.batch_size
        transaction = x['transaction']
        res = x['res']
        poi = x['poi']
        trp = x['trp']

        # residual connection
        x = res[:B]
        y = transaction + x
        
        if self.use_resident:
            res_edge = edge['res']
            res_edge[:, 1] += B
            trg_res = res[B:]
            
            res_x = torch.cat((x, trg_res), dim=0)
            res_x = self.res_conv(res_x, res_edge.T)
            res_x = res_x[:B]
            y = y + res_x

        if self.use_poi:
            poi_edge = edge['poi']
            poi_edge[:, 1] += B
            poi_x = torch.cat((x, poi), dim=0)
            poi_x = self.poi_conv(poi_x, poi_edge.T)
            poi_x = poi_x[:B]
            y = y + poi_x

        if self.use_trp:
            trp_edge = edge['trp']
            trp_edge[:, 1] += B
            trp_x = torch.cat((x, trp), dim=0)
            trp_x = self.trp_conv(trp_x, trp_edge.T)
            trp_x = trp_x[:B]
            y = y + trp_x
        
        return y
