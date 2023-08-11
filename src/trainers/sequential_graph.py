from .base import BaseTrainer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import copy
import wandb
import tqdm
import ipdb


class SequentialGraphTrainer(BaseTrainer):
    name = 'sequential_graph'
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 val_loader,
                 test_loader,   
                 standardize_dict,
                 logger, 
                 model):
        
        super().__init__(cfg, device, 
                         train_loader, 
                         val_loader,
                         test_loader,
                         standardize_dict,
                         logger,
                         model)  
        
    def compute_loss(self, batch, mode):
        """
        [input] batch: (N + neighbors, T, D)
        [model output] preds: (N, 1) 
        [output] preds / targets / mask: (N, 1)
        """
        n = self.cfg.batch_size
        preds = self.model(batch).unsqueeze(-1)

        targets = batch['price'][:n, -1:]
        mask = batch['mask'][:n, -1:]
        
        cold_mask = (torch.sum(batch['mask'][:n], 1) < self.cfg.warm_start).unsqueeze(-1)
        warm_mask = (torch.sum(batch['mask'][:n], 1) >= self.cfg.warm_start).unsqueeze(-1)
        zero_mask = (torch.sum(batch['mask'][:n], 1) == 1).unsqueeze(-1)

        if self.cfg.loss_type == 'mse':
            loss = F.mse_loss(preds, targets, reduction='none')

        if self.cfg.loss_type == 'mae':
            loss = F.l1_loss(preds, targets, reduction='none')

        mask_dict = {
            'total': mask,
            'cold': cold_mask,
            'warm': warm_mask,
            'zero': zero_mask
        }

        return loss, preds, targets, mask_dict
