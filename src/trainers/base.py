from abc import *
from typing import Tuple
import numpy as np
import tqdm
import random
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.common.train_utils import CosineAnnealingWarmupRestarts, get_grad_norm_stats
from sklearn.metrics import f1_score
from einops import rearrange
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

class BaseTrainer():
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 val_loader,
                 test_loader,
                 standardize_dict,
                 logger,
                 model):
        super().__init__()
        self.cfg = cfg  
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.standardize_dict = standardize_dict
        self.logger = logger

        self.model = model.to(self.device)
        self.optimizer = self._build_optimizer(cfg.optimizer)

    @classmethod
    def get_name(cls):
        return cls.name 

    def _build_optimizer(self, optimizer_cfg):
        optimizer_type = optimizer_cfg.pop('type')
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), 
                              **optimizer_cfg)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), 
                              **optimizer_cfg)
        else:
            raise ValueError
    
    @abstractmethod
    def compute_loss(self, batch, mode):
        """
        [output] loss
        [output] preds
        [output] targets
        [output] mask_dict: dictionary of masks to index the condition (e.g., cold-start transactions)
        """
        pass

    def _inv_normalize(self, value, key):
        mean, std = self.standardize_dict[key]
        value = value * std + mean

        return value

    def _rmse_mae_mre(self, preds, targets, reduction='mean'):
        preds = self._inv_normalize(preds, 'price')
        targets = self._inv_normalize(targets, 'price')
        EPS = torch.mean(targets) * 1e-2

        if reduction == 'mean':
            rmse = torch.sqrt(F.mse_loss(preds, targets))
            mae = F.l1_loss(preds, targets)
            mre = torch.mean(mae / (targets + EPS))

        elif reduction == 'none':
            rmse = torch.sqrt(F.mse_loss(preds, targets, reduction='none'))
            mae = F.l1_loss(preds, targets, reduction='none')
            mre = mae / (targets + EPS)

        return rmse, mae, mre

    def _compute_logs_with_mask(self, preds, targets, mask_dict, mode):
        log_data = {}
        
        for key, mask in mask_dict.items():
            _preds = preds[mask]
            _targets = targets[mask]

            if len(_preds) == 0:
                continue
            
            rmse, mae, mre = self._rmse_mae_mre(_preds, _targets)
            rmse, mae, mre = rmse.item(), mae.item(), mre.item()
            num_mask = torch.sum(mask).item()

            log_data[key + '_rmse'] = (rmse, num_mask)
            log_data[key + '_mae'] = (mae, num_mask)
            log_data[key + '_mre'] = (mre, num_mask)

        prefix = mode
        _log_data = {}
        for key, value in log_data.items():
            _log_data[prefix + '_' + key] = value
        log_data = _log_data

        return log_data
    
    def train(self):
        step = 0

        # initial evaluation
        val_logs = self.evaluate(mode='val', step=step)
        test_logs = self.evaluate(mode='test', step=step)
        self.logger.write_log(val_logs, step)
        self.logger.write_log(test_logs, step)
        best_metric_val = val_logs['val_loss']

        # train
        for e in range(1, self.cfg.num_epochs+1):
            for batch in tqdm.tqdm(self.train_loader): 
                self.model.train()
                self.optimizer.zero_grad()
                batch = {k:v.to(self.device) for k,v in batch.items()}
                loss, preds, targets, mask_dict = self.compute_loss(batch, mode='train')

                # cold-warm trade off
                warm_loss = torch.mean(loss * mask_dict['warm'])
                cold_loss = torch.mean(loss * mask_dict['cold'])
                loss = warm_loss + self.cfg.cold_lmbda * cold_loss
                loss = torch.mean(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                self.optimizer.step()

                # logging
                train_logs = self._compute_logs_with_mask(preds, targets, mask_dict, mode='train')
                train_logs['train_loss'] = loss.item()
                train_logs['train_warm_loss'] = warm_loss.item()
                train_logs['train_cold_loss'] = cold_loss.item()

                self.logger.update_log(**train_logs)
                if step % self.cfg.log_every == 0:
                    train_logs = self.logger.fetch_log()
                    self.logger.write_log(train_logs, step)

                step += 1

            if e % self.cfg.eval_every == 0:
                val_logs = self.evaluate(mode='val', step=step)
                self.logger.write_log(val_logs, step)

                metric_val = val_logs['val_loss']
                if metric_val < best_metric_val:
                    best_metric_val = metric_val
                    test_logs = self.evaluate(mode='test', step=step)
                    self.logger.write_log(test_logs, step)
            
            if e % self.cfg.save_every == 0:
                self.logger.save_state_dict(model=self.model, name=e)

    def evaluate(self, mode, step=0):
        self.model.eval()
        if mode == 'val':
            loader = self.val_loader
        elif mode == 'test':
            loader = self.test_loader
        else:
            raise ValueError

        ##################################
        # standard evaluation
        n = self.cfg.batch_size

        idx_list, preds_list, targets_list, history_len_list = [], [], [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(loader):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                loss, preds, targets, mask_dict = self.compute_loss(batch, mode=mode)
                eval_logs = self._compute_logs_with_mask(preds, targets, mask_dict, mode)

                eval_logs[mode + '_loss'] = torch.mean(loss).item()
                self.logger.update_log(**eval_logs)

                idx_list.append(batch['idx'][:n])
                preds_list.append(preds)
                targets_list.append(targets)
                history_len = torch.sum(batch['mask'][:n], 1)
                history_len_list.append(history_len)

        eval_logs = self.logger.fetch_log()

        ###################################
        # log the metric w.r.t history len
        idxs = torch.stack(idx_list).flatten()
        preds = torch.stack(preds_list).flatten()
        targets = torch.stack(targets_list).flatten()
        history_lens = torch.stack(history_len_list).flatten()

        rmse, mae, mre = self._rmse_mae_mre(preds, targets, reduction='none')
        rmse, mae, mre = rmse.cpu().numpy(), mae.cpu().numpy(), mre.cpu().numpy()
        history_lens = history_lens.cpu().numpy()

        bin_interval = self.cfg.history_bin_interval
        bin_cnt = self.cfg.max_seq_length // bin_interval
        history_bins = np.arange(bin_cnt) * bin_interval
        history_lens = np.digitize(history_lens, history_bins)
            
        bin_rmse_list, bin_mae_list, bin_mre_list = [], [], []
        for bin_idx in range(1, bin_cnt+1):
            bin_rmse = np.mean(rmse[history_lens==bin_idx])
            bin_mae = np.mean(mae[history_lens==bin_idx])
            bin_mre = np.mean(mre[history_lens==bin_idx])

            bin_rmse_list.append(bin_rmse)
            bin_mae_list.append(bin_mae)
            bin_mre_list.append(bin_mre)

        xticks = []
        history_bins = history_bins.tolist()
        history_bins.append(bin_interval * bin_cnt)
        for idx in range(bin_cnt):
            bin_start = history_bins[idx] 
            bin_end = history_bins[idx + 1]

            xtick = str(bin_start) + '-' + str(bin_end)
            xticks.append(xtick)

        def draw_histogram(x, y, xticks):
            fig, ax = plt.subplots()
            sns.barplot(x=x, y=y, ax=ax)
            ax.set_xticklabels(xticks)

            return fig

        ##########################
        # save the prediction
        pred_df = pd.DataFrame(columns=['trans_idx', 'preds', 'targets'])
        pred_df['trans_idx'] = loader.sampler.indices[:len(preds)]
        preds = self._inv_normalize(preds, 'price')
        targets = self._inv_normalize(targets, 'price')
        pred_df['preds'] = preds.cpu().numpy()
        pred_df['targets'] = targets.cpu().numpy()

        self.logger.save_pred(pred_df=pred_df, name=step)

        return eval_logs

