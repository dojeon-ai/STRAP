import wandb
import torch
import os
import json
from dateutil.tz import gettz
from datetime import datetime
from omegaconf import OmegaConf
from src.common.class_utils import save__init__args
from collections import deque
import numpy as np
import pandas as pd
from pathlib import Path
import ipdb


class WandbTrainerLogger(object):
    def __init__(self, cfg):
        self.cfg = cfg
        dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(project=cfg.project_name,
                config=dict_cfg,
                group=cfg.exp_name,
                settings=wandb.Settings(start_method="thread"))  

        self.logger = TrainerLogger()

    def update_log(self, **kwargs):
        self.logger.update_log(**kwargs)

    def fetch_log(self):
        # logger's average meter set is automatically flushed
        log_data = self.logger.fetch_log()
        return log_data
    
    def write_log(self, log_data, step):
        wandb.log(log_data, step=step)

    def save_pred(self, pred_df, name):
        path = wandb.run.dir + '/' + str(name) + '/pred.csv'
        _dir = os.path.dirname(path)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        pred_df.to_csv(path)

    def save_state_dict(self, model, name):
        path = wandb.run.dir + '/' + str(name) + '/model.pth'
        _dir = os.path.dirname(path)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        state_dict = {'model_state_dict': model.state_dict()}
        torch.save(state_dict, path)
    
    def load_state_dict(self, path, device):
        return torch.load(path, map_location=device)


class TrainerLogger(object):
    def __init__(self):
        self.average_meter_set = AverageMeterSet()
        self.media_set = {}
    
    def update_log(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, float) or isinstance(v, int):
                self.average_meter_set.update(k, v)
            elif isinstance(v, tuple):
                _v, n = v
                self.average_meter_set.update(k, _v, n)
            else:
                self.media_set[k] = v

    def fetch_log(self):
        log_data = {}
        log_data.update(self.average_meter_set.averages())
        log_data.update(self.media_set)

        self.average_meter_set = AverageMeterSet()
        self.media_set = {}
        
        return log_data


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # TODO: description for using n
        self.val = val
        self.sum += (val * n)
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)