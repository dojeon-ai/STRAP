import os
import ipdb
import argparse
from dotmap import DotMap
from pathlib import Path
from collections import OrderedDict, deque

import wandb
import torch
import numpy as np
import omegaconf
import hydra
import os
from hydra import compose, initialize

from src.dataloaders import build_dataloader
from src.models import build_model
from src.trainers import build_trainer
from src.common.train_utils import set_global_seeds
from src.common.logger import WandbTrainerLogger

def run(args):
    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    # Hydra Compose
    initialize(version_base=None, config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)

    # integrate configs
    param_dict = {'device': cfg.device,
                  'seed': cfg.seed,
                  'batch_size': cfg.batch_size,
                  'max_seq_length': cfg.max_seq_length}
    
    for key, value in param_dict.items():
        if key in cfg.dataloader:
            cfg.dataloader[key] = value
            
        if key in cfg.model.backbone:
            cfg.model.backbone[key] = value

        if key in cfg.model.graph:
            cfg.model.graph[key] = value
            
        if key in cfg.trainer:
            cfg.trainer[key] = value

    # seed & device
    set_global_seeds(cfg.seed)
    device = torch.device(cfg.device)

    # dataset & dataloader
    torch.set_num_threads(1) # when dataset on disk
    train_loader, val_loader, test_loader, categorical_ids, standardize_dict = build_dataloader(cfg.dataloader)

    # model
    model = build_model(cfg.model, categorical_ids)    

    if cfg.use_pretrained:
        model_path = cfg.pretrained_model_path
        print('use pretrained model from', model_path)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])

    # logger
    logger = WandbTrainerLogger(cfg)

    # trainer
    trainer = build_trainer(cfg=cfg.trainer,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            standardize_dict=standardize_dict,
                            device=device,
                            logger=logger,
                            model=model)

    trainer.train()
    wandb.finish()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_path',  type=str,    default='configs')
    parser.add_argument('--config_name', type=str,     default='strap') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()
    run(vars(args))
