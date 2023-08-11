from .base import BaseTrainer
from dotmap import DotMap
from omegaconf import OmegaConf
import ipdb

from src.common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, BaseTrainer)

TRAINERS = {subclass.get_name():subclass
            for subclass in all_subclasses(BaseTrainer)}


def build_trainer(cfg,
                  device,
                  train_loader,
                  val_loader,
                  test_loader,
                  standardize_dict,
                  logger,
                  model):    
    cfg = DotMap(OmegaConf.to_container(cfg))

    # trainer
    trainer_type = cfg.pop('type')
    trainer = TRAINERS[trainer_type]
    return trainer(cfg=cfg,
                   device=device,
                   train_loader=train_loader,
                   val_loader=val_loader,
                   test_loader=test_loader,
                   standardize_dict=standardize_dict,
                   logger=logger,
                   model=model)