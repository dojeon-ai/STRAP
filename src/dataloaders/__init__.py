from .base import BaseLoader
from omegaconf import OmegaConf
from src.common.class_utils import all_subclasses, import_all_subclasses

import_all_subclasses(__file__, __name__, BaseLoader)

LOADERS = {subclass.get_name():subclass
          for subclass in all_subclasses(BaseLoader)}

def build_dataloader(cfg):
    cfg = OmegaConf.to_container(cfg)   # Omegaconf func
    loader_type = cfg.pop('type')   # sequential_graph
    loader = LOADERS[loader_type](**cfg)    # parameters

    train_loader, val_loader, test_loader = loader.get_dataloader()
    categorical_ids = loader.categorical_ids
    standardize_dict = loader.standardize_dict

    return train_loader, val_loader, test_loader, categorical_ids, standardize_dict
