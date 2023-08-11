from .backbones import *  
from .graphs import * 
from .predictors import * 
from .base import Model
from omegaconf import OmegaConf
from src.common.class_utils import all_subclasses
import torch


BACKBONES = {subclass.get_name():subclass
            for subclass in all_subclasses(BaseBackbone)}

GRAPHS = {subclass.get_name():subclass
         for subclass in all_subclasses(BaseGraph)}

PREDICTORS = {subclass.get_name():subclass
             for subclass in all_subclasses(BasePredictor)}

def build_model(cfg, categorical_ids):
    cfg = OmegaConf.to_container(cfg)
    backbone_cfg = cfg['backbone']
    graph_cfg = cfg['graph']
    predictor_cfg = cfg['predictor']

    backbone_type = backbone_cfg.pop('type')
    graph_type = graph_cfg.pop('type')
    predictor_type = predictor_cfg.pop('type')

    # backbone
    backbone = BACKBONES[backbone_type]
    backbone = backbone(**backbone_cfg, categorical_ids=categorical_ids)
    
    # graph
    graph = GRAPHS[graph_type]
    graph_cfg['in_dim'] = backbone.get_output_dim()
    graph = graph(**graph_cfg)
    
    # predictor
    predictor = PREDICTORS[predictor_type]
    predictor_cfg['in_dim'] = graph.get_output_dim()
    predictor = predictor(**predictor_cfg)

    # model
    model = Model(backbone=backbone, graph=graph, predictor=predictor)
    
    return model
