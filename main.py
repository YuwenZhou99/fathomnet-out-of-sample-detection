import os
import random
from dataclasses import dataclass
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.utils import parse_categories_cell, filter_existing_images, load_category_key
from src.dataset.dataset import FathomNetMultiLabelDataset, build_transforms
from src.dataset.statistics import compute_pos_weight_tensor
from src.network.resnet_baseline import ResNetBaseline
from src.network.vit_baseline import ViTBaseline
from src.model.train_model import Trainer
from src.dataset.preprocessing import prepare_dataloaders
import yaml
import sys
from itertools import product


def load_yaml_config(general_yaml_path: str, model_yaml_path: str) -> tuple[dict, dict]:
    '''
    Returns config dictionaries for general purpose and model configurations
    
    :param general_yaml_path: Description
    :type general_yaml_path: str
    :param model_yaml_path: Description
    :type model_yaml_path: str
    :return: Dictionaries of general and model configurations
    :rtype: Tuple of dicts
    '''
    with open(general_yaml_path, "r") as inp:
        try:
            general_cfg = yaml.safe_load(inp)
        except yaml.YAMLError as exc:
            print(exc)

    with open(model_yaml_path, "r") as inp:
        try:
            model_cfg = yaml.safe_load(inp)
        except yaml.YAMLError as exc:
            print(exc)

    return general_cfg, model_cfg


def expand_grid(grid_params: dict):
    '''
    Expands every possibility for grid search of parameters in grid_params dict
    
    :param grid_params: parameters to be present in grid search
    :type grid_params: dict
    '''
    keys = grid_params.keys()
    values = grid_params.values()
    for combo in product(*values):
        yield dict(zip(keys, combo))


def set_seed(seed: int):
    # not sure if this seed is used in other files too
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # load config
    if len(sys.argv) == 2 and sys.argv[1] in ['resnet', 'efficientnet', 'vit_b_16', 'convnext-tiny', 'grid']:
        model_yaml_path = f"./config/{sys.argv[1]}_baseline.yaml"
    else:
        print("Usage: python main.py [resnet|efficientnet|vit_b_16|convnext-tiny|grid]")
        quit()
    if sys.argv[1] == 'grid':
        model_yaml_path = f"./config/resnet_baseline.yaml"  # placeholder, will be overridden in grid search
    general_cfg, model_cfg = load_yaml_config("./config/general.yaml", model_yaml_path=model_yaml_path)
    set_seed(general_cfg['seed'])
    train_loader, val_loader, cat_map = prepare_dataloaders(general_cfg)
    pos_weight_tensor = compute_pos_weight_tensor(train_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    loss_fn = model_cfg.get('loss_fn', 'BCEWithLogits')
    optimizer = model_cfg.get('optimizer', 'Adam')

    # determining what method to load
    if sys.argv[1] == 'resnet':
        model = ResNetBaseline(num_classes=len(cat_map.keys()), pretrained=True, layer_size=model_cfg.get('layer_size', None))
    elif sys.argv[1] == 'vit_b_16':
        model = ViTBaseline(num_classes=len(cat_map.keys()), pretrained=True, layer_size=model_cfg.get('layer_size', None))
    elif sys.argv[1] == 'grid':
        if not general_cfg.get('grid_search', False):
            raise NotImplementedError("Grid search not implemented yet.")
        grid_params = general_cfg.get('grid_params', {})
        configs = list(expand_grid(grid_params))
        print(len(configs))
        quit()
    else:
        raise NotImplementedError(f"Model {sys.argv[1]} not implemented yet.")
    
    # Initializing trainer object
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        general_cfg=general_cfg,
        model_cfg=model_cfg,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        pos_weight_tensor=pos_weight_tensor
    )
    trainer.train()
    trainer.plot_losses()

if __name__ == "__main__":
    main()