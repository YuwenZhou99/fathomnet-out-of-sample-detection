import os
import random
from dataclasses import dataclass
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.utils import parse_categories_cell, filter_existing_images, build_uuid_to_dive_map, dive_group_split, load_category_key
from src.dataset.dataset import FathomNetMultiLabelDataset, build_transforms
from src.network.resnet_baseline import ResNetBaseline
from src.model.train_model import Trainer
from src.dataset.preprocessing import prepare_dataloaders
import yaml




def load_yaml_config(general_yaml_path: str, model_yaml_path: str) -> dict:
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


def set_seed(seed: int):
    # not sure if this seed is used in other files too
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # load config
    general_cfg, model_cfg = load_yaml_config("./config/general.yaml", model_yaml_path="./config/resnet_baseline.yaml")
    set_seed(general_cfg['seed'])
    # os.makedirs(cfg.split_dir, exist_ok=True)
    train_loader, val_loader, cat_map = prepare_dataloaders(general_cfg)

    
    # TODO: model train and evaluation
    loss_fn = model_cfg.get('loss_fn', 'BCEWithLogits')
    optimizer = model_cfg.get('optimizer', 'Adam')
    RNBaseline = ResNetBaseline(num_classes=len(cat_map.keys()), pretrained=True)
    trainer = Trainer(
        model=RNBaseline,
        train_loader=train_loader,
        val_loader=val_loader,
        general_cfg=general_cfg,
        model_cfg=model_cfg,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    trainer.train()

if __name__ == "__main__":
    main()