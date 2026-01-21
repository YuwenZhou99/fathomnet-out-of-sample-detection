import os
import random
from dataclasses import dataclass
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.utils import parse_categories_cell, filter_existing_images, build_uuid_to_dive_map, dive_group_split, load_category_key
from src.dataset.dataset import FathomNetMultiLabelDataset, build_transforms
from src.dataset.statistics import compute_pos_weight_tensor
from src.network.resnet_baseline import ResNetBaseline
from src.network.vit_baseline import ViTBaseline
from src.model.train_model import Trainer
from src.dataset.preprocessing import prepare_dataloaders
import yaml
import sys



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
    if len(sys.argv) == 2 and sys.argv[1] in ['resnet', 'efficientnet', 'vit_b_16', 'convnext-tiny']:
        model_yaml_path = f"./config/{sys.argv[1]}_baseline.yaml"
    else:
        print("Usage: python main.py [resnet|efficientnet|vit_b_16|convnext-tiny]")
        quit()
    general_cfg, model_cfg = load_yaml_config("./config/general.yaml", model_yaml_path=model_yaml_path)
    set_seed(general_cfg['seed'])
    # os.makedirs(cfg.split_dir, exist_ok=True)
    train_loader, val_loader, cat_map = prepare_dataloaders(general_cfg)

    pos_weight_tensor = compute_pos_weight_tensor(train_loader, device='cuda' if torch.cuda.is_available() else 'cpu')

    
    # TODO: model train and evaluation
    loss_fn = model_cfg.get('loss_fn', 'BCEWithLogits')
    optimizer = model_cfg.get('optimizer', 'Adam')

    if sys.argv[1] == 'resnet':
        model = ResNetBaseline(num_classes=len(cat_map.keys()), pretrained=True, layer_size=model_cfg.get('layer_size', None))
    elif sys.argv[1] == 'vit_b_16':
        model = ViTBaseline(num_classes=len(cat_map.keys()), pretrained=True, layer_size=model_cfg.get('layer_size', None))
    else:
        raise NotImplementedError(f"Model {sys.argv[1]} not implemented yet.")
        quit()
    
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