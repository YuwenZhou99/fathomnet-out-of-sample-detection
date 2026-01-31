import os
import random
from dataclasses import dataclass
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.utils import parse_categories_cell, filter_existing_images, build_uuid_to_dive_map, dive_group_split, \
    load_category_key
from src.dataset.dataset import FathomNetMultiLabelDataset, build_transforms
from src.dataset.statistics import compute_pos_weight_tensor
from src.network.resnet_baseline import ResNetBaseline
from src.network.vit_baseline import ViTBaseline
from src.model.train_model import Trainer
from src.dataset.preprocessing import prepare_dataloaders
import yaml
import sys
from itertools import product
import torch.nn as nn


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

def overfit_one_batch(model, train_loader, device, pos_weight_tensor, lr=1e-4, steps=900, val_loader=None):
        model.to(device)
        model.train()

        for p in model.model.parameters():
            p.requires_grad = False
        for p in model.model.fc.parameters():
            p.requires_grad = True

        x, y, *rest = next(iter(train_loader))
        x = x.to(device)
        y = y.to(device)
        print(x.dtype, x.min().item(), x.max().item(), x.shape)
        print(y.dtype, y.min().item(), y.max().item(), y.shape)
        print("unique target values (sample):", torch.unique(y)[:20])
        print("positives per sample:", y.sum(dim=1)[:10])

        # IMPORTANT: logits loss expects raw logits (no sigmoid)
        optim = torch.optim.AdamW(model.model.fc.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        img1, y1, *r1 = train_loader.dataset[0]
        img2, y2, *r2 = train_loader.dataset[0]
        print(torch.allclose(img1, img2), torch.equal(y1, y2))
        
   

        for t in range(steps):
            optim.zero_grad(set_to_none=True)
            logits = model(x)

            # sanity checks
            assert logits.shape == y.shape, (logits.shape, y.shape)
            assert torch.isfinite(logits).all()

            loss = criterion(logits, y)
            loss.backward()
            
            # 1) gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            optim.step()


            if t % 25 == 0 or t == steps - 1:
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    pred_pos_rate = (probs > 0.5).float().mean().item()
                    true_pos_rate = y.mean().item()
                print(f"step {t:4d} | loss {loss.item():.4f} | pred_pos_rate {pred_pos_rate:.4f} | true_pos_rate {true_pos_rate:.4f}")
        return loss.item()

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
    #train_loader, val_loader, test_loader, cat_map = prepare_dataloaders(general_cfg)
    train_loader, val_loader, cat_map, n_existing, new2orig = prepare_dataloaders(general_cfg)

    save_dir = general_cfg.get('save_dir_pos_weight', 'src/dataset/pos_weights')
    os.makedirs(save_dir, exist_ok=True)
    pos_weight_path = os.path.join(save_dir, 'pos_weight.pt')

    pos_weight_tensor = compute_pos_weight_tensor(
        train_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_path=pos_weight_path,
    )
    print(pos_weight_tensor)
    print(pos_weight_tensor.shape)
    
    loss_fn = model_cfg.get('loss_fn', 'BCEWithLogits')
    optimizer = model_cfg.get('optimizer', 'Adam')

    # determining what method to load
    if sys.argv[1] == 'resnet':
        model = ResNetBaseline(num_classes=len(new2orig.keys()), pretrained=True, layer_size=model_cfg.get('layer_size', None))
    elif sys.argv[1] == 'vit_b_16':
        model = ViTBaseline(num_classes=len(cat_map.keys()), pretrained=True, layer_size=model_cfg.get('layer_size', None))
    elif sys.argv[1] == 'grid':
        if not general_cfg.get('grid_search', False):
            raise NotImplementedError("Grid search not implemented yet.")
        grid_params = general_cfg.get('grid_params', {})
        configs = list(expand_grid(grid_params))
        meta_model = model_cfg.copy()
        for config in configs[2:]:
            model_cfg = meta_model.copy()
            for k, v in config.items():
                model_cfg[k] = v
            model_cfg['model_name'] = f"{model_cfg['model_name']}_layer:{model_cfg['layer_size']}_lr:{model_cfg['learning_rate']}_smoothep:{model_cfg['smoothing_epsilon']}_freeze:{str(model_cfg['freeze'])}"
            if config['model_name'] == 'resnet50_baseline':
                model = ResNetBaseline(num_classes=n_existing, pretrained=True, layer_size=model_cfg.get('layer_size', None))
            else:
                model = ViTBaseline(num_classes=len(cat_map.keys()), pretrained=True, layer_size=model_cfg.get('layer_size', None))
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=train_loader,
                general_cfg=general_cfg,
                model_cfg=model_cfg,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                pos_weight_tensor=pos_weight_tensor
        )
            trainer.train()
            trainer.plot_losses()
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
        pos_weight_tensor=pos_weight_tensor,
        n_classes=n_existing
    )
    #overfit_one_batch(model, train_loader, 'cuda', pos_weight_tensor, 1e-3, 300, val_loader=val_loader)
    trainer.train()
    trainer.plot_losses()

if __name__ == "__main__":
    main()