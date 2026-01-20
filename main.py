import os
import random
from dataclasses import dataclass
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import parse_categories_cell, filter_existing_images, build_uuid_to_dive_map, dive_group_split, load_category_key
from dataset import FathomNetMultiLabelDataset, build_transforms
import yaml



def load_yaml_config(path: str) -> dict:
    with open(path, "r") as inp:
        try:
            general_cfg = yaml.safe_load(inp)
        except yaml.YAMLError as exc:
            print(exc)
    return general_cfg


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # load config
    cfg = load_yaml_config("./config/general.yaml")
    set_seed(cfg['seed'])
    # os.makedirs(cfg.split_dir, exist_ok=True)

    # 1) Load labels
    labels_df = pd.read_csv(cfg['label_csv'])
    labels_df["labels"] = labels_df["categories"].apply(parse_categories_cell)

    # 2) Filter to existing images
    labels_df = filter_existing_images(labels_df, cfg['image_dir'], cfg['img_ext'])

    # 3) Build uuid -> dive_id mapping from COCO train.json
    uuid_to_dive = build_uuid_to_dive_map(cfg['coco_train_json'])
    print(f"[INFO] uuid->dive_id map size: {len(uuid_to_dive)}")

    # 4) Split by dive_id groups
    train_df, val_df = dive_group_split(labels_df, uuid_to_dive, cfg['val_ratio'], cfg['seed'])
    print(f"[INFO] Split by dive_id: train={len(train_df)} val={len(val_df)}")
    print(f"[INFO] Unique dives: train={train_df['dive_id'].nunique()} val={val_df['dive_id'].nunique()}")

    # 5) Category space
    all_cat_ids = sorted({cid for labs in labels_df["labels"].tolist() for cid in labs})
    print(f"[INFO] num_classes={len(all_cat_ids)}")

    # category names
    cat_map = load_category_key(cfg['category_key_csv'])
    # 6) DataLoaders
    train_tf, val_tf = build_transforms(cfg['image_size'])
    train_ds = FathomNetMultiLabelDataset(train_df, all_cat_ids, cfg['image_dir'], cfg['img_ext'], transform=train_tf)
    val_ds = FathomNetMultiLabelDataset(val_df, all_cat_ids, cfg['image_dir'], cfg['img_ext'], transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
    )

    # TODO: model train and evaluation

if __name__ == "__main__":
    main()