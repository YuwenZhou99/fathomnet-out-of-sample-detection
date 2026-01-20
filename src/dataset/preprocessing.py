import os
import random
from dataclasses import dataclass
from typing import Any
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.utils import parse_categories_cell, filter_existing_images, build_uuid_to_dive_map, dive_group_split, load_category_key
from src.dataset.dataset import FathomNetMultiLabelDataset, build_transforms

def prepare_dataloaders(general_cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[int, str]]:

    # 1) Load labels
    labels_df = pd.read_csv(general_cfg['label_csv'])
    labels_df["labels"] = labels_df["categories"].apply(parse_categories_cell)
    # 2) Filter to existing images
    labels_df = filter_existing_images(labels_df, general_cfg['image_dir'], general_cfg['img_ext'])
    # 3) Build uuid -> dive_id mapping from COCO train.json
    uuid_to_dive, annotations = build_uuid_to_dive_map(general_cfg['coco_train_json'])

    # adding annotations to dataframe
    labels_df["bounding_boxes"] = [[] for _ in range(len(labels_df))]
    for ann in annotations:
        try:
            labels_df.iloc[ann['image_id']-1]["bounding_boxes"].append((ann['bbox'], ann['category_id']))
        except IndexError:
            print(f"IndexError for image_id {ann['image_id']}")

    # not sure how robust this code is, might need to be changed
    
    print(f"[INFO] uuid->dive_id map size: {len(uuid_to_dive)}")

    train_df, val_df = dive_group_split(labels_df, uuid_to_dive,general_cfg['val_ratio'], general_cfg['seed'])
    print(f"[INFO] Split by dive_id: train={len(train_df)} val={len(val_df)}")
    print(f"[INFO] Unique dives: train={train_df['dive_id'].nunique()} val={val_df['dive_id'].nunique()}")

    # 5) Category space
    # this is weird since certain categories are not in train set. This gives error when calculating loss with mismatching shapes
    all_cat_ids = sorted({cid for labs in labels_df["labels"].tolist() for cid in labs})
    print(f"[INFO] num_classes={len(all_cat_ids)}")

    # category names
    cat_map = load_category_key(general_cfg['category_key_csv'])
    # 6) DataLoaders
    train_tf, val_tf = build_transforms(general_cfg['image_size'])
    train_ds = FathomNetMultiLabelDataset(train_df, cat_map, general_cfg['image_dir'], general_cfg['img_ext'], transform=train_tf)

    # not sure if val should have augmentations
    val_ds = FathomNetMultiLabelDataset(val_df, cat_map, general_cfg['image_dir'], general_cfg['img_ext'], transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=general_cfg['batch_size'], shuffle=True,
        num_workers=general_cfg['num_workers'], pin_memory=general_cfg['pin_memory']
    )
    val_loader = DataLoader(
        val_ds, batch_size=general_cfg['batch_size'], shuffle=False,
        num_workers=general_cfg['num_workers'], pin_memory=general_cfg['pin_memory']
    )

    return train_loader, val_loader, cat_map