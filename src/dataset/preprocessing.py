import json
import os
import random
from dataclasses import dataclass
from typing import Any
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.utils import parse_categories_cell, filter_existing_images, build_uuid_to_dive_map, dive_group_split, \
    load_category_key, random_split
from src.dataset.dataset import FathomNetMultiLabelDataset, build_transforms
import cv2


def parse_categories(cat_str: str) -> list[int]:
    if pd.isna(cat_str) or cat_str == "":
        return []
    return [int(x) for x in cat_str.replace(",", " ").split()]

import ast
import pandas as pd

def parse_categories_float_list(cell) -> list[int]:
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        # in case you already parsed it elsewhere
        return [int(float(x)) for x in cell]

    s = str(cell).strip()
    if s == "" or s == "[]":
        return []

    # safe parse of Python literal lists: "[1.0, 9.0]"
    try:
        values = ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Could not parse categories cell: {cell}") from e

    if not isinstance(values, (list, tuple)):
        return []

    # convert float-like to int ids
    return [int(float(v)) for v in values]


def compact_labels_to_str(cats: list[int]) -> str:
    return "[" + ", ".join(f"{float(c):.1f}" for c in cats) + "]"


def prepare_dataloaders(general_cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[int, str]]:
    '''
    Prepares dataloaders based on configurations in general_cfg

    :param general_cfg: Descriptions for where to load data and how to split
    :type general_cfg: dict[str, Any]
    :return: Tuple of dataloaders and dict which maps categories to keys
    :rtype: tuple[DataLoader, DataLoader, dict[int, str]]
    '''

    # 1) Load labels
    labels_df = pd.read_csv(general_cfg['label_csv'])
    if general_cfg.get('quick_test', False):
        labels_df = labels_df.head(general_cfg['n_samples'])

    labels_df["categories_list"] = labels_df["categories"].apply(parse_categories_float_list)
    
    existing_classes = sorted({c for cats in labels_df["categories_list"] for c in cats})
    n_existing = len(existing_classes)
    print(existing_classes)
    print("Existing classes:", n_existing)
    
    orig2new = {orig_id: i for i, orig_id in enumerate(existing_classes)}
    new2orig = {i: orig_id for orig_id, i in orig2new.items()}

    labels_df["labels_compact"] = labels_df["categories_list"].apply(
    lambda cats: [orig2new[c] for c in cats if c in orig2new]
)
    labels_df["categories"] = labels_df["labels_compact"].apply(compact_labels_to_str)


    print(labels_df.head())

    

    labels_df["labels"] = labels_df["categories"].apply(parse_categories_cell)

    # 2) Filter to existing images
    labels_df = filter_existing_images(labels_df, general_cfg['image_dir'], general_cfg['img_ext'])

    # 3) Build uuid -> dive_id mapping from COCO train.json
    uuid_to_dive, annotations = build_uuid_to_dive_map(general_cfg['coco_train_json'])

    # adding annotations to dataframe
    labels_df["bounding_boxes"] = [[] for _ in range(len(labels_df))]
    for ann in annotations:
        try:
            labels_df.iloc[ann['image_id'] - 1]["bounding_boxes"].append((ann['bbox'], ann['category_id']))
        except IndexError:
            print(f"IndexError for image_id {ann['image_id']}")

    # not sure how robust this code is, might need to be changed

    print(f"[INFO] uuid->dive_id map size: {len(uuid_to_dive)}")

    #train_df, val_df = dive_group_split(labels_df, uuid_to_dive, general_cfg['val_ratio'], general_cfg['seed'])
    train_df, val_df = random_split(labels_df, val_ratio=general_cfg['val_ratio'], seed=general_cfg['seed'])
    os.makedirs(general_cfg['split_dir'], exist_ok=True)
    # check if train/val ids files already exist
    if not os.path.exists(os.path.join(general_cfg['split_dir'], 'train_ids.json')) and not os.path.exists(
            os.path.join(general_cfg['split_dir'], 'val_ids.json')):
        with open(os.path.join(general_cfg['split_dir'], 'train_ids.json'), 'w') as f:
            lst = train_df['id'].tolist()
            json.dump(lst, f)
        with open(os.path.join(general_cfg['split_dir'], 'val_ids.json'), 'w') as f:
            lst = val_df['id'].tolist()
            json.dump(lst, f)
    print(f"[INFO] Split by dive_id: train={len(train_df)} val={len(val_df)}")
    #print(f"[INFO] Unique dives: train={train_df['dive_id'].nunique()} val={val_df['dive_id'].nunique()}")

    # 5) Load test labels from organizer-provided eval set
    test_df = pd.read_csv(general_cfg['eval_label_csv'])
    test_df["labels"] = test_df["categories"].apply(parse_categories_cell)
    if general_cfg.get('quick_test', False):
        test_df = test_df.head(general_cfg['n_samples'])

    test_df = filter_existing_images(test_df, general_cfg['eval_image_dir'], general_cfg['img_ext'])
    print(f"[INFO] Test set (organizer eval): n={len(test_df)}")

    # 6) Category space
    # this is weird since certain categories are not in train set. This gives error when calculating loss with mismatching shapes
    all_cat_ids = sorted({cid for labs in labels_df["labels"].tolist() for cid in labs})
    print(f"[INFO] num_classes={len(all_cat_ids)}")

    # category names
    cat_map = load_category_key(general_cfg['category_key_csv'])
    # 7) DataLoaders
    if general_cfg.get('interpolation', 'bilinear') == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    train_tf, val_tf = build_transforms(general_cfg['image_size'], general_cfg.get('resize_size', 256),
                                        general_cfg.get('crop_size', 224),
                                        interpolation=interpolation,
                                        augmentations=general_cfg.get('augmentations', {}),
                                        mean=general_cfg.get('mean', [0.485, 0.456, 0.406]),
                                        std=general_cfg.get('std', [0.229, 0.224, 0.225]))
    train_ds = FathomNetMultiLabelDataset(train_df, new2orig, general_cfg['image_dir'], general_cfg['img_ext'],
                                          transform=train_tf)

    # not sure if val should have augmentations
    val_ds = FathomNetMultiLabelDataset(val_df, new2orig, general_cfg['image_dir'], general_cfg['img_ext'],
                                        transform=val_tf)

    test_ds = FathomNetMultiLabelDataset(test_df, cat_map, general_cfg['eval_image_dir'], general_cfg['img_ext'],
                                        transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=general_cfg['batch_size'], shuffle=False,
        num_workers=general_cfg['num_workers'], pin_memory=general_cfg['pin_memory']
    )
    val_loader = DataLoader(
        val_ds, batch_size=general_cfg['batch_size'], shuffle=False,
        num_workers=general_cfg['num_workers'], pin_memory=general_cfg['pin_memory']
    )
    test_loader = DataLoader(
       test_ds, batch_size=general_cfg['batch_size'], shuffle=False,
       num_workers=general_cfg['num_workers'], pin_memory=general_cfg['pin_memory']
    )

    return train_loader, val_loader, test_loader, cat_map, n_existing, new2orig