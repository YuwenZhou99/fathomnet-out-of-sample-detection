import json
import os
import random
from dataclasses import dataclass
from typing import Any
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.utils import (
    parse_categories_cell,
    filter_existing_images,
    build_uuid_to_dive_map,
    dive_group_split,
    load_category_key,
    random_split,
)
from src.dataset.dataset import FathomNetMultiLabelDataset, build_transforms
import cv2
import ast


def parse_categories_float_list(cell) -> list[int]:
    """
    Parse cells like "[1.0, 218.0]" into list[int] -> [1, 218]
    """
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return [int(float(x)) for x in cell]

    s = str(cell).strip()
    if s == "" or s == "[]":
        return []

    try:
        values = ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Could not parse categories cell: {cell}") from e

    if not isinstance(values, (list, tuple)):
        return []

    return [int(float(v)) for v in values]


def prepare_dataloaders(general_cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader, dict[int, str], int, dict[int, int]]:
    """
    Returns:
      train_loader, val_loader, test_loader, cat_map, n_existing, new2orig

    Key change:
      - build train seen class set from label_csv
      - compute osd_target for eval_label_csv based on unseen classes
      - map BOTH train/val and test categories into the SAME compact label space
    """

    # -----------------------------
    # 1) Load TRAIN/VAL labels
    # -----------------------------
    labels_df = pd.read_csv(general_cfg["label_csv"])
    if general_cfg.get("quick_test", False):
        labels_df = labels_df.head(general_cfg["n_samples"])

    # parse original category ids (e.g., 218, 242)
    labels_df["categories_list"] = labels_df["categories"].apply(parse_categories_float_list)

    # Seen classes in training distribution (ORIGINAL ID space)
    train_class_set = set(c for cats in labels_df["categories_list"] for c in cats)

    # Build compact class space from training labels only
    existing_classes = sorted(train_class_set)
    n_existing = len(existing_classes)
    print("[INFO] Existing classes:", n_existing)

    orig2new = {orig_id: i for i, orig_id in enumerate(existing_classes)}
    new2orig = {i: orig_id for orig_id, i in orig2new.items()}

    # Convert TRAIN/VAL labels into compact indices (0..C-1)
    labels_df["labels"] = labels_df["categories_list"].apply(
        lambda cats: [orig2new[c] for c in cats if c in orig2new]
    )

    # 2) Filter to existing images for TRAIN/VAL
    labels_df = filter_existing_images(labels_df, general_cfg["image_dir"], general_cfg["img_ext"])

    # 3) Build uuid -> dive_id mapping from COCO train.json
    uuid_to_dive, annotations = build_uuid_to_dive_map(general_cfg["coco_train_json"])

    # add bbox column (kept as in your original)
    labels_df["bounding_boxes"] = [[] for _ in range(len(labels_df))]
    for ann in annotations:
        try:
            labels_df.iloc[ann["image_id"] - 1]["bounding_boxes"].append((ann["bbox"], ann["category_id"]))
        except IndexError:
            print(f"IndexError for image_id {ann['image_id']}")

    print(f"[INFO] uuid->dive_id map size: {len(uuid_to_dive)}")

    # Split (you used random_split; keep it)
    train_df, val_df = random_split(labels_df, val_ratio=general_cfg["val_ratio"], seed=general_cfg["seed"])
    os.makedirs(general_cfg["split_dir"], exist_ok=True)

    if not os.path.exists(os.path.join(general_cfg["split_dir"], "train_ids.json")) and not os.path.exists(
        os.path.join(general_cfg["split_dir"], "val_ids.json")
    ):
        with open(os.path.join(general_cfg["split_dir"], "train_ids.json"), "w") as f:
            json.dump(train_df["id"].tolist(), f)
        with open(os.path.join(general_cfg["split_dir"], "val_ids.json"), "w") as f:
            json.dump(val_df["id"].tolist(), f)

    print(f"[INFO] Split: train={len(train_df)} val={len(val_df)}")

    # -----------------------------
    # 4) Load TEST labels (eval_label.csv)
    # -----------------------------
    test_df = pd.read_csv(general_cfg["eval_label_csv"])
    if general_cfg.get("quick_test", False):
        test_df = test_df.head(general_cfg["n_samples"])

    # Parse original category ids for test
    test_df["categories_list"] = test_df["categories"].apply(parse_categories_float_list)

    # Compute osd_target from unseen classes (ORIGINAL ID space)
    # osd_target=1 if any class in image is NOT in train_class_set
    def compute_osd_target(cats: list[int]) -> int:
        return int(any(c not in train_class_set for c in cats))

    test_df["osd_target"] = test_df["categories_list"].apply(compute_osd_target)

    # Convert TEST labels into compact indices too (drop unseen classes because model can't predict them)
    test_df["labels"] = test_df["categories_list"].apply(
        lambda cats: [orig2new[c] for c in cats if c in orig2new]
    )

    # Filter to existing images for TEST
    test_df = filter_existing_images(test_df, general_cfg["eval_image_dir"], general_cfg["img_ext"])
    print(f"[INFO] Test set size: n={len(test_df)}")

    # Sanity check for osd_target distribution (must have both 0 and 1 for AUCROC)
    vc = test_df["osd_target"].value_counts(dropna=False).to_dict()
    print("[INFO] test osd_target counts:", vc)

    # Category names map (kept for display)
    cat_map = load_category_key(general_cfg["category_key_csv"])

    # -----------------------------
    # 5) Transforms
    # -----------------------------
    interpolation = cv2.INTER_LINEAR if general_cfg.get("interpolation", "bilinear") == "bilinear" else cv2.INTER_NEAREST

    train_tf, val_tf = build_transforms(
        general_cfg["image_size"],
        general_cfg.get("resize_size", 256),
        general_cfg.get("crop_size", 224),
        interpolation=interpolation,
        augmentations=general_cfg.get("augmentations", {}),
        mean=general_cfg.get("mean", [0.485, 0.456, 0.406]),
        std=general_cfg.get("std", [0.229, 0.224, 0.225]),
    )

    # -----------------------------
    # 6) Datasets / Loaders
    # IMPORTANT: dataset expects num_classes now
    # -----------------------------
    num_classes = n_existing

    train_ds = FathomNetMultiLabelDataset(
        train_df, num_classes, general_cfg["image_dir"], general_cfg["img_ext"], transform=train_tf
    )
    val_ds = FathomNetMultiLabelDataset(
        val_df, num_classes, general_cfg["image_dir"], general_cfg["img_ext"], transform=val_tf
    )
    test_ds = FathomNetMultiLabelDataset(
        test_df, num_classes, general_cfg["eval_image_dir"], general_cfg["img_ext"], transform=val_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=general_cfg["batch_size"],
        shuffle=False,
        num_workers=general_cfg["num_workers"],
        pin_memory=general_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=general_cfg["batch_size"],
        shuffle=False,
        num_workers=general_cfg["num_workers"],
        pin_memory=general_cfg["pin_memory"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=general_cfg["batch_size"],
        shuffle=False,
        num_workers=general_cfg["num_workers"],
        pin_memory=general_cfg["pin_memory"],
    )

    return train_loader, val_loader, test_loader, cat_map, n_existing, new2orig
