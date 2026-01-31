import os
from typing import Any, List
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FathomNetMultiLabelDataset(Dataset):
    """
    Returns:
      - (img, y, uuid) if no osd_target column exists
      - (img, y, uuid, osd_target) if osd_target exists
    Assumption:
      - df["labels"] is a list of compact class indices in [0, num_classes-1]
    """
    def __init__(self, df: pd.DataFrame, num_classes: int,
                 image_dir: str, img_ext: str, transform=None,
                 osd_col: str = "osd_target"):
        self.df = df.reset_index(drop=True)
        self.num_classes = int(num_classes)
        self.image_dir = image_dir
        self.img_ext = img_ext
        self.transform = transform
        self.osd_col = osd_col
        self.has_osd = (osd_col in self.df.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        uuid = row["id"]
        labels = row["labels"]  # list[int] in compact index space

        img_path = os.path.join(self.image_dir, uuid + self.img_ext)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            if isinstance(self.transform, A.core.composition.Compose):
                img_np = np.array(img)
                img = self.transform(image=img_np)["image"]
            else:
                img = self.transform(img)

        y = torch.zeros((self.num_classes,), dtype=torch.float32)
        for cid in labels:
            cid = int(cid)
            if 0 <= cid < self.num_classes:
                y[cid] = 1.0

        if self.has_osd:
            osd_target = float(row[self.osd_col])
            osd_target = torch.tensor(osd_target, dtype=torch.float32)
            return img, y, uuid, osd_target

        return img, y, uuid


def build_transforms(image_size: int, resize_size: int, crop_size: int, interpolation: int,
                     augmentations: dict[str, Any],
                     mean: List[float], std: List[float]):
    train_tf = A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=interpolation),
        A.HorizontalFlip(p=augmentations.get('horizontal_flip', 0.5)),
        A.Rotate(limit=augmentations.get('rotation_limit', 10), p=augmentations.get('rotation_prob', 0.3), border_mode=0),
        A.RandomBrightnessContrast(p=augmentations.get('brightness_contrast', 0.3)),
        A.HueSaturationValue(p=augmentations.get('hue_saturation', 0.2)),
        A.GaussNoise(
            std_range=(
                augmentations.get('gauss_noise', [0.2, 0.4])[0],
                augmentations.get('gauss_noise', [0.2, 0.4])[1]),
                p=augmentations.get('gauss_noise_prob', 0.3)
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    debug_tf = A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=interpolation),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    val_tf = A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=interpolation),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return debug_tf, val_tf
