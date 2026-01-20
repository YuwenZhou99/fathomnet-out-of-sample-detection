import os
from typing import List
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FathomNetMultiLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, all_cat_ids: List[int], image_dir: str, img_ext: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.all_cat_ids = all_cat_ids
        self.id2idx = {cid: i for i, cid in enumerate(all_cat_ids)}
        self.image_dir = image_dir
        self.img_ext = img_ext
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        uuid = row["id"]
        labels = row["labels"]

        img_path = os.path.join(self.image_dir, uuid + self.img_ext)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            # Albumentations expects numpy array and returns dict with "image"
            if isinstance(self.transform, A.core.composition.Compose):
                img_np = np.array(img)
                img = self.transform(image=img_np)["image"]
            else:
                # torchvision-style transform
                img = self.transform(img)

        y = torch.zeros((len(self.all_cat_ids),), dtype=torch.float32)
        for cid in labels:
            if cid in self.id2idx:
                y[self.id2idx[cid]] = 1.0

        return {"id": uuid, "image": img, "target": y}


def build_transforms(image_size: int):
    # Train: Albumentations augmentation
    train_tf = A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.85, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.2),
        A.GaussNoise(std_range=(0.2, 0.4), p=0.3),

        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_tf = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tf, val_tf
