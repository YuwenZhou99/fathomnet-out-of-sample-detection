import os
from typing import List
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
            img = self.transform(img)

        y = torch.zeros((len(self.all_cat_ids),), dtype=torch.float32)
        for cid in labels:
            if cid in self.id2idx:
                y[self.id2idx[cid]] = 1.0

        return {"id": uuid, "image": img, "target": y}


def build_transforms(image_size: int):
    # TODO: expand later to realize data augmentation
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf