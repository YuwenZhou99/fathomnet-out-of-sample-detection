import pandas as pd
from typing import Any
from torch.utils.data import DataLoader
import cv2

from src.utils import parse_categories_cell, filter_existing_images, load_category_key
from src.dataset.dataset import FathomNetMultiLabelDataset, build_transforms


def prepare_dataloaders(general_cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[int, str]]:
    '''
    Prepares dataloaders based on configurations in general_cfg
    
    :param general_cfg: Descriptions for where to load data and how to split
    :type general_cfg: dict[str, Any]
    :return: Tuple of dataloaders and dict which maps categories to keys
    :rtype: tuple[DataLoader, DataLoader, dict[int, str]]
    '''

    # 1) Load category key and define a stable class space
    cat_map = load_category_key(general_cfg['category_key_csv'])
    all_cat_ids = sorted(cat_map.keys())
    print(f"[INFO] num_classes(from category_key)={len(all_cat_ids)}")

    # 2) Load train labels
    train_df = pd.read_csv(general_cfg['train_label_csv'])
    train_df["labels"] = train_df["categories"].apply(parse_categories_cell)
    if general_cfg.get('quick_test', False):
        train_df = train_df.head(general_cfg['n_samples'])
    train_df = filter_existing_images(train_df, general_cfg['train_image_dir'], general_cfg['img_ext'])

    # 3) Load eval labels
    eval_df = pd.read_csv(general_cfg['eval_label_csv'])
    eval_df["labels"] = eval_df["categories"].apply(parse_categories_cell)
    if general_cfg.get('quick_test', False):
        eval_df = eval_df.head(general_cfg['n_samples'])
    eval_df = filter_existing_images(eval_df, general_cfg['eval_image_dir'], general_cfg['img_ext'])

    print(f"[INFO] train={len(train_df)} eval={len(eval_df)}")

    # 4) Transforms
    interpolation = cv2.INTER_LINEAR if general_cfg.get('interpolation',
                                                        'bilinear') == 'bilinear' else cv2.INTER_NEAREST
    train_tf, eval_tf = build_transforms(
        general_cfg['image_size'],
        general_cfg.get('resize_size', 256),
        general_cfg.get('crop_size', 224),
        interpolation=interpolation,
        augmentations=general_cfg.get('augmentations', {}),
        mean=general_cfg.get('mean', [0.485, 0.456, 0.406]),
        std=general_cfg.get('std', [0.229, 0.224, 0.225])
    )

    # 5) Datasets
    train_ds = FathomNetMultiLabelDataset(train_df, all_cat_ids, general_cfg['train_image_dir'], general_cfg['img_ext'],
                                          transform=train_tf)
    eval_ds = FathomNetMultiLabelDataset(eval_df, all_cat_ids, general_cfg['eval_image_dir'], general_cfg['img_ext'],
                                         transform=eval_tf)

    # 6) Loaders
    train_loader = DataLoader(
        train_ds, batch_size=general_cfg['batch_size'], shuffle=True,
        num_workers=general_cfg['num_workers'], pin_memory=general_cfg['pin_memory']
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=general_cfg['batch_size'], shuffle=False,
        num_workers=general_cfg['num_workers'], pin_memory=general_cfg['pin_memory']
    )

    return train_loader, eval_loader, cat_map