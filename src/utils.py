import os
import ast
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import GroupShuffleSplit


def parse_categories_cell(x: str) -> List[int]:
    vals = ast.literal_eval(x) # e.g. [1.0, 9.0, 11.0]
    out = []
    for v in vals:
        out.append(int(float(v)))

    return out


def load_category_key(path: str) -> Dict[int, Dict[str, str]]:
    df = pd.read_csv(path)
    mapping = {}
    for _, row in df.iterrows():
        mapping[int(row["id"])] = {"name": str(row["name"]), "supercat": str(row["supercat"])}

    return mapping


def filter_existing_images(df: pd.DataFrame, image_dir: str, img_ext: str) -> pd.DataFrame:
    def exists_fn(uuid: str) -> bool:
        return os.path.exists(os.path.join(image_dir, uuid + img_ext))

    mask = df["id"].apply(exists_fn)
    missing = int((~mask).sum())
    if missing > 0:
        print(f"[WARN] {missing} samples missing image files in '{image_dir}'. Dropping them.")

    return df[mask].reset_index(drop=True)

def uuid_from_file_name(file_name: str) -> str:
    # "uuid.png" -> "uuid"
    return os.path.splitext(os.path.basename(file_name))[0]


def extract_dive_id_from_coco_url(url: str) -> Optional[str]:
    """
    Example:
      https://fathomnet.org/static/m3/framegrabs/Tiburon/images/1120/03_50_15_04.png
    dive_id = 1120
    """
    if not isinstance(url, str):
        return None
    token = "/images/"
    if token not in url:
        return None
    tail = url.split(token, 1)[1]  # "1120/03_50_15_04.png"
    dive = tail.split("/", 1)[0]
    return dive if dive.isdigit() else dive  # sometimes not purely digits; keep as string


def build_uuid_to_dive_map(coco_json_path: str) -> Tuple[Dict[str, str], list[dict]]:
    """
    Build mapping from uuid -> dive_id by reading COCO train.json images list.
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    m: Dict[str, str] = {}
    for im in coco.get("images", []):
        fn = im.get("file_name", "")
        uuid = uuid_from_file_name(fn)
        url = im.get("coco_url", im.get("flickr_url", ""))
        dive = extract_dive_id_from_coco_url(url)
        if uuid and dive:
            m[uuid] = str(dive)
    annotations = coco.get("annotations", [])

    return m, annotations


def dive_group_split(
    df: pd.DataFrame,
    uuid_to_dive: Dict[str, str],
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split samples by dive_id groups (no dive appears in both train and val).
    """
    df = df.copy()
    df["dive_id"] = df["id"].map(uuid_to_dive)

    

    # Drop samples without dive_id (cannot group)
    missing = df["dive_id"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} samples have no dive_id (missing in COCO json). Dropping them for dive split.")
        df = df.dropna(subset=["dive_id"]).reset_index(drop=True)

    groups = df["dive_id"].astype(str).values
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    idx = list(range(len(df)))
    train_idx, val_idx = next(splitter.split(idx, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # sanity check: no overlap
    inter = set(train_df["dive_id"]) & set(val_df["dive_id"])
    if inter:
        raise RuntimeError(f"Dive leakage detected! Overlapping dive_ids: {list(inter)[:10]}")
    

    return train_df, val_df