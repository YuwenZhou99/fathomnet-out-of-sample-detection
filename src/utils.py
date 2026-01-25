import os
import ast
import pandas as pd
from typing import Dict, List


def parse_categories_cell(x: str) -> List[int]:
    vals = ast.literal_eval(x)  # e.g. [1.0, 9.0, 11.0]
    return [int(float(v)) for v in vals]


def load_category_key(path: str) -> Dict[int, Dict[str, str]]:
    df = pd.read_csv(path)
    mapping = {}
    for _, row in df.iterrows():
        mapping[int(row["id"])] = {
            "name": str(row["name"]),
            "supercat": str(row["supercat"])
        }
    return mapping


def filter_existing_images(df: pd.DataFrame, image_dir: str, img_ext: str) -> pd.DataFrame:
    def exists_fn(uuid: str) -> bool:
        return os.path.exists(os.path.join(image_dir, uuid + img_ext))

    mask = df["id"].apply(exists_fn)
    missing = int((~mask).sum())
    if missing > 0:
        print(f"[WARN] {missing} samples missing image files in '{image_dir}'. Dropping them.")
    return df[mask].reset_index(drop=True)
