import json
import pandas as pd
from pathlib import Path

def load_category_map(category_key_csv: str) -> dict:
    """
    Build mapping: concept_name -> category_id (float-like in output)
    """
    df = pd.read_csv(category_key_csv)
    # normalize names to avoid small mismatches
    df["name_norm"] = df["name"].astype(str).str.strip()
    # if duplicates exist, keep first
    mapping = dict(zip(df["name_norm"], df["id"]))
    return mapping

def normalize_concept(name: str) -> str:
    return str(name).strip()

def main(
    merged_json_path: str,
    category_key_csv: str,
    out_csv_path: str,
    out_missing_csv: str = "missing_concepts.csv",
):
    merged = json.loads(Path(merged_json_path).read_text(encoding="utf-8"))
    images = merged["images"]

    concept_to_id = load_category_map(category_key_csv)

    rows = []
    missing_rows = []  # for concepts not found in category_key

    for img in images:
        uuid = img.get("uuid")
        bboxes = img.get("boundingBoxes") or []
        concepts = []

        for b in bboxes:
            c = b.get("concept")
            if c is None:
                continue
            concepts.append(normalize_concept(c))

        # unique concepts per image
        concepts = sorted(set(concepts))

        cat_ids = []
        for c in concepts:
            if c in concept_to_id:
                cat_ids.append(float(concept_to_id[c]))  # match your sample formatting
            else:
                missing_rows.append({"uuid": uuid, "missing_concept": c})

        # sort ids for stability
        cat_ids = sorted(set(cat_ids))

        # stringify like "[1.0, 9.0, 11.0]"
        rows.append({
            "id": uuid,
            "categories": "[" + ", ".join(f"{x:.1f}" for x in cat_ids) + "]"
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv_path, index=False)

    if missing_rows:
        miss_df = pd.DataFrame(missing_rows)
        miss_df.to_csv(out_missing_csv, index=False)
        print(f"[WARN] {len(missing_rows)} concept mentions not found in category_key. Saved -> {out_missing_csv}")

    print(f"[DONE] wrote -> {out_csv_path}  (rows={len(out_df)})")

if __name__ == "__main__":
    main(
        merged_json_path="output_eval.json",
        category_key_csv="category_key.csv",
        out_csv_path="eval_label.csv",
        out_missing_csv="out_of_sample.csv",
    )
