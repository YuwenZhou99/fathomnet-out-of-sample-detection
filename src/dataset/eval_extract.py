import argparse
import json
import os
import time
from typing import Any, Dict, List

# pip install fathomnet-py
from fathomnet.api import images as fn_images


def stem_uuid(file_name: str) -> str:
    """'xxxx.png' -> 'xxxx'"""
    base = os.path.basename(file_name)
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def to_jsonable(obj: Any) -> Any:
    """
    Convert fathomnet-py DTOs (and nested objects) to JSON-serializable structures.
    Tries common patterns: model_dump (pydantic v2), dict (pydantic v1),
    to_dict, __dict__.
    """
    if obj is None:
        return None

    # primitives
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # lists / tuples
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]

    # dict
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # datetime-like
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass

    # pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return to_jsonable(obj.model_dump())
        except Exception:
            pass

    # pydantic v1
    if hasattr(obj, "dict"):
        try:
            return to_jsonable(obj.dict())
        except Exception:
            pass

    # custom to_dict
    if hasattr(obj, "to_dict"):
        try:
            return to_jsonable(obj.to_dict())
        except Exception:
            pass

    # fallback __dict__
    if hasattr(obj, "__dict__"):
        try:
            # drop private attrs
            d = {k: v for k, v in obj.__dict__.items() if not str(k).startswith("_")}
            return to_jsonable(d)
        except Exception:
            pass

    # last resort: stringify
    return str(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", required=True, help="Path to eval.json (COCO-style, with images list)")
    ap.add_argument("--out_json", required=True, help="Output JSON path (merged ImageDTO list)")
    ap.add_argument("--sleep", type=float, default=0.15, help="Sleep between API calls (seconds)")
    ap.add_argument("--max_images", type=int, default=-1, help="Debug: limit number of images")
    args = ap.parse_args()

    with open(args.eval_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_list: List[Dict[str, Any]] = coco.get("images", [])
    if args.max_images > 0:
        images_list = images_list[: args.max_images]

    merged: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for i, img in enumerate(images_list, start=1):
        uuid = stem_uuid(img["file_name"])

        try:
            dto = fn_images.find_by_uuid(uuid)  # AImageDTO
            dto_json = to_jsonable(dto)

            # Ensure uuid exists at top-level (some versions might name it 'uuid' already)
            if isinstance(dto_json, dict) and "uuid" not in dto_json:
                dto_json["uuid"] = uuid

            merged.append(dto_json)

        except Exception as e:
            missing.append({"uuid": uuid, "reason": str(e)})

        if args.sleep > 0:
            time.sleep(args.sleep)

        if i % 50 == 0:
            print(f"[INFO] processed {i}/{len(images_list)}; ok={len(merged)} missing={len(missing)}")

    out = {
        "source_eval_json": os.path.abspath(args.eval_json),
        "num_images": len(images_list),
        "images": merged,
        "missing": missing,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote {args.out_json}")
    print(f"       ok={len(merged)} missing={len(missing)}")


if __name__ == "__main__":
    main()
