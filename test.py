import os
import sys
import yaml
import torch

from src.dataset.preprocessing import prepare_dataloaders
from src.model.train_model import Trainer
from src.network.resnet_baseline import ResNetBaseline
from src.network.vit_baseline import ViTBaseline


def load_yaml_config(general_yaml_path: str, model_yaml_path: str) -> tuple[dict, dict]:
    """Load YAML configs for general settings and model settings."""
    with open(general_yaml_path, "r") as f:
        general_cfg = yaml.safe_load(f)
    with open(model_yaml_path, "r") as f:
        model_cfg = yaml.safe_load(f)
    return general_cfg, model_cfg


def extract_state_dict(ckpt_obj: object) -> dict:
    """
    Extract a PyTorch state_dict from common checkpoint formats.
    Supported:
      - torch.save(model.state_dict())
      - torch.save({"state_dict": ...})
      - torch.save({"model_state_dict": ...})
    """
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj:
            return ckpt_obj["model_state_dict"]
        # Fallback: assume the dict itself is a state_dict-like object
        return ckpt_obj
    # Rare fallback: some people save the state_dict directly
    return ckpt_obj


def main() -> None:
    """
    Run inference on test_loader and export a submission-like CSV:
      image_id, categories, osd

    Usage:
      python test.py resnet path/to/model.pth
      python test.py vit_b_16 path/to/model.pth
    """
    if len(sys.argv) != 3:
        print("Usage: python test.py [resnet|vit_b_16] path/to/model.pth")
        sys.exit(1)

    arch = sys.argv[1].strip()
    ckpt_path = sys.argv[2].strip()

    # Require a .pth checkpoint as requested
    if not ckpt_path.lower().endswith(".pth"):
        raise ValueError(f"Expected a .pth checkpoint file, got: {ckpt_path}")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nCWD: {os.getcwd()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pick the model YAML matching the architecture
    if arch == "resnet":
        model_yaml_path = "./config/resnet_baseline.yaml"
    elif arch == "vit_b_16":
        model_yaml_path = "./config/vit_b_16_baseline.yaml"
    else:
        raise ValueError("arch must be one of: resnet, vit_b_16")

    general_cfg, model_cfg = load_yaml_config("./config/general.yaml", model_yaml_path)

    # Build dataloaders (new2orig is used to map predicted class indices back to original IDs)
    train_loader, val_loader, test_loader, cat_map, n_existing, new2orig = prepare_dataloaders(general_cfg)

    # Build the model (the architecture must match training)
    if arch == "resnet":
        model = ResNetBaseline(
            num_classes=n_existing,
            pretrained=False,
            layer_size=model_cfg.get("layer_size", None),
        )
    else:
        model = ViTBaseline(
            num_classes=len(cat_map.keys()),
            pretrained=False,
            layer_size=model_cfg.get("layer_size", None),
        )

    # Trainer is used only for tau calibration + the evaluate_test() export logic
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        general_cfg=general_cfg,
        model_cfg=model_cfg,
        optimizer=model_cfg.get("optimizer", "Adam"),
        loss_fn=model_cfg.get("loss_fn", "BCEWithLogits"),
        device=device,
        pos_weight_tensor=None,  # not needed for inference-only script
        n_classes=n_existing,
    )

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt_obj = torch.load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(ckpt_obj)

    missing, unexpected = trainer.model.load_state_dict(state_dict, strict=False)
    print("[INFO] load_state_dict done.")
    if missing:
        print("  missing keys (first 20):", missing[:20])
    if unexpected:
        print("  unexpected keys (first 20):", unexpected[:20])

    trainer.model.to(device).eval()

    # Calibrate entropy threshold tau on validation set (never calibrate on test)
    trainer.calibrate_entropy_tau()

    # Run test inference and export CSV
    out_csv = "evaluation/preds/test_entropy.csv"
    pred_df, results = trainer.evaluate_test(
        test_loader,
        save_csv_path=out_csv,
        new2orig=new2orig,
        compute_metrics_if_gt=True,  # set False for official test without GT
        k=20,
    )

    print(pred_df.head())
    if results:
        print("[INFO] Metrics:", results)
    print(f"[INFO] Saved predictions to {out_csv}")


if __name__ == "__main__":
    main()
