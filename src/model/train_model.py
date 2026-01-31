from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, AdamW
import matplotlib.pyplot as plt
import os
import torch
from tqdm.auto import tqdm
from src.model.earlystopper import EarlyStopper
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning

import logging
import pandas as pd
import numpy as np
import torch.nn as nn


logging.basicConfig(
    filename="warnings.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_warning(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"{category.__name__}: {message}")

warnings.showwarning = log_warning
warnings.filterwarnings("default", category=UndefinedMetricWarning)


def get_next_filename(base_path, ext="png"):
    i = 0
    while True:
        candidate = f"{base_path}_{i}.{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def map_at_k_multi_label(y_true_sets, y_pred_lists, k=20) -> float:
    """Mean Average Precision @ K for multi-label (ranking) predictions."""
    aps = []
    for true_set, pred in zip(y_true_sets, y_pred_lists):
        pred_k = pred[:k]
        if len(true_set) == 0:
            aps.append(0.0)
            continue

        hit = 0
        ap = 0.0
        for rank, cid in enumerate(pred_k, start=1):
            if cid in true_set:
                hit += 1
                ap += hit / rank
        ap /= min(k, len(true_set))
        aps.append(ap)
    return float(np.mean(aps)) if len(aps) else 0.0


class Trainer:
    def __init__(self, model, train_loader, val_loader, general_cfg, model_cfg,
                 optimizer, loss_fn, device, pos_weight_tensor=None, n_classes=290):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.general_cfg = general_cfg
        self.model_cfg = model_cfg
        self.lr = model_cfg.get('learning_rate', 0.001)
        self.wd = model_cfg.get('weight_decay', 0.0)
        self.freeze = model_cfg.get('freeze', False)

        if self.freeze:
            self.freeze_backbone()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=self.lr) if optimizer == 'AdamW' else Adam(
            trainable_params, lr=self.lr, weight_decay=self.wd
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=1,
            threshold=1e-4,
            min_lr=1e-7
        )

        self.loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight_tensor) if loss_fn == 'BCEWithLogits' else None
        self.smoothing_epsilon = general_cfg.get('smooting_epsilon', 0)
        self.device = device
        self.unfreeze_epoch = model_cfg.get('unfreeze_epoch', None)

        self.batch_size = model_cfg.get('batch_size', 32)
        self.train_losses = []
        self.val_losses = []
        self.batch_train_losses = []
        self.batch_val_losses = []

        self.n_training_examples = len(train_loader.dataset)
        self.n_validation_examples = len(val_loader.dataset) if val_loader is not None else 0

        self.early_stopper = EarlyStopper(
            patience=model_cfg.get('patience', 1),
            min_delta=model_cfg.get('min_delta', 0.0)
        )

        self.save_dir = general_cfg.get('save_dir', 'evaluation/model_checkpoints/')
        self.save_model = general_cfg.get('save_model', True)
        self.n_classes = n_classes

        # Entropy OSD calibration params
        self.entropy_tau = None
        self.entropy_quantile = general_cfg.get("entropy_quantile", 0.95)
        self.entropy_alpha = general_cfg.get("entropy_alpha", 1.0)

        print(f'[INFO] Model architecture: {type(self.model)}')
        print(f'[INFO] Trainer initialized with {self.n_training_examples} training examples and {self.n_validation_examples} validation examples.')
        print(f'lr: {self.lr}, wd: {self.wd}, freeze: {self.freeze}, unfreeze_epoch: {self.unfreeze_epoch}, smooth_ep: {self.smoothing_epsilon}, ', end='')
        print(f'patience: {model_cfg.get("patience", 1)}, criterion: {loss_fn}')

    def freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False

        if hasattr(self.model.model, "fc"):
            for p in self.model.model.fc.parameters():
                p.requires_grad = True

        if hasattr(self.model.model, "heads"):
            for p in self.model.model.heads.parameters():
                p.requires_grad = True

        print('[INFO] Backbone Frozen')

    def unfreeze_backbone(self, epoch):
        for p in self.model.parameters():
            p.requires_grad = True

        decay, no_decay = [], []
        head_decay, head_no_decay = [], []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            is_head = any(k in name for k in ["fc", "classifier", "head", "heads"])
            is_no_decay = (p.ndim == 1) or name.endswith(".bias")

            if is_head:
                (head_no_decay if is_no_decay else head_decay).append(p)
            else:
                (no_decay if is_no_decay else decay).append(p)

        backbone_lr = self.lr * 0.01
        head_lr = self.lr * 0.1

        self.optimizer = AdamW(
            [
                {"params": decay,         "lr": backbone_lr, "weight_decay": 1e-2},
                {"params": no_decay,      "lr": backbone_lr, "weight_decay": 0.0},
                {"params": head_decay,    "lr": head_lr,     "weight_decay": self.wd},
                {"params": head_no_decay, "lr": head_lr,     "weight_decay": 1e-2},
            ]
        )

        self.freeze = False
        print(f"[INFO] Unfroze model backbone at epoch {epoch+1}.")

    @staticmethod
    def smooth_labels(targets, epsilon=0.05):
        return targets * (1 - epsilon) + 0.5 * epsilon

    @staticmethod
    def sigmoid_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Average Bernoulli entropy across classes for multi-label logits."""
        p = torch.sigmoid(logits)
        ent = -(p * (p.clamp_min(eps).log()) + (1 - p) * ((1 - p).clamp_min(eps).log()))
        return ent.mean(dim=1)

    def calibrate_entropy_tau(self) -> float:
        """Calibrate entropy threshold tau on validation set (in-distribution)."""
        if self.val_loader is None:
            raise RuntimeError("val_loader is None; cannot calibrate entropy tau.")

        self.model.eval()
        ent_list = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Calibrating entropy tau", unit="batch"):
                images = batch[0].to(self.device)
                logits = self.model(images)
                ent = self.sigmoid_entropy(logits).detach().cpu()
                ent_list.append(ent)

        ent_all = torch.cat(ent_list, dim=0)
        tau = float(torch.quantile(ent_all, self.entropy_quantile))
        self.entropy_tau = tau
        print(f"[INFO] Calibrated entropy tau={tau:.6f} (quantile={self.entropy_quantile}, alpha={self.entropy_alpha})")
        return tau

    @staticmethod
    def _is_id_container(x, B: int) -> bool:
        """Heuristic: list/tuple of length B with str/int elements."""
        return isinstance(x, (list, tuple)) and len(x) == B and (len(x) == 0 or isinstance(x[0], (str, int)))

    @staticmethod
    def _tensor_unique_small_ints(x: torch.Tensor, max_classes: int = 3) -> bool:
        """Check if tensor has few unique values (e.g., {0,1})."""
        try:
            u = torch.unique(x.detach().cpu())
            return len(u) <= max_classes
        except Exception:
            return False

    def _parse_batch(self, batch, require_gt: bool):
        """
        Robustly parse a batch into:
          images: Tensor(B, ...)
          image_id: list[str|int] or None
          targets: Tensor(B,C) or None
          osd_target: Tensor(B,) or None (values 0/1)
        """
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError(f"Unexpected batch type/len: type={type(batch)} len={len(batch) if hasattr(batch,'__len__') else 'NA'}")

        images = batch[0]
        if not torch.is_tensor(images):
            raise ValueError("batch[0] must be images tensor")
        B = images.size(0)

        image_id = None
        targets = None
        osd_target = None

        # 1) Find image_id: prefer list/tuple of str/int
        for j in range(1, len(batch)):
            if self._is_id_container(batch[j], B):
                image_id = list(batch[j])
                break
        # Tensor IDs (rare): shape (B,) int-like
        if image_id is None:
            for j in range(1, len(batch)):
                bj = batch[j]
                if torch.is_tensor(bj) and bj.ndim == 1 and bj.numel() == B:
                    # Could be ids or could be osd_target; disambiguate later
                    # If dtype is integer, treat as candidate id for now
                    if bj.dtype in (torch.int32, torch.int64):
                        image_id = bj.detach().cpu().tolist()
                        break

        # 2) Find targets: Tensor(B,C) with C>=2
        for j in range(1, len(batch)):
            bj = batch[j]
            if torch.is_tensor(bj) and bj.ndim == 2 and bj.size(0) == B:
                targets = bj
                break

        # 3) Find osd_target: Tensor(B,) or (B,1) with {0,1} values
        for j in range(1, len(batch)):
            bj = batch[j]
            if torch.is_tensor(bj) and bj.numel() == B and bj.ndim in (1, 2):
                cand = bj.view(-1)
                # Prefer tensors that look binary {0,1}
                if self._tensor_unique_small_ints(cand, max_classes=2):
                    # Also avoid picking integer image_id tensor as osd_target if already used as id
                    if image_id is not None and torch.is_tensor(batch[j]) and batch[j].dtype in (torch.int32, torch.int64):
                        # If image_id already came from this tensor, skip as osd_target
                        pass
                    osd_target = cand
                    break

        # Fallback: if image_id is still None, try last element (common pattern)
        if image_id is None:
            last = batch[-1]
            if self._is_id_container(last, B):
                image_id = list(last)
            elif torch.is_tensor(last) and last.ndim == 1 and last.numel() == B:
                image_id = last.detach().cpu().tolist()

        if require_gt:
            if targets is None:
                raise RuntimeError(
                    "GT category targets not found in batch, cannot compute MAP@20. "
                    "Please ensure test_loader returns targets Tensor(B,C)."
                )
            if osd_target is None:
                raise RuntimeError(
                    "OSD ground-truth (0/1) not found in batch, cannot compute AUC-ROC. "
                    "Please ensure test_loader returns osd_target Tensor(B,) or Tensor(B,1)."
                )

        return images, image_id, targets, osd_target

    def evaluate_test(
        self,
        test_loader,
        save_csv_path: str | None = None,
        new2orig: dict | None = None,
        compute_metrics_if_gt: bool = False,
        k: int = 20,
    ):
        """
        Outputs:
          pred_df columns: image_id, categories, osd, (optional) osd_target
          results: {'osd_aucroc':..., 'map@20':...} when GT is available
        categories format:
          "c1 c2 ... cK, osd_prob"
        """
        if self.entropy_tau is None:
            if self.val_loader is None:
                raise RuntimeError("entropy_tau is None and val_loader is None.")
            self.calibrate_entropy_tau()

        self.model.eval()
        rows = []

        osd_scores, osd_targets = [], []
        y_true_sets, y_pred_lists = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test eval", unit="batch"):
                images, image_id, targets, osd_target = self._parse_batch(batch, require_gt=compute_metrics_if_gt)
                images = images.to(self.device)

                logits = self.model(images)
                probs = torch.sigmoid(logits)

                # Top-k ranking for MAP@20
                _, topk_idx = torch.topk(probs, k, dim=1)  # (B,k)

                # Entropy-based OSD score
                ent = self.sigmoid_entropy(logits)  # (B,)
                tau = float(self.entropy_tau)
                osd_prob = torch.sigmoid(self.entropy_alpha * (ent - tau))  # (B,)

                B = probs.size(0)
                for i in range(B):
                    # Predicted class IDs
                    pred_ids = topk_idx[i].detach().cpu().tolist()

                    # Map predictions to original IDs if provided
                    if new2orig is not None:
                        pred_ids_mapped = []
                        for x in pred_ids:
                            x = int(x)
                            pred_ids_mapped.append(int(new2orig[x]) if x in new2orig else x)
                        pred_ids = pred_ids_mapped

                    osd_i = osd_prob[i].detach().cpu().item()

                    # Safe image_id fallback
                    img_id_i = image_id[i] if image_id is not None else f"idx_{len(rows)}"

                    cats_str = " ".join(str(x) for x in pred_ids)
                    row = {
                        "image_id": img_id_i,
                        "categories": cats_str + f", {osd_i}",
                        "osd": float(osd_i),
                    }

                    # If GT exists, attach it and accumulate metrics buffers
                    if compute_metrics_if_gt:
                        # MAP@20 GT categories
                        gt_idx = torch.where(targets[i].detach().cpu() == 1)[0].tolist()

                        # Decide whether GT indices are new-index space or already original ID space:
                        # If targets width matches mapping size, GT likely uses new-indices.
                        gt_is_new_space = (new2orig is not None) and (targets.size(1) == len(new2orig))
                        if gt_is_new_space and new2orig is not None:
                            gt_idx_mapped = []
                            for x in gt_idx:
                                x = int(x)
                                gt_idx_mapped.append(int(new2orig[x]) if x in new2orig else x)
                            gt_idx = gt_idx_mapped

                        y_true_sets.append(set(int(x) for x in gt_idx))
                        y_pred_lists.append([int(x) for x in pred_ids])

                        # AUC-ROC GT OSD
                        osd_t = float(osd_target[i].detach().cpu().item())
                        osd_targets.append(osd_t)
                        osd_scores.append(float(osd_i))
                        row["osd_target"] = osd_t

                    rows.append(row)

        pred_df = pd.DataFrame(rows)

        if save_csv_path is not None:
            os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
            pred_df.to_csv(save_csv_path, index=False)
            print(f"[INFO] Saved test predictions -> {save_csv_path}")

        results = {}
        if compute_metrics_if_gt:
            if len(osd_targets) == 0:
                raise RuntimeError("No osd_targets collected; cannot compute AUC-ROC.")
            if len(set(osd_targets)) <= 1:
                raise RuntimeError(f"AUC-ROC undefined: osd_targets has only one class: {set(osd_targets)}")
            results["osd_aucroc"] = roc_auc_score(osd_targets, osd_scores)

            if len(y_true_sets) == 0:
                raise RuntimeError("No GT category labels collected; cannot compute MAP@20.")
            results["map@20"] = map_at_k_multi_label(y_true_sets, y_pred_lists, k=k)

            print("[TEST METRICS]", results)

        return pred_df, results

    @staticmethod
    def count_pos_simple(loader, num_classes):
        pos = torch.zeros(num_classes)
        n = 0
        for batch in loader:
            images, targets = batch[0], batch[1]
            targets = targets.float()
            pos += targets.sum(dim=0).cpu()
            n += targets.size(0)
        return pos.numpy(), n

    def train(self):
        threshold = self.general_cfg['threshold']
        train_pos, _ = self.count_pos_simple(self.train_loader, self.n_classes)
        epochs = self.model_cfg['num_epochs']
        for epoch in range(epochs):
            if self.freeze and self.unfreeze_epoch is not False and epoch == self.unfreeze_epoch:
                self.unfreeze_backbone(epoch)
            self.model.train()
            running_loss = 0.0
            running_val_loss = 0.0

            if not self.general_cfg['test_eval']:
                pbar = tqdm(
                    self.train_loader,
                    total=len(self.train_loader),
                    desc=f"Epoch {epoch+1}/{epochs}",
                    unit="batch",
                    leave=False,
                )
                for batch in pbar:
                    images, targets = batch[0], batch[1]
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    if self.smoothing_epsilon is not None:
                        targets = self.smooth_labels(targets, self.smoothing_epsilon)
                    self.optimizer.zero_grad()
                    logits = self.model(images)
                    loss = self.loss_fn(logits, targets)
                    self.batch_train_losses.append(loss.item())
                    running_loss += loss.item() * images.size(0)
                    loss.backward()
                    self.optimizer.step()
                    pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(running_loss/self.n_training_examples):.4f}")

            epoch_avg_loss = running_loss / self.n_training_examples
            self.train_losses.append(epoch_avg_loss)
            print(f"Epoch {epoch+1} - Training loss: {epoch_avg_loss}")

            if self.save_model:
                os.makedirs(self.save_dir, exist_ok=True)
                ckpt_path = os.path.join(self.save_dir, f"{self.model_cfg['model_name']}_{epoch + 1}.pth")
                torch.save(self.model.state_dict(), ckpt_path)

            if self.val_loader is not None:
                self.model.eval()
                all_targets = []
                all_logits = []

                with torch.no_grad():
                    for batch in self.val_loader:
                        images, targets = batch[0], batch[1]
                        images = images.to(self.device)
                        targets = targets.to(self.device)
                        logits = self.model.forward(images)
                        probs = torch.sigmoid(logits)

                        _top_k_probs, _top_k_classes = torch.topk(probs, self.general_cfg['top_k'])
                        val_loss = self.loss_fn(logits, targets)
                        self.batch_val_losses.append(val_loss.item())
                        running_val_loss += val_loss.item() * images.size(0)
                        all_targets.append(targets.cpu())
                        all_logits.append(logits.cpu())

                epoch_avg_val_loss = running_val_loss / self.n_validation_examples
                self.scheduler.step(epoch_avg_val_loss)
                self.val_losses.append(epoch_avg_val_loss)
                print(f"Epoch {epoch+1} - Validation loss: {epoch_avg_val_loss}")

                all_targets = torch.cat(all_targets, dim=0).numpy()
                probs = torch.sigmoid(torch.cat(all_logits, dim=0)).numpy()
                pos = all_targets.sum(axis=0)
                valid = (pos > 0) & (pos < all_targets.shape[0])
                learnable = valid & (train_pos > 0)
                print("val-valid classes:", valid.sum())
                print("learnable classes:", learnable.sum())

                if valid.sum() == 0:
                    print('undef')
                else:
                    true_pos_rate = all_targets[:, learnable].mean()
                    roc_auc_micro = roc_auc_score(all_targets, probs, average="micro")
                    roc_auc_macro = roc_auc_score(all_targets, probs, average="weighted")
                    preds = (probs >= 0.5).astype(int)
                    pred_pos_rate = (preds == 1)[:, learnable].mean()
                    f1_macro = f1_score(all_targets, preds, average="macro", zero_division=0)
                    f1_micro = f1_score(all_targets, preds, average="micro", zero_division=0)
                    avg_precision = average_precision_score(all_targets[:, valid], probs[:, valid], average="weighted")
                    print(f"valid labels: {valid.sum()}/{all_targets.shape[1]}")
                    print(f"ROC AUC micro: {roc_auc_micro:.4f}, ROC AUC macro: {roc_auc_macro:.4f}, PR AUC: {avg_precision:.4f}, F1_micro: {f1_micro:.4f}, F1_macro: {f1_macro:.4f}")
                    print("true_pos_rate:", true_pos_rate, "pred_pos_rate:", pred_pos_rate)

                if self.early_stopper.early_stop(epoch_avg_val_loss):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')

        if len(self.val_losses) > 0:
            plt.plot(self.val_losses, label='Validation Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Batch')
        plt.legend()

        filename = get_next_filename("evaluation/loss_plots/plot")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
