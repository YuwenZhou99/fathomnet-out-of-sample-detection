from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, AdamW, lr_scheduler
import matplotlib.pyplot as plt
import os
import torch
from tqdm.auto import tqdm
from src.model.earlystopper import EarlyStopper
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, auc
import warnings
from sklearn.exceptions import UndefinedMetricWarning

import logging
import pandas as pd
from evaluation.fathomnet_metric import score
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


# MAP@K (multi-label) calculate
def map_at_k_multi_label(y_true_sets, y_pred_lists, k=20):
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
    def energy_score(logits, T=1.0):
        return -torch.logsumexp(logits / T, dim=1)

    @staticmethod
    def energy_to_osd(energy, tau, alpha=1.0):
        return torch.sigmoid(alpha * (energy - tau))

    @staticmethod
    def sigmoid_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ent = -(p * (p.clamp_min(eps).log()) + (1 - p) * ((1 - p).clamp_min(eps).log()))
        return ent.mean(dim=1)

    def calibrate_entropy_tau(self) -> float:
        if self.val_loader is None:
            raise RuntimeError("val_loader is None; cannot calibrate entropy tau.")

        self.model.eval()
        ent_list = []

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Calibrating entropy tau", unit="batch"):
                images = images.to(self.device)
                logits = self.model(images)
                ent = self.sigmoid_entropy(logits).detach().cpu()
                ent_list.append(ent)

        ent_all = torch.cat(ent_list, dim=0)
        tau = float(torch.quantile(ent_all, self.entropy_quantile))
        self.entropy_tau = tau
        print(f"[INFO] Calibrated entropy tau={tau:.6f} (quantile={self.entropy_quantile}, alpha={self.entropy_alpha})")
        return tau

    # eval test
    def evaluate_test(
        self,
        test_loader,
        save_csv_path: str | None = None,
        new2orig: dict | None = None,
        compute_metrics_if_gt: bool = False,
        k: int = 20,
    ):
        """
        Produces:
          - pred_df: image_id, categories, osd
          - results: optional metrics if GT exists locally (AUCROC, MAP@20)
        categories format:
          "c1 c2 ... c20, osd_prob"
        """
        if self.entropy_tau is None:
            if self.val_loader is None:
                raise RuntimeError("entropy_tau is None and val_loader is None.")
            self.calibrate_entropy_tau()

        self.model.eval()
        rows = []

        # optional GT metrics buffers
        osd_scores, osd_targets = [], []
        y_true_sets, y_pred_lists = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test eval", unit="batch"):
                images = batch[0].to(self.device)

                # get image_id
                # try find list/tuple of ids
                image_id = None
                for j in range(1, len(batch)):
                    bj = batch[j]
                    if isinstance(bj, (list, tuple)) and len(bj) == images.size(0) and isinstance(bj[0], (str, int)):
                        image_id = bj
                        break
                if image_id is None:
                    image_id = batch[-1]
                if torch.is_tensor(image_id):
                    image_id = image_id.detach().cpu().tolist()

                # optional GT for local evaluation
                targets = None
                osd_target = None
                if compute_metrics_if_gt:
                    for j in range(1, len(batch)):
                        bj = batch[j]
                        if torch.is_tensor(bj) and bj.ndim == 2 and bj.size(0) == images.size(0):
                            targets = bj
                            break
                    for j in range(1, len(batch)):
                        bj = batch[j]
                        if torch.is_tensor(bj) and bj.size(0) == images.size(0) and bj.ndim in (1, 2) and bj.numel() == images.size(0):
                            osd_target = bj.view(-1)
                            break

                logits = self.model(images)
                probs = torch.sigmoid(logits)

                # MAP@20 need ranking
                topk_probs, topk_idx = torch.topk(probs, k, dim=1)  # (B,k)

                # OSD
                ent = self.sigmoid_entropy(logits)  # (B,)
                tau = float(self.entropy_tau)
                osd_prob = torch.sigmoid(self.entropy_alpha * (ent - tau))  # (B,)

                B = probs.size(0)
                for i in range(B):
                    pred_ids = topk_idx[i].detach().cpu().tolist()
                    if new2orig is not None:
                        pred_ids = [int(new2orig[int(x)]) for x in pred_ids]

                    osd_i = osd_prob[i].detach().cpu().item()

                    cats_str = " ".join(str(x) for x in pred_ids)
                    rows.append({
                        "image_id": image_id[i],
                        "categories": cats_str + f", {osd_i}",
                        "osd": osd_i,
                    })

                    if compute_metrics_if_gt and (targets is not None):
                        gt_idx = torch.where(targets[i].detach().cpu() == 1)[0].tolist()
                        if new2orig is not None:
                            t_idx = [int(new2orig[int(x)]) if int(x) in new2orig else int(x) for x in gt_idx]
                        y_true_sets.append(set(gt_idx))
                        y_pred_lists.append(pred_ids)

                    if compute_metrics_if_gt and (osd_target is not None):
                        osd_targets.append(float(osd_target[i].detach().cpu().item()))
                        osd_scores.append(float(osd_i))

        pred_df = pd.DataFrame(rows)

        if save_csv_path is not None:
            os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
            pred_df.to_csv(save_csv_path, index=False)
            print(f"[INFO] Saved test predictions -> {save_csv_path}")

        results = {}
        if compute_metrics_if_gt and len(osd_targets) > 0 and len(set(osd_targets)) > 1:
            results["osd_aucroc"] = roc_auc_score(osd_targets, osd_scores)
        if compute_metrics_if_gt and len(y_true_sets) > 0:
            results["map@20"] = map_at_k_multi_label(y_true_sets, y_pred_lists, k=20)

        if results:
            print("[TEST METRICS]", results)

        return pred_df, results


    @staticmethod
    def count_pos_simple(loader, num_classes):
        pos = torch.zeros(num_classes)
        n = 0
        for images, targets in loader:
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
                for images, targets in pbar:
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
                    for _, (images, targets) in enumerate(self.val_loader):
                        images = images.to(self.device)
                        targets = targets.to(self.device)
                        logits = self.model.forward(images)
                        probs = torch.sigmoid(logits)

                        top_k_probs, top_k_classes = torch.topk(probs, self.general_cfg['top_k'])
                        mask_k = top_k_probs >= threshold

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

                    thresholds = np.linspace(0.002, 0.30, 20)
                    best = (0, 0.5)
                    for t in thresholds:
                        preds = (probs >= t).astype(int)
                        f1 = f1_score(all_targets[:, valid], preds[:, valid], average="micro", zero_division=0)
                        if f1 > best[0]:
                            best = (f1, t)
                    print("best micro-F1:", best[0], "at threshold:", best[1])

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

