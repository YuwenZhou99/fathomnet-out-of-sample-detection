from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, AdamW, lr_scheduler
import matplotlib.pyplot as plt
import os
import torch
from tqdm.auto import tqdm
from src.model.earlystopper import EarlyStopper
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning

import warnings
import logging
from sklearn.exceptions import UndefinedMetricWarning
import pandas as pd
from evaluation.fathomnet_metric import score
import numpy as np

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


class Trainer:
    def __init__(self, model, train_loader, val_loader, general_cfg, model_cfg, optimizer, loss_fn, device, pos_weight_tensor=None,):
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
        self.optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=self.wd) if optimizer == 'AdamW' else Adam(trainable_params, lr=self.lr, weight_decay=self.wd)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=model_cfg.get('lr_step_size', 10), gamma=model_cfg.get('lr_gamma', 0.1))
        self.loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight_tensor) if loss_fn == 'BCEWithLogits' else None
        # can also experiment with label smooting
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

        # Model check
        print(f'[INFO] Model architecture: {type(self.model)}')

        print(f'[INFO] Trainer initialized with {self.n_training_examples} training examples and {self.n_validation_examples} validation examples.')


    def freeze_backbone(self):
        '''
        freezing backbone of model, so only head/fc is trainable
        '''
        # freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # unfreeze common classifier heads
        if hasattr(self.model.model, "fc"):  # ResNet-style
            for p in self.model.model.fc.parameters():
                p.requires_grad = True

        if hasattr(self.model.model, "heads"):  # ViT-style
            for p in self.model.model.heads.parameters():
                p.requires_grad = True


    def unfreeze_backbone(self, epoch):
        for p in self.model.parameters():
            p.requires_grad = True

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr * 0.1,        # usually lower LR after unfreeze
            weight_decay=self.wd,
        )

        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.model_cfg.get("lr_step_size", 10),
            gamma=self.model_cfg.get("lr_gamma", 0.1),
        )

        self.freeze = False
        print(f"[INFO] Unfroze model backbone at epoch {epoch+1}.")


    @staticmethod
    def smooth_labels(targets, epsilon=0.05):
        '''
        smoothing labels for robustness
        
        :param targets: labels to be smoothened
        :param epsilon: amount of smoothing
        '''
        # for out-of-sample detection, might not be useful
        return targets * (1 - epsilon) + 0.5 * epsilon
    

    @staticmethod
    def energy_score(logits, T=1.0):
        '''
        Calculating energy score to implicitly determine osd
        
        :param logits: logits predicted by model
        :param T: hyperparameter to increase/decrease confidence
        '''
        # Use stable energy score calculation
        return -torch.logsumexp(logits/T, dim=1)
    
    
    @staticmethod
    def energy_to_osd(energy, tau, alpha=1.0):
        '''
        normalzing osd score
        
        :param energy: Energy calculated with energy score
        :param tau: to be determined from data
        :param alpha: Hyperparameter to increase/decrease confidence
        '''
        return torch.sigmoid(alpha * (energy - tau))
    
    @staticmethod
    def prepare_dict_for_fathom_df(image_id, targets, top_k, osd_target, osd_prob, mask_k):
        rows = []
        row_targets = []
        for i in range(targets.shape[0]):
            masked_top_k = top_k[i][mask_k[i]]
            row = {
                "image_id": image_id[i],
                "categories": " ".join([str(k) for k in masked_top_k.tolist()]) + f', {str(osd_prob[i])}',
                "osd": float(osd_prob[i])
            }
            row_target = {
                "image_id": image_id[i],
                "categories": " ".join([str(idx.item()) for idx in torch.where(targets[i] == 1)[0]]),
                "osd": float(osd_target[i])
            }
            rows.append(row)
            row_targets.append(row_target)

        return rows, row_targets


    def train(self):
        '''
        Train model with hyperparameters set in trainer object
        '''
        epochs = self.model_cfg['num_epochs']
        for epoch in range(epochs):
            # Unfreeze backbone if specified
            if self.freeze and self.unfreeze_epoch is not None and epoch == self.unfreeze_epoch:
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
                leave=False,   # set True if you want to keep each epoch bar
                )
                for images, targets, _ in pbar:   
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    if self.smoothing_epsilon is not None:
                        targets = self.smooth_labels(targets, self.smoothing_epsilon)
                    self.optimizer.zero_grad()
                    logits = self.model(images)
                    # compute loss, backpropagation, optimizer
                    loss = self.loss_fn(logits, targets)
                    self.batch_train_losses.append(loss.item())
                    running_loss += loss.item() * images.size(0)
                    loss.backward()
                    self.optimizer.step()

                    # update bar text
                    pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(running_loss/self.n_training_examples):.4f}")

            self.scheduler.step()


            epoch_avg_loss = running_loss / self.n_training_examples
            self.train_losses.append(epoch_avg_loss)
            print(f"Epoch {epoch+1} - Training loss: {epoch_avg_loss}")

            if self.save_model:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"{self.model_cfg['model_name']}_{epoch+1}.pth"))

            # Validation loop can be added here
            if self.val_loader is not None:
                self.model.eval()
                all_targets = []
                all_logits = []
                rows = []
                row_targets = []

                with torch.no_grad():
                    for _, (images, targets, image_id, osd_target) in enumerate(self.val_loader):
                        threshold = self.general_cfg['threshold']

                        images = images.to(self.device)
                        targets = targets.to(self.device)
                        logits = self.model.forward(images)
                        probs = torch.softmax(logits, dim=1)
                        mask = (logits >= threshold)
                        thresholded_predictions = probs[mask]
                        

                        # top_k might have to filtered on threshold because it now always outputs 20 predictions (for example just thresholding probability >=0.5)
                        top_k_probs, top_k_classes = torch.topk(probs, self.general_cfg['top_k'])
                        mask_k = top_k_probs >= threshold

                        # this has to be checked
                        energy = self.energy_score(logits)
                        tau = torch.quantile(energy, 0.95)
                        osd_prob = self.energy_to_osd(energy, tau=tau)
                        
                        # for future fathomscore calculation
                        for i in range(targets.shape[0]):
                            row, row_target = self.prepare_dict_for_fathom_df(image_id, targets, top_k_classes, osd_target, osd_prob, mask_k)
                            rows.extend(row)
                            row_targets.extend(row_target)

                        val_loss = self.loss_fn(logits, targets)
                        self.batch_val_losses.append(val_loss.item())
                        running_val_loss += val_loss.item() * images.size(0)
                        all_targets.append(targets.cpu())
                        all_logits.append(logits.cpu())

                # creating dataframes from lists for fathomscore calculation
                target_df = pd.DataFrame(row_targets)
                predictions_df = pd.DataFrame(rows)

                epoch_avg_val_loss = running_val_loss / self.n_validation_examples
                self.val_losses.append(epoch_avg_val_loss)
                print(f"Epoch {epoch+1} - Validation loss: {epoch_avg_val_loss}")
                # evaluate metrics
                all_targets = torch.cat(all_targets, dim=0).numpy()
                probs = torch.sigmoid(torch.cat(all_logits, dim=0)).numpy()
                pos = all_targets.sum(axis=0)
                valid = (pos > 0) & (pos < all_targets.shape[0])

                if valid.sum() == 0:
                    print("ROC-AUC undefined: no class has both positive and negative samples in validation.")
                else:
                    roc_auc = roc_auc_score(all_targets[:, valid], probs[:, valid], average="weighted")
                    avg_precision = average_precision_score(all_targets[:, valid], probs[:, valid], average="weighted")
                    fathomnet_score = score(
                        target_df,
                        predictions_df,
                        'image_id',
                        'osd',
                        20
                    )
                    print(f'fathom score: {fathomnet_score}')

                    preds = (probs >= 0.5).astype(int)
                    f1 = f1_score(all_targets[:, valid], preds[:, valid], average="weighted", zero_division=0)

                    print(f"valid labels: {valid.sum()}/{all_targets.shape[1]}")
                    print(f"ROC AUC: {roc_auc:.4f}, AP: {avg_precision:.4f}, F1: {f1:.4f}")

                # Early stopping check
                if self.early_stopper.early_stop(epoch_avg_val_loss):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss (per batch)')
        
        if len(self.val_losses) > 0:
            plt.plot(self.val_losses, label='Validation Loss (per batch)')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Batch')
        plt.legend()

        filename = get_next_filename("evaluation/loss_plots/plot")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

