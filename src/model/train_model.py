from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, AdamW
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
    def __init__(self, model, train_loader, val_loader, general_cfg, model_cfg, optimizer, loss_fn, device, pos_weight_tensor=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.general_cfg = general_cfg
        self.model_cfg = model_cfg
        self.optimizer = AdamW(self.model.parameters(), lr=model_cfg.get('learning_rate', 0.001)) if optimizer == 'AdamW' else Adam(self.model.parameters(), lr=model_cfg.get('learning_rate', 0.001))
        self.loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight_tensor) if loss_fn == 'BCEWithLogits' else loss_fn
        self.device = device
        self.freeze = model_cfg.get('freeze', False)
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

        print(f'[INFO] Trainer initialized with {self.n_training_examples} training examples and {self.n_validation_examples} validation examples.')

    def train(self):
        epochs = self.model_cfg['num_epochs']
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            last_loss = 0.0
            running_val_loss = 0.0
            last_val_loss = 0.0

            pbar = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
            unit="batch",
            leave=False,   # set True if you want to keep each epoch bar
            )
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
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
                with torch.no_grad():
                    for batch_idx, (images, targets) in enumerate(self.val_loader):
                        logits = self.model.forward(images)
                        val_loss = self.loss_fn(logits, targets)
                        self.batch_val_losses.append(val_loss.item())
                        running_val_loss += val_loss.item() * images.size(0)
                        all_targets.append(targets.cpu())
                        all_logits.append(logits.cpu())
                epoch_avg_val_loss = running_val_loss / self.n_validation_examples
                self.val_losses.append(epoch_avg_val_loss)
                print(f"Epoch {epoch+1} - Validation loss: {epoch_avg_val_loss}")
                # evaluate metrics
                all_targets = torch.cat(all_targets, dim=0).numpy()
                all_predictions = torch.sigmoid(torch.cat(all_logits, dim=0)).numpy()
                all_predictions = (all_predictions >= 0.5).astype(int)
                try:
                    #print(f"printing shapes: {all_targets.shape}, {all_predictions.shape}\nAnd Types: {all_targets.dtype}, {all_predictions.dtype}")
                    # Might be an error here because of mismatching dtypes, however precision still gets calculated so might be ok.
                    roc_auc = roc_auc_score(all_targets, all_predictions, average='weighted')
                    avg_precision = average_precision_score(all_targets, all_predictions, average='weighted')
                    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
                    print(f"Epoch {epoch+1} - ROC AUC: {roc_auc:.4f}, Average Precision: {avg_precision:.4f}, F1 Score: {f1:.4f}")
                except ValueError as e:
                    print(f"Could not compute ROC AUC or Average Precision: {e}")

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
