from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, AdamW
import matplotlib.pyplot as plt
import os
import torch

def get_next_filename(base_path, ext="png"):
    i = 0
    while True:
        candidate = f"{base_path}_{i}.{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1

class Trainer:
    def __init__(self, model, train_loader, val_loader, general_cfg, model_cfg, optimizer, loss_fn, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.general_cfg = general_cfg
        self.model_cfg = model_cfg
        self.optimizer = AdamW(self.model.parameters(), lr=model_cfg.get('learning_rate', 0.001)) if optimizer == 'AdamW' else Adam(self.model.parameters(), lr=model_cfg.get('learning_rate', 0.001))
        self.loss_fn = BCEWithLogitsLoss() if loss_fn == 'BCEWithLogits' else loss_fn
        self.device = device
        self.freeze = model_cfg.get('freeze', False)
        self.batch_size = model_cfg.get('batch_size', 32)
        self.train_losses = []
        self.val_losses = []
        self.batch_train_losses = []
        self.batch_val_losses = []
        self.n_training_examples = len(train_loader.dataset)
        self.n_validation_examples = len(val_loader.dataset) if val_loader is not None else 0

        print(f'[INFO] Trainer initialized with {self.n_training_examples} training examples and {self.n_validation_examples} validation examples.')

    def train(self):
        epochs = self.model_cfg['num_epochs']
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            last_loss = 0.0
            running_val_loss = 0.0
            last_val_loss = 0.0
            for batch_idx, (images, targets) in enumerate(self.train_loader):
                print(f'Batch {batch_idx+1}/{len(self.train_loader)}')
                self.optimizer.zero_grad()
                logits = self.model(images)
                # compute loss, backpropagation, optimizer
                loss = self.loss_fn(logits, targets)
                self.batch_train_losses.append(loss.item())
                running_loss += loss.item() * images.size(0)
                loss.backward()
                self.optimizer.step()

            epoch_avg_loss = running_loss / self.n_training_examples
            self.train_losses.append(epoch_avg_loss)
            print(f"Epoch {epoch+1} - Training loss: {epoch_avg_loss}")

            # Validation loop can be added here
            if self.val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (images, targets) in enumerate(self.val_loader):
                        logits = self.model.forward(images)
                        val_loss = self.loss_fn(logits, targets)
                        self.batch_val_losses.append(val_loss.item())
                        running_val_loss += val_loss.item() * images.size(0)
                epoch_avg_val_loss = running_val_loss / self.n_validation_examples
                self.val_losses.append(epoch_avg_val_loss)
                print(f"Epoch {epoch+1} - Validation loss: {epoch_avg_val_loss}")

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss (per batch)')
        
        if len(self.val_losses) > 0:
            plt.plot(self.val_losses, label='Validation Loss (per batch)')

        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Batch')
        plt.legend()

        filename = get_next_filename("evaluation/loss_plots/plot")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
