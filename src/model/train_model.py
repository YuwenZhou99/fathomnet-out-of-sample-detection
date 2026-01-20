from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, AdamW

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
                print(batch_idx, images.shape, targets.shape)
                logits = self.model.forward(images)
                print(f'logits shape: {logits.shape}\n targets shape: {targets.shape}')
                # compute loss, backpropagation, optimizer
                loss = self.loss_fn(logits, targets)
                print(loss.item())
                if batch_idx+1 == epochs:
                    last_loss = last_loss / images.shape[0] * self.batch_size
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch+1} - Training loss: {running_loss/len(self.train_loader)} - Last batch loss: {last_loss}")

            # Validation loop can be added here
            if self.val_loader is not None:
                self.validate()
                for batch_idx, (images, targets) in enumerate(self.val_loader):
                    print(batch_idx, images.shape, targets.shape)
                    logits = self.model.forward(images)
                    val_loss = self.loss_fn(logits, targets)
                    if batch_idx+1 == epochs:
                        last_val_loss = last_val_loss / images.shape[0] * self.batch_size
                    running_val_loss += val_loss.item()
                print(f"Epoch {epoch+1} - Validation loss: {running_val_loss/len(self.val_loader)} - Last batch val loss: {last_val_loss}")


