class Trainer:
    def __init__(self, model, train_loader, val_loader, general_cfg, model_cfg, optimizer, loss_fn, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.general_cfg = general_cfg
        self.model_cfg = model_cfg
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.freeze = model_cfg.get('freeze', False)

    def train(self):
        epochs = self.model_cfg['num_epochs']
        for epoch in range(epochs):
            self.model.train()
            cummulative_loss = 0.0
            for batch_idx, (images, targets) in enumerate(self.train_loader):
                print(batch_idx, images.shape, targets.shape)
                logits = self.model.forward(images)
                print(logits.shape)
                print(targets[0])
                print(targets[1])
                print(targets[2])
                # compute loss, backpropagation, optimizer
                loss = self.loss_fn(logits, targets)
                cummulative_loss += loss.item()
