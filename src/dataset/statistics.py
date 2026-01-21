import torch

def compute_pos_weight_tensor(train_loader, device="cpu"):
    # Accumulate positives per class across the whole training set
    pos = None
    n_samples = 0

    for _, targets in train_loader:
        # targets: (B, C) with 0/1 (or bool)
        targets = targets.detach()
        if targets.dtype != torch.float32:
            targets = targets.float()

        if pos is None:
            pos = targets.sum(dim=0)
        else:
            pos += targets.sum(dim=0)

        n_samples += targets.size(0)

    neg = n_samples - pos
    pos_weight = neg / (pos + 1e-8)  # avoid divide-by-zero

    # classes which can be learned (at least one positive sample)
    learned_classes = (pos > 0).float()
    pos_weight = torch.ones_like(pos)
    pos_weight[learned_classes.bool()] = (neg[learned_classes.bool()] / pos[learned_classes.bool()]).clamp(min=1.0, max=100.0)
    print(f'pos_weight: {pos_weight}')

    return pos_weight.to(device)
