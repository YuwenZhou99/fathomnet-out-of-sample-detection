import torch
import os

def compute_pos_weight_tensor(train_loader, device="cpu", save_path: str = None, force_compute: bool = False):
    '''
    Computes pos weight for each class to counter class imbalance in train set.
    If `save_path` is provided and the file exists (and `force_compute` is False),
    the tensor will be loaded from disk instead of being recomputed.

    :param train_loader: trainloader for training set
    :param device: target device for the returned tensor
    :param save_path: optional path to save/load the computed tensor
    :param force_compute: if True, always recompute even if save_path exists
    :return: pos_weight tensor representing class imbalances
    '''
    # Try loading from disk first if requested
    if save_path is not None and os.path.exists(save_path) and not force_compute:
        try:
            pos_weight = torch.load(save_path, map_location=device)
            print(f'Loaded pos_weight tensor from {save_path}')
            return pos_weight.to(device)
        except Exception as e:
            print(f'Failed to load pos_weight from {save_path}: {e}. Recomputing...')

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

    # classes which can be learned (at least one positive sample)
    valid = (pos > 0)
    pos_weight = torch.ones_like(pos)
    ratio = (neg[valid] / (pos[valid] + 1e-8)).clamp(min=1.0)  # avoid <1 due to noise
    ratio = ratio.pow(0.5)  # power=0.5 => sqrt weighting
    pos_weight[valid] = ratio.clamp(max=10.0)

    # Save to disk (on CPU) for reuse
    if save_path is not None:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(pos_weight.cpu(), save_path)
            print(f'Saved pos_weight tensor to {save_path}')
        except Exception as e:
            print(f'Failed to save pos_weight to {save_path}: {e}')

    return pos_weight.to(device)
