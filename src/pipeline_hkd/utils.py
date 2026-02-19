"""
Utility functions: seed, device, checkpointing, label-smoothed loss.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def save_checkpoint(model, optimizer, epoch, val_auc, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_auc": val_auc,
    }, path)
    print(f"  Checkpoint saved → {path} (val_auc={val_auc:.4f})")


def load_checkpoint(model, path, device, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  Loaded checkpoint from {path} (epoch={checkpoint['epoch']}, val_auc={checkpoint['val_auc']:.4f})")
    return checkpoint


class LabelSmoothingBCELoss(nn.Module):
    """Binary cross-entropy with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Smooth: 0 -> smoothing/2, 1 -> 1 - smoothing/2
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets_smooth)


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to a batch."""
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup-augmented batch."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
