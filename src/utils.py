"""
src/utils.py
Shared utility helpers for the Thyroid Ultrasound Classification project.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import yaml


###############################################################################
#  Config
###############################################################################

def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML config and return as a nested dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


###############################################################################
#  Reproducibility
###############################################################################

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###############################################################################
#  Optimizers & Schedulers
###############################################################################

def get_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    """Build and return the optimizer specified in config."""
    train_cfg = cfg["training"]
    name = train_cfg.get("optimizer", "adam").lower()
    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: '{name}'. Choose from adam | adamw | sgd.")


def get_scheduler(optimizer: optim.Optimizer, cfg: dict):
    """Build and return the LR scheduler specified in config (or None)."""
    train_cfg = cfg["training"]
    name = train_cfg.get("scheduler", "cosine").lower()
    epochs = int(train_cfg.get("epochs", 30))

    if name == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        step_size = int(train_cfg.get("step_size", 10))
        gamma = float(train_cfg.get("gamma", 0.1))
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: '{name}'. Choose from cosine | step | none.")


###############################################################################
#  Running average meter (for loss / accuracy tracking)
###############################################################################

class AverageMeter:
    """Tracks and computes the running average of a scalar value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def __repr__(self):
        return f"AverageMeter({self.name}): avg={self.avg:.4f}"


###############################################################################
#  Result directory helper
###############################################################################

def get_result_dir(cfg: dict) -> str:
    """Return (and create if necessary) results/<model_name>/."""
    base = cfg["results"].get("base_dir", "results/")
    model_name = cfg["model"]["name"]
    result_dir = os.path.join(base, model_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir
