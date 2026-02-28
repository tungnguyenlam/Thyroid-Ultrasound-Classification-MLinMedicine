"""
src/train.py
Training script for Thyroid Ultrasound Classification.

Supports all models in models/ (res18 | res50 | densenet | efficientnet).
Features:
  - AMP (Automatic Mixed Precision) via torch.cuda.amp
  - Config-driven (config/config.yaml)
  - tqdm epoch progress bar (cleared after each epoch)
  - Per-epoch validation with early stopping
  - Exports training_log.csv -> results/<model_name>/

CLI:
  python src/train.py
  python src/train.py --config config/config.yaml

Can also be imported and called as a function from a Jupyter notebook:
  from src.train import train
  train(config_path="config/config.yaml")
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ── project root on path (needed when launched as a module from notebook)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataloader import get_dataloaders
from src.utils import get_optimizer, get_result_dir, get_scheduler, load_config, set_seed
from models.loss import get_loss_fn


###############################################################################
#  Model factory
###############################################################################

def build_model(cfg: dict) -> nn.Module:
    """
    Return a pretrained torchvision model with its head replaced to output
    ``num_classes`` logits.

    Supported names (``config.model.name``):
        res18       -> ResNet-18
        res50       -> ResNet-50
        densenet    -> DenseNet-121
        efficientnet -> EfficientNet-B0
    """
    import torchvision.models as tvm

    num_classes: int = int(cfg["data"]["num_classes"])
    name: str = cfg["model"]["name"].lower()

    if name == "res18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "res50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "densenet":
        model = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif name == "efficientnet":
        model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model name '{name}'. "
            "Choose from: res18 | res50 | densenet | efficientnet"
        )

    return model


###############################################################################
#  One-epoch helpers
###############################################################################

def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    device: torch.device,
    phase: str,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float, float, float, float]:
    """
    Run one forward (and optionally backward) pass over *loader*.

    Args:
        phase:  "train" or "val"
    Returns:
        (avg_loss, accuracy_pct, precision, recall, f1)
        precision / recall / f1 are weighted averages (0.0 for train phase).
    """
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # Collect predictions & labels for val metrics
    all_preds:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    desc = f"Epoch {epoch}/{total_epochs} [{phase:>5}]"
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, unit="batch")

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            # AMP forward pass
            with autocast(enabled=(scaler is not None)):
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += batch_size

            if not is_train:
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.1f}%",
            )

    pbar.close()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    if not is_train and all_preds:
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(   y_true, y_pred, average="weighted", zero_division=0)
        f1        = f1_score(       y_true, y_pred, average="weighted", zero_division=0)
    else:
        precision = recall = f1 = 0.0

    return avg_loss, accuracy, precision, recall, f1


###############################################################################
#  Main training function
###############################################################################

def train(config_path: str = "config/config.yaml") -> Dict:
    """
    Full training pipeline.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        dict with keys 'best_val_loss', 'best_val_acc', 'best_epoch',
        'log_path', 'checkpoint_path'.
    """
    # ── Config & reproducibility ────────────────────────────────────────────
    cfg = load_config(config_path)
    set_seed(cfg["data"]["seed"])

    train_cfg = cfg["training"]
    epochs    = int(train_cfg["epochs"])
    patience  = int(train_cfg.get("early_stopping_patience", 7))

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[train] Device: {device} | AMP: {use_amp}")

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, _, _ = get_dataloaders(cfg)

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    model_name = cfg["model"]["name"]
    print(f"[train] Model: {model_name} | "
          f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Loss / Optimizer / Scheduler ────────────────────────────────────────
    criterion = get_loss_fn(cfg, device)
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    scaler    = GradScaler() if use_amp else None

    # ── Output dirs ─────────────────────────────────────────────────────────
    result_dir = get_result_dir(cfg)          # results/<model_name>/
    ckpt_dir   = os.path.join("checkpoints", model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path  = os.path.join(result_dir, "training_log.csv")
    ckpt_path = os.path.join(ckpt_dir,   "best.pth")

    # ── CSV log header ───────────────────────────────────────────────────────
    csv_fields = [
        "epoch", "train_loss", "train_acc",
        "val_loss", "val_acc", "val_precision", "val_recall", "val_f1",
        "lr",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_acc  = 0.0
    best_epoch    = 0
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ---- train one epoch
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc, _, _, _ = _run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, "train", epoch, epochs,
        )

        # ---- validate
        val_loss, val_acc, val_prec, val_rec, val_f1 = _run_epoch(
            model, val_loader, criterion, None, None,
            device, "val", epoch, epochs,
        )

        # ---- scheduler step
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        # ---- console summary (single clean line per epoch)
        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
            f"val_prec={val_prec:.4f}  val_rec={val_rec:.4f}  val_f1={val_f1:.4f}  "
            f"lr={current_lr:.2e}  [{elapsed:.0f}s]"
        )

        # ---- CSV append
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow({
                "epoch":         epoch,
                "train_loss":    round(train_loss, 6),
                "train_acc":     round(train_acc,  4),
                "val_loss":      round(val_loss,   6),
                "val_acc":       round(val_acc,    4),
                "val_precision": round(val_prec,   4),
                "val_recall":    round(val_rec,    4),
                "val_f1":        round(val_f1,     4),
                "lr":            current_lr,
            })

        # ---- checkpoint (best val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_epoch    = epoch
            no_improve    = 0
            torch.save({
                "epoch":       epoch,
                "model_name":  model_name,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_loss":    best_val_loss,
                "val_acc":     best_val_acc,
                "cfg":         cfg,
            }, ckpt_path)
        else:
            no_improve += 1

        # ---- early stopping
        if no_improve >= patience:
            print(f"[train] Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break

    print(f"\n[train] Best  val_loss={best_val_loss:.4f}  "
          f"val_acc={best_val_acc:.1f}%  @ epoch {best_epoch}")
    print(f"[train] Checkpoint -> {ckpt_path}")
    print(f"[train] Log        -> {log_path}")

    return {
        "best_val_loss":    best_val_loss,
        "best_val_acc":     best_val_acc,
        "best_epoch":       best_epoch,
        "log_path":         log_path,
        "checkpoint_path":  ckpt_path,
    }


###############################################################################
#  CLI entry-point
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a thyroid classification model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    args = parser.parse_args()
    train(config_path=args.config)
