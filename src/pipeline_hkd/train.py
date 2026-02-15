"""
Training script with CLI, balanced epoch sampling, mixup, label smoothing,
freeze/unfreeze fine-tuning, SWA, early stopping, checkpointing, and CSV logging.

Usage:
    cd src
    python -m pipeline_hkd.train --model efficientnet --epochs 50
    python -m pipeline_hkd.train --model resnet50 --epochs 50
    python -m pipeline_hkd.train --model densenet --epochs 50
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)

from .config import Config
from .dataset import get_dataloaders
from .models import get_model
from .utils import (
    set_seed, get_device, save_checkpoint,
    LabelSmoothingBCELoss, mixup_data, mixup_criterion,
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation and compute metrics."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Predictions
    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs >= 0.5).astype(int)
    labels_np = all_labels.numpy().astype(int)

    # Metrics
    avg_loss = total_loss / max(n_batches, 1)
    precision = precision_score(labels_np, preds, zero_division=0)
    recall = recall_score(labels_np, preds, zero_division=0)
    f1 = f1_score(labels_np, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels_np, probs)
    except ValueError:
        auc = 0.0

    return {
        "val_loss": avg_loss,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "val_roc_auc": auc,
    }


# ---------------------------------------------------------------------------
# Freeze / Unfreeze Backbone
# ---------------------------------------------------------------------------

def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name:
            param.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone frozen. Trainable params: {n_trainable:,}")


def unfreeze_backbone(model):
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone unfrozen. Trainable params: {n_trainable:,}")


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(cfg: Config):
    """Main training function."""

    set_seed(cfg.seed)
    device = get_device()
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)

    # --- Data ---
    print("\n=== Loading Data ===")
    train_loader, val_loader, _, train_sampler = get_dataloaders(cfg.data)

    # --- Model ---
    print("\n=== Building Model ===")
    model = get_model(cfg.model.name, cfg.model.pretrained, cfg.model.dropout)
    model = model.to(device)

    # --- Loss, Optimizer, Scheduler ---
    criterion = LabelSmoothingBCELoss(smoothing=cfg.train.label_smoothing)
    # For validation, use plain BCE (no smoothing)
    val_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.train.scheduler_T0,
        T_mult=cfg.train.scheduler_Tmult,
    )

    # --- SWA setup ---
    swa_start = cfg.train.epochs + cfg.train.swa_start_epoch  # e.g. 50 + (-5) = 45
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=cfg.train.swa_lr)

    # --- Freeze backbone initially ---
    print("\n=== Phase 1: Frozen Backbone ===")
    freeze_backbone(model)

    # --- CSV log ---
    log_path = cfg.paths.training_log
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    csv_columns = [
        "epoch", "train_loss", "val_loss",
        "val_precision", "val_recall", "val_f1", "val_roc_auc",
    ]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

    # --- Training ---
    best_auc = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(
        cfg.paths.checkpoint_dir, f"best_{cfg.model.name}.pt"
    )

    print(f"\n=== Training for {cfg.train.epochs} epochs ===\n")

    for epoch in range(1, cfg.train.epochs + 1):
        epoch_start = time.time()

        # Unfreeze backbone after freeze_epochs
        if epoch == cfg.train.freeze_epochs + 1:
            print("\n=== Phase 2: Full Fine-Tuning ===")
            unfreeze_backbone(model)
            # Lower LR for backbone
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.train.lr * cfg.train.unfreeze_lr_factor

        # Re-sample majority class for this epoch
        train_sampler.set_epoch(epoch)

        # --- Train one epoch ---
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Mixup
            if cfg.train.mixup_alpha > 0 and epoch > cfg.train.freeze_epochs:
                images, y_a, y_b, lam = mixup_data(
                    images, labels, cfg.train.mixup_alpha
                )
                logits = model(images)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = total_train_loss / max(n_train_batches, 1)

        # --- Scheduler step ---
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # --- Validate ---
        val_metrics = validate(model, val_loader, val_criterion, device)
        elapsed = time.time() - epoch_start

        # --- Log ---
        row = {
            "epoch": epoch,
            "train_loss": f"{avg_train_loss:.6f}",
            "val_loss": f"{val_metrics['val_loss']:.6f}",
            "val_precision": f"{val_metrics['val_precision']:.4f}",
            "val_recall": f"{val_metrics['val_recall']:.4f}",
            "val_f1": f"{val_metrics['val_f1']:.4f}",
            "val_roc_auc": f"{val_metrics['val_roc_auc']:.4f}",
        }
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writerow(row)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{cfg.train.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"F1={val_metrics['val_f1']:.4f} | "
            f"AUC={val_metrics['val_roc_auc']:.4f} | "
            f"lr={lr_now:.2e} | "
            f"{elapsed:.1f}s"
        )

        # --- Early stopping & checkpointing ---
        if val_metrics["val_roc_auc"] > best_auc:
            best_auc = val_metrics["val_roc_auc"]
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_auc, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={cfg.train.patience})")
                break

    # --- Finalize SWA ---
    if epoch >= swa_start:
        print("\n=== Updating SWA Batch Normalization ===")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_path = os.path.join(cfg.paths.checkpoint_dir, f"swa_{cfg.model.name}.pt")
        torch.save(swa_model.state_dict(), swa_path)
        print(f"  SWA model saved → {swa_path}")

    print(f"\n=== Training Complete ===")
    print(f"  Best val AUC: {best_auc:.4f}")
    print(f"  Best checkpoint: {checkpoint_path}")
    print(f"  Training log: {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train thyroid ultrasound classifier")
    parser.add_argument("--model", type=str, default="efficientnet",
                        choices=["efficientnet", "resnet50", "densenet"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.model.name = args.model
    cfg.model.pretrained = not args.no_pretrained
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.mixup_alpha = args.mixup_alpha
    cfg.data.batch_size = args.batch_size
    cfg.seed = args.seed

    train(cfg)


if __name__ == "__main__":
    main()
