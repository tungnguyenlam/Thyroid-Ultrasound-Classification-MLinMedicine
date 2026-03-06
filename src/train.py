"""
src/train.py
5-Fold Cross-Validation training for Thyroid Ultrasound Classification.

Supports all models in models/ (res18 | res50 | densenet | efficientnet).
Features:
  - Stratified k-fold CV with a fixed held-out test set
  - AMP (Automatic Mixed Precision) via torch.cuda.amp
  - Config-driven (config/config.yaml)
  - TTA (Test-Time Augmentation) at test evaluation
  - Per-fold: training_log.csv, test_metrics.csv, classification_report.txt
  - Cross-fold: cv_summary.csv

CLI:
  python src/train.py
  python src/train.py --config config/config.yaml

Notebook:
  from src.train import train_cv
  cv_results = train_cv(config_path="config/config.yaml")
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
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Project root on path (needed when launched as a module from notebook)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataloader import get_dataloaders, get_cv_fold_loaders, get_tta_transforms
from src.utils import get_optimizer, get_result_dir, get_scheduler, load_config, set_seed
from models.loss import get_loss_fn


#==============================================================================
#  Model factory
#==============================================================================

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


#==============================================================================
#  One-epoch helpers
#==============================================================================

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
) -> Tuple[float, float, float, float, float, Optional[np.ndarray], Optional[np.ndarray]]:
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
            with autocast("cuda", enabled=(scaler is not None)):
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
        y_true = y_pred = None

    return avg_loss, accuracy, precision, recall, f1, y_true, y_pred


###############################################################################
#  TTA evaluation
###############################################################################

def _run_tta_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    img_size: int,
) -> Tuple[float, float, float, float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Test-Time Augmentation evaluation.

    For every batch, runs ``N`` augmented views (from ``get_tta_transforms``),
    averages the softmax probabilities, and takes an argmax for the final
    prediction. Loss is computed on the original (non-augmented) view.

    Args:
        model:    Model in eval mode (weights already loaded).
        loader:   DataLoader whose dataset uses the plain test transform.
        criterion: Loss function (for test_loss reporting).
        device:   Torch device.
        img_size: Spatial resolution from config — passed to get_tta_transforms.

    Returns:
        (avg_loss, accuracy_pct, precision, recall, f1, y_true, y_pred)
    """
    tta_transforms = get_tta_transforms(img_size)   # list of N transforms
    n_tta = len(tta_transforms)

    model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0
    all_preds:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    pbar = tqdm(loader, desc="[TTA test]", leave=False, dynamic_ncols=True, unit="batch")

    with torch.no_grad():
        for images, labels in pbar:
            labels = labels.to(device, non_blocking=True)
            batch_size = labels.size(0)

            # ---- loss on original view (images already transformed by loader)
            images_dev = images.to(device, non_blocking=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits_orig = model(images_dev)
                loss = criterion(logits_orig, labels)
            running_loss += loss.item() * batch_size

            # ---- TTA: re-apply each augmentation to the raw PIL images
            import torchvision.transforms.functional as F_tv
            from PIL import Image as PILImage

            # Collect per-transform softmax probabilities
            probs_sum = torch.zeros(batch_size, logits_orig.size(1), device=device)

            for t_fn in tta_transforms:
                # Convert tensor batch -> PIL -> apply t_fn -> tensor
                batch_tensors = []
                for i in range(batch_size):
                    # Denormalise to [0,1] then to PIL
                    img_t = images[i]  # CPU tensor
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_01 = img_t * std + mean
                    img_01 = img_01.clamp(0, 1)
                    pil_img = F_tv.to_pil_image(img_01)
                    batch_tensors.append(t_fn(pil_img))

                aug_batch = torch.stack(batch_tensors).to(device, non_blocking=True)
                with autocast("cuda", enabled=device.type == "cuda"):
                    logits_aug = model(aug_batch)
                probs_sum += torch.softmax(logits_aug, dim=1)

            preds = probs_sum.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += batch_size

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.1f}%",
            )

    pbar.close()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(   y_true, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(       y_true, y_pred, average="weighted", zero_division=0)

    print(f"[TTA] {n_tta} views averaged | "
          f"test_acc={accuracy:.2f}%  prec={precision:.4f}  rec={recall:.4f}  f1={f1:.4f}")

    return avg_loss, accuracy, precision, recall, f1, y_true, y_pred

#==============================================================================
#  Cross-Validation training
#==============================================================================

def train_cv(config_path: str = "config/config.yaml") -> Dict:
    """
    5-fold (configurable) stratified cross-validation training loop.

    For each fold:
      - Builds fresh model, optimizer, scheduler, scaler.
      - Runs the full early-stopping training loop.
      - Saves checkpoint  -> checkpoints/<model>/fold<k>/best.pth
      - Saves CSV log     -> results/<model>/fold<k>/training_log.csv
      - Runs inference on the shared held-out test set and reports metrics.

    After all folds:
      - Prints mean ± std of val_acc, val_f1, test_acc, test_f1 across folds.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        dict with keys 'fold_results' (list of per-fold dicts) and
        'mean_val_acc', 'mean_val_f1', 'mean_test_acc', 'mean_test_f1'.
    """
    cfg = load_config(config_path)
    set_seed(cfg["data"]["seed"])

    train_cfg  = cfg["training"]
    epochs     = int(train_cfg["epochs"])
    patience   = int(train_cfg.get("early_stopping_patience", 7))
    n_folds    = int(train_cfg.get("n_folds", 5))
    model_name = cfg["model"]["name"]

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"\n[train_cv] Device: {device} | AMP: {use_amp}")
    print(f"[train_cv] Model : {model_name} | Folds: {n_folds}\n")

    fold_results: List[Dict] = []

    for fold_idx in range(n_folds):
        print(f"\n{'='*70}")
        print(f"  FOLD {fold_idx + 1} / {n_folds}")
        print(f"{'='*70}")

        # Per-fold output dirs
        base_result = cfg["results"].get("base_dir", "results/")
        fold_result_dir = os.path.join(base_result, model_name, f"fold{fold_idx + 1}")
        fold_ckpt_dir   = os.path.join("checkpoints", model_name, f"fold{fold_idx + 1}")
        os.makedirs(fold_result_dir, exist_ok=True)
        os.makedirs(fold_ckpt_dir,  exist_ok=True)

        log_path  = os.path.join(fold_result_dir, "training_log.csv")
        ckpt_path = os.path.join(fold_ckpt_dir,   "best.pt")

        # Data
        train_loader, val_loader, test_loader, _ = get_cv_fold_loaders(
            cfg, fold_idx=fold_idx, n_folds=n_folds
        )

        # Fresh model (reset weights each fold)
        set_seed(cfg["data"]["seed"] + fold_idx)   # different init per fold
        model = build_model(cfg).to(device)
        print(f"[fold {fold_idx + 1}] Params: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Loss / Optimizer / Scheduler / Scaler
        criterion = get_loss_fn(cfg, device)
        optimizer = get_optimizer(model, cfg)
        scheduler = get_scheduler(optimizer, cfg)
        scaler    = GradScaler("cuda") if use_amp else None

        # CSV log header
        csv_fields = [
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc", "val_precision", "val_recall", "val_f1",
            "lr",
        ]
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

        # Training loop
        best_val_loss = float("inf")
        best_val_acc  = 0.0
        best_val_f1   = 0.0
        best_epoch    = 0
        no_improve    = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_loss, train_acc, _, _, _, _, _ = _run_epoch(
                model, train_loader, criterion, optimizer, scaler,
                device, "train", epoch, epochs,
            )
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = _run_epoch(
                model, val_loader, criterion, None, None,
                device, "val", epoch, epochs,
            )

            if scheduler is not None:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            elapsed    = time.time() - t0

            print(
                f"[F{fold_idx + 1}] Epoch {epoch:>3}/{epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
                f"val_f1={val_f1:.4f}  lr={current_lr:.2e}  [{elapsed:.0f}s]"
            )

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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc  = val_acc
                best_val_f1   = val_f1
                best_epoch    = epoch
                no_improve    = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"[fold {fold_idx + 1}] Early stopping at epoch {epoch}.")
                break

        # ── Test evaluation (best checkpoint, with optional TTA) ─────────────
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        use_tta  = bool(train_cfg.get("use_tta", True))
        img_size = int(cfg["data"].get("img_size", 512))

        if use_tta:
            print(f"\n[fold {fold_idx + 1}] Running TTA evaluation...")
            test_loss, test_acc, test_prec, test_rec, test_f1, test_y_true, test_y_pred = \
                _run_tta_epoch(model, test_loader, criterion, device, img_size)
        else:
            test_loss, test_acc, test_prec, test_rec, test_f1, test_y_true, test_y_pred = \
                _run_epoch(model, test_loader, criterion, None, None, device, "val", 0, 0)

        print(
            f"\n[fold {fold_idx + 1}] BEST -> epoch {best_epoch}  "
            f"val_acc={best_val_acc:.1f}%  val_f1={best_val_f1:.4f}"
        )
        print(
            f"[fold {fold_idx + 1}] TEST  -> "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.1f}%  "
            f"test_prec={test_prec:.4f}  test_rec={test_rec:.4f}  test_f1={test_f1:.4f}"
        )

        # Save test metrics to a small CSV alongside the training log
        test_log_path = os.path.join(fold_result_dir, "test_metrics.csv")
        test_fields   = ["fold", "best_epoch", "val_acc", "val_f1",
                         "test_loss", "test_acc", "test_precision", "test_recall", "test_f1"]
        with open(test_log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=test_fields)
            w.writeheader()
            w.writerow({
                "fold":           fold_idx + 1,
                "best_epoch":     best_epoch,
                "val_acc":        round(best_val_acc,  4),
                "val_f1":         round(best_val_f1,   4),
                "test_loss":      round(test_loss,     6),
                "test_acc":       round(test_acc,      4),
                "test_precision": round(test_prec,     4),
                "test_recall":    round(test_rec,      4),
                "test_f1":        round(test_f1,       4),
            })

        # Save classification report
        class_names = cfg["data"].get("class_names", None)
        report_str = classification_report(
            test_y_true, test_y_pred,
            target_names=class_names,
            zero_division=0,
        )
        report_path = os.path.join(fold_result_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Fold {fold_idx + 1} — Classification Report\n")
            f.write("=" * 50 + "\n")
            f.write(report_str)
        print(f"[fold {fold_idx + 1}] Report -> {report_path}")

        fold_results.append({
            "fold":       fold_idx + 1,
            "best_epoch": best_epoch,
            "val_acc":    best_val_acc,
            "val_f1":     best_val_f1,
            "test_loss":  test_loss,
            "test_acc":   test_acc,
            "test_prec":  test_prec,
            "test_rec":   test_rec,
            "test_f1":    test_f1,
            "log_path":   log_path,
            "ckpt_path":  ckpt_path,
        })

    # Aggregate summary across all folds
    val_accs   = np.array([r["val_acc"]   for r in fold_results])
    val_f1s    = np.array([r["val_f1"]    for r in fold_results])
    test_accs  = np.array([r["test_acc"]  for r in fold_results])
    test_precs = np.array([r["test_prec"] for r in fold_results])
    test_recs  = np.array([r["test_rec"]  for r in fold_results])
    test_f1s   = np.array([r["test_f1"]   for r in fold_results])

    print(f"\n{'='*70}")
    print(f"  {n_folds}-Fold CV Summary  ({model_name})")
    print(f"{'='*70}")
    header = f"{'Fold':>5}  {'Val Acc':>8}  {'Val F1':>8}  {'Test Acc':>9}  {'Test F1':>8}"
    print(header)
    print("-" * len(header))
    for r in fold_results:
        print(
            f"{r['fold']:>5}  {r['val_acc']:>7.2f}%  {r['val_f1']:>8.4f}"
            f"  {r['test_acc']:>8.2f}%  {r['test_f1']:>8.4f}"
        )
    print("-" * len(header))
    print(
        f"{'Mean':>5}  {val_accs.mean():>7.2f}%  {val_f1s.mean():>8.4f}"
        f"  {test_accs.mean():>8.2f}%  {test_f1s.mean():>8.4f}"
    )
    print(
        f"{'Std':>5}  {val_accs.std():>7.2f}%  {val_f1s.std():>8.4f}"
        f"  {test_accs.std():>8.2f}%  {test_f1s.std():>8.4f}"
    )
    print(f"{'='*70}\n")

    # Save cross-fold summary CSV
    summary_dir  = os.path.join(cfg["results"].get("base_dir", "results/"), model_name)
    summary_path = os.path.join(summary_dir, "cv_summary.csv")
    os.makedirs(summary_dir, exist_ok=True)
    sum_fields = ["fold", "best_epoch", "val_acc", "val_f1",
                  "test_acc", "test_precision", "test_recall", "test_f1"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        for r in fold_results:
            w.writerow({
                "fold":           r["fold"],
                "best_epoch":     r["best_epoch"],
                "val_acc":        round(r["val_acc"],   4),
                "val_f1":         round(r["val_f1"],    4),
                "test_acc":       round(r["test_acc"],  4),
                "test_precision": round(r["test_prec"], 4),
                "test_recall":    round(r["test_rec"],  4),
                "test_f1":        round(r["test_f1"],   4),
            })
        # Append mean/std rows
        for stat, fn in [("mean", np.mean), ("std", np.std)]:
            w.writerow({
                "fold":           stat,
                "best_epoch":     "",
                "val_acc":        round(float(fn(val_accs)),    4),
                "val_f1":         round(float(fn(val_f1s)),     4),
                "test_acc":       round(float(fn(test_accs)),   4),
                "test_precision": round(float(fn(test_precs)),  4),
                "test_recall":    round(float(fn(test_recs)),   4),
                "test_f1":        round(float(fn(test_f1s)),    4),
            })

    print(f"[train_cv] Summary saved -> {summary_path}")

    return {
        "fold_results":   fold_results,
        "mean_val_acc":   float(val_accs.mean()),
        "mean_val_f1":    float(val_f1s.mean()),
        "mean_test_acc":  float(test_accs.mean()),
        "mean_test_f1":   float(test_f1s.mean()),
        "summary_path":   summary_path,
    }

#==============================================================================
#  CLI entry-point
#==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="5-Fold Cross-Validation training for thyroid classification."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    args = parser.parse_args()
    train_cv(config_path=args.config)