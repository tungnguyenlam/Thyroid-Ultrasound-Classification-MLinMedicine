"""
src/test.py
Evaluation script for Thyroid Ultrasound Classification.

Automatically loads the best checkpoint for the model specified in
config/config.yaml, runs it over the held-out test set, and writes a
full report to results/<model_name>/.

Outputs:
  - test_report.txt  - classification report (precision / recall / F1)
  - test_metrics.csv - accuracy, AUC, sensitivity, specificity, F1
  - confusion_matrix.png

CLI:
  python src/test.py
  python src/test.py --config config/config.yaml

Can also be called from a Jupyter notebook:
  from src.test import evaluate
  metrics = evaluate(config_path="config/config.yaml")
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

# Project root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataloader import get_dataloaders
from src.utils import get_result_dir, load_config, set_seed
from src.train import build_model


#==============================================================================
#  Helpers
#==============================================================================

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str,
) -> None:
    """Save a styled confusion-matrix PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


#==============================================================================
#  Main evaluation function
#==============================================================================

def evaluate(config_path: str = "config/config.yaml") -> Dict:
    """
    Load the best checkpoint and evaluate on the test set.

    Args:
        config_path: Path to the YAML config.

    Returns:
        dict with keys: accuracy, auc, sensitivity, specificity,
                        f1_weighted, report_path, cm_path, metrics_csv_path.
    """
    # Config & reproducibility
    cfg = load_config(config_path)
    set_seed(cfg["data"]["seed"])

    num_classes = int(cfg["data"]["num_classes"])
    model_name  = cfg["model"]["name"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] Device: {device}")

    # Data
    _, _, test_loader, _ = get_dataloaders(cfg)

    # Recover class names from the base dataset
    from torchvision import datasets
    base_ds    = datasets.ImageFolder(root=cfg["data"]["data_dir"])
    class_names = base_ds.classes           # ['benign', 'malignant']

    # Load checkpoint
    ckpt_path = os.path.join("checkpoints", model_name, "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt_path}'. "
            "Please run train.py first."
        )

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    trained_epoch = ckpt.get("epoch", "?")
    print(f"[test] Loaded checkpoint from epoch {trained_epoch}  "
          f"(val_loss={ckpt.get('val_loss', float('nan')):.4f}  "
          f"val_acc={ckpt.get('val_acc', float('nan')):.1f}%)")

    # Inference
    all_preds  = []
    all_labels = []
    all_probs  = []   # softmax probabilities for AUC

    pbar = tqdm(test_loader, desc="Testing", leave=True, dynamic_ncols=True, unit="batch")
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())

    pbar.close()

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()    # (N, C)

    # Metrics
    accuracy = float((all_preds == all_labels).mean()) * 100.0

    # AUC (binary -> prob of positive class; multi-class -> OvR)
    if num_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="weighted")

    # Confusion matrix -> sensitivity (recall of positive) & specificity
    cm = confusion_matrix(all_labels, all_preds)
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Per-class recall diagonal
        per_class_recall = cm.diagonal() / cm.sum(axis=1)
        sensitivity = float(per_class_recall.mean())  # macro avg sensitivity
        specificity = float("nan")                     # not well-defined for multi-class

    report = classification_report(all_labels, all_preds, target_names=class_names)

    # Weighted F1 from report
    from sklearn.metrics import f1_score
    f1_w = f1_score(all_labels, all_preds, average="weighted")

    # Console output
    print("\n" + "=" * 60)
    print(f"  Model       : {model_name}")
    print(f"  Accuracy    : {accuracy:.2f}%")
    print(f"  AUC         : {auc:.4f}")
    print(f"  Sensitivity : {sensitivity:.4f}")
    if num_classes == 2:
        print(f"  Specificity : {specificity:.4f}")
    print(f"  F1 (weighted): {f1_w:.4f}")
    print("=" * 60)
    print("\nClassification Report:\n")
    print(report)

    # Save outputs
    result_dir = get_result_dir(cfg)

    # 1. Text report
    report_path = os.path.join(result_dir, "test_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint epoch: {trained_epoch}\n\n")
        f.write(f"Accuracy    : {accuracy:.2f}%\n")
        f.write(f"AUC         : {auc:.4f}\n")
        f.write(f"Sensitivity : {sensitivity:.4f}\n")
        if num_classes == 2:
            f.write(f"Specificity : {specificity:.4f}\n")
        f.write(f"F1 (weighted): {f1_w:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # 2. Metrics CSV
    metrics_csv = os.path.join(result_dir, "test_metrics.csv")
    row = {
        "model":        model_name,
        "accuracy":     round(accuracy,   4),
        "auc":          round(auc,        4),
        "sensitivity":  round(sensitivity, 4),
        "specificity":  round(specificity, 4) if num_classes == 2 else "N/A",
        "f1_weighted":  round(f1_w,        4),
        "best_epoch":   trained_epoch,
    }
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    # 3. Confusion matrix PNG
    cm_path = os.path.join(result_dir, "confusion_matrix.png")
    _plot_confusion_matrix(cm, class_names, cm_path)

    print(f"\n[test] Report       -> {report_path}")
    print(f"[test] Metrics CSV  -> {metrics_csv}")
    print(f"[test] Confusion mat-> {cm_path}")

    return {
        "accuracy":         accuracy,
        "auc":              auc,
        "sensitivity":      sensitivity,
        "specificity":      specificity if num_classes == 2 else None,
        "f1_weighted":      f1_w,
        "report_path":      report_path,
        "cm_path":          cm_path,
        "metrics_csv_path": metrics_csv,
    }


#==============================================================================
#  CLI entry-point
#==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a thyroid classification model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    args = parser.parse_args()
    evaluate(config_path=args.config)
