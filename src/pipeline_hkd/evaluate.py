"""
Test-set evaluation with Test-Time Augmentation (TTA).
Generates: confusion matrix, ROC curve, PR curve, classification report.

Usage:
    cd src
    python -m pipeline_hkd.evaluate --model efficientnet --checkpoint checkpoints/efficientnet/best_efficientnet.pt
"""

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, f1_score
)

from .config import Config
from .dataset import get_dataloaders, get_tta_transforms, ThyroidDataset
from .models import get_model
from .utils import set_seed, get_device, load_checkpoint


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_with_tta(model, test_dataset, tta_transforms, device, batch_size=16):
    """Average predictions over multiple augmented views."""
    model.eval()

    all_probs = []
    all_labels = []

    # Collect labels once
    for i in range(len(test_dataset)):
        _, label = test_dataset[i]
        all_labels.append(label.item())
    all_labels = np.array(all_labels)

    # For each TTA transform, create a new dataset and predict
    for t_idx, transform in enumerate(tta_transforms):
        tta_dataset = ThyroidDataset(
            test_dataset.image_paths,
            test_dataset.labels,
            transform=transform,
        )
        loader = torch.utils.data.DataLoader(
            tta_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        view_logits = []
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            view_logits.append(logits.cpu())

        view_logits = torch.cat(view_logits)
        view_probs = torch.sigmoid(view_logits).numpy()
        all_probs.append(view_probs)
        print(f"  TTA view {t_idx + 1}/{len(tta_transforms)} done")

    # Average probabilities across views
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs, all_labels


@torch.no_grad()
def predict_standard(model, loader, device):
    """Standard prediction (no TTA)."""
    model.eval()
    all_logits, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).numpy()
    probs = torch.sigmoid(all_logits).numpy()
    return probs, all_labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
    )
    plt.xlabel("Predicted", fontsize=13)
    plt.ylabel("True", fontsize=13)
    plt.title("Confusion Matrix", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curve(y_true, y_probs, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("ROC Curve", fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pr_curve(y_true, y_probs, save_path):
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR (AP = {ap:.4f})")
    plt.xlabel("Recall", fontsize=13)
    plt.ylabel("Precision", fontsize=13)
    plt.title("Precision-Recall Curve", fontsize=15)
    plt.legend(loc="lower left", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(log_path, save_path):
    """Plot training curves from CSV log."""
    import pandas as pd
    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o", ms=3)
    axes[0, 0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker="s", ms=3)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # AUC-ROC
    axes[0, 1].plot(df["epoch"], df["val_roc_auc"], label="Val AUC-ROC",
                    color="green", marker="o", ms=3)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("AUC-ROC")
    axes[0, 1].set_title("Validation AUC-ROC")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision & Recall
    axes[1, 0].plot(df["epoch"], df["val_precision"], label="Precision",
                    marker="o", ms=3)
    axes[1, 0].plot(df["epoch"], df["val_recall"], label="Recall",
                    marker="s", ms=3)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Precision & Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # F1
    axes[1, 1].plot(df["epoch"], df["val_f1"], label="F1",
                    color="purple", marker="o", ms=3)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].set_title("Validation F1 Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg: Config, checkpoint_path: str, use_tta: bool = True):
    """Run full evaluation on the test set."""

    set_seed(cfg.seed)
    device = get_device()
    model_results_dir = os.path.join(cfg.paths.results_dir, cfg.model.name)
    os.makedirs(model_results_dir, exist_ok=True)

    # --- Data ---
    print("\n=== Loading Data ===")
    _, _, test_loader, _, _ = get_dataloaders(cfg.data)
    test_dataset = test_loader.dataset

    # --- Model ---
    print("\n=== Loading Model ===")
    model = get_model(cfg.model.name, pretrained=False, dropout=cfg.model.dropout)
    model = model.to(device)
    load_checkpoint(model, checkpoint_path, device)

    # --- Predict ---
    print("\n=== Running Predictions ===")
    if use_tta:
        print("  Using Test-Time Augmentation (5 views)")
        tta_transforms = get_tta_transforms(cfg.data)
        probs, labels = predict_with_tta(
            model, test_dataset, tta_transforms, device
        )
    else:
        probs, labels = predict_standard(model, test_loader, device)

    preds = (probs >= 0.5).astype(int)

    # --- Metrics ---
    print("\n=== Test Results ===")
    auc_score = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds)
    report = classification_report(
        labels, preds,
        target_names=["Benign", "Malignant"],
        digits=4,
    )
    print(f"  AUC-ROC: {auc_score:.4f}")
    print(f"  F1:      {f1:.4f}")
    print(f"\n{report}")

    # Save report
    report_path = os.path.join(model_results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {cfg.model.name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"TTA: {use_tta}\n")
        f.write(f"AUC-ROC: {auc_score:.4f}\n")
        f.write(f"F1: {f1:.4f}\n\n")
        f.write(report)
    print(f"  Saved: {report_path}")

    # --- Plots ---
    print("\n=== Generating Plots ===")

    plot_confusion_matrix(
        labels, preds,
        os.path.join(model_results_dir, "confusion_matrix.png"),
    )
    plot_roc_curve(
        labels, probs,
        os.path.join(model_results_dir, "roc_curve.png"),
    )
    plot_pr_curve(
        labels, probs,
        os.path.join(model_results_dir, "pr_curve.png"),
    )

    # Plot training curves if log exists
    training_log = os.path.join(model_results_dir, "training_log.csv")
    if os.path.exists(training_log):
        plot_training_curves(
            training_log,
            os.path.join(model_results_dir, "training_curves.png"),
        )

    print("\n=== Evaluation Complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate thyroid ultrasound classifier")
    parser.add_argument("--model", type=str, required=True,
                        choices=["efficientnet", "resnet50", "densenet"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable Test-Time Augmentation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.model.name = args.model
    cfg.data.batch_size = args.batch_size
    cfg.seed = args.seed

    evaluate(cfg, args.checkpoint, use_tta=not args.no_tta)


if __name__ == "__main__":
    main()
