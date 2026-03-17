"""
src/test.py
Ensemble evaluation script for Thyroid Ultrasound Classification (5-Fold CV).

Loads all fold checkpoints from checkpoints/<model>/fold<k>/best.pt,
averages their softmax probabilities over the shared test set, and writes
a full ensemble report to results/<model_name>/.

Outputs:
  - ensemble_report.txt          - classification report (precision / recall / F1)
  - ensemble_metrics.csv         - accuracy, AUC, sensitivity, specificity, F1
  - ensemble_confusion_matrix.png
  - normalize_ensemble_confusion_matrix.png
  - ensemble_roc_curve.png

CLI:
  python src/test.py
  python src/test.py --config config/config.yaml
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# Project root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataloader import get_cv_fold_loaders, get_tta_transforms
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
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
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
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_normalized_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str,
) -> None:
    """Save a row-normalized (recall) confusion-matrix PNG with % annotations."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Proportion", fontsize=9)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Normalized Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(
                j, i,
                f"{cm_norm[i, j]:.2%}\n({cm[i, j]})",
                ha="center", va="center",
                color=color,
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_roc_curve(
    all_labels: np.ndarray,
    probs: np.ndarray,
    class_names: list,
    auc: float,
    save_path: str,
) -> None:
    """Save a ROC curve PNG (binary or multi-class one-vs-rest)."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    n_classes = probs.shape[1]

    if n_classes == 2:
        # Binary: one curve for the positive (malignant) class
        fpr, tpr, _ = roc_curve(all_labels, probs[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f"{class_names[1]} (AUC = {auc:.4f})")
    else:
        # Multi-class: one curve per class (OvR)
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(all_labels, classes=list(range(n_classes)))
        for i, name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            cls_auc = roc_auc_score(y_bin[:, i], probs[:, i])
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {cls_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set(xlim=[0, 1], ylim=[0, 1.02],
           xlabel="False Positive Rate",
           ylabel="True Positive Rate",
           title="ROC Curve (Ensemble)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)



#==============================================================================
#  CV Ensemble evaluation
#==============================================================================

def evaluate_cv_ensemble(
    config_path: str = "config/config.yaml",
    n_folds: int | None = None,
) -> Dict:
    """
    Ensemble evaluation across all CV fold checkpoints with TTA.

    For each fold model, runs ``N`` TTA views (original + flips + rotations)
    per image and averages the softmax probabilities. Probabilities are then
    averaged across all fold models, giving ``n_folds x n_tta`` predictions
    per image in total.

    Args:
        config_path: Path to YAML config.
        n_folds:     Number of folds (overrides config if given).

    Returns:
        dict with accuracy, auc, sensitivity, specificity, f1_weighted,
        report_path, cm_path, metrics_csv_path.
    """
    cfg = load_config(config_path)
    set_seed(cfg["data"]["seed"])

    num_classes = int(cfg["data"]["num_classes"])
    model_name  = cfg["model"]["name"]
    n_folds     = n_folds or int(cfg["training"].get("n_folds", 5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ensemble] Device: {device} | Folds: {n_folds}")

    # Shared test set — use fold 0 loader (all folds share the same test set,
    # held out with the same 15% direct split as used in train_cv)
    _, _, test_loader, _ = get_cv_fold_loaders(cfg, fold_idx=0, n_folds=n_folds)

    from torchvision import datasets as tvd
    class_names = tvd.ImageFolder(root=cfg["data"]["data_dir"]).classes

    # Accumulate softmax probabilities from each fold
    probs_sum  = None      # (N, C) running sum
    all_labels = None

    use_tta  = bool(cfg["training"].get("use_tta", True))
    img_size = int(cfg["data"].get("img_size", 512))

    tta_transforms = get_tta_transforms(img_size) if use_tta else None
    n_tta = len(tta_transforms) if use_tta else 1
    print(f"[ensemble] TTA: {'enabled (' + str(n_tta) + ' views)' if use_tta else 'disabled'} | "
          f"Total predictions per image: {n_folds * n_tta}")

    for fold_k in range(1, n_folds + 1):
        ckpt_path = os.path.join("checkpoints", model_name, f"fold{fold_k}", "best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        print(f"  Fold {fold_k}: loaded {ckpt_path}")

        fold_probs  = []
        fold_labels = []

        import torchvision.transforms.functional as F_tv

        with torch.no_grad():
            for images, labels in tqdm(test_loader,
                                       desc=f"  Fold {fold_k}",
                                       leave=False,
                                       dynamic_ncols=True):
                batch_size = labels.size(0)

                if use_tta:
                    probs_sum_batch = torch.zeros(batch_size, cfg["data"]["num_classes"])
                    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

                    for t_fn in tta_transforms:
                        aug_imgs = []
                        for i in range(batch_size):
                            img_01  = (images[i] * std_t + mean_t).clamp(0, 1)
                            pil_img = F_tv.to_pil_image(img_01)
                            aug_imgs.append(t_fn(pil_img))
                        aug_batch = torch.stack(aug_imgs).to(device, non_blocking=True)
                        logits    = model(aug_batch)
                        probs_sum_batch += torch.softmax(logits, dim=1).cpu()

                    fold_probs.append(probs_sum_batch / n_tta)
                else:
                    images = images.to(device, non_blocking=True)
                    logits = model(images)
                    fold_probs.append(torch.softmax(logits, dim=1).cpu())

                fold_labels.append(labels)

        fold_probs  = torch.cat(fold_probs).numpy()   # (N, C)
        fold_labels = torch.cat(fold_labels).numpy()  # (N,)

        if probs_sum is None:
            probs_sum  = fold_probs
            all_labels = fold_labels
        else:
            probs_sum += fold_probs

    probs_avg = probs_sum / n_folds
    all_preds = probs_avg.argmax(axis=1)

    # Metrics
    accuracy = float((all_preds == all_labels).mean()) * 100.0

    if num_classes == 2:
        auc = roc_auc_score(all_labels, probs_avg[:, 1])
    else:
        auc = roc_auc_score(all_labels, probs_avg,
                            multi_class="ovr", average="weighted")

    cm = confusion_matrix(all_labels, all_preds)
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        per_class_recall = cm.diagonal() / cm.sum(axis=1)
        sensitivity = float(per_class_recall.mean())
        specificity = float("nan")

    report = classification_report(all_labels, all_preds, target_names=class_names)

    from sklearn.metrics import f1_score
    f1_w = f1_score(all_labels, all_preds, average="weighted")

    # Console
    print("\n" + "=" * 60)
    print(f"  Ensemble ({n_folds} folds)  -  {model_name}")
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

    report_path = os.path.join(result_dir, "ensemble_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}  |  Ensemble of {n_folds} folds\n\n")
        f.write(f"Accuracy    : {accuracy:.2f}%\n")
        f.write(f"AUC         : {auc:.4f}\n")
        f.write(f"Sensitivity : {sensitivity:.4f}\n")
        if num_classes == 2:
            f.write(f"Specificity : {specificity:.4f}\n")
        f.write(f"F1 (weighted): {f1_w:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    metrics_csv = os.path.join(result_dir, "ensemble_metrics.csv")
    row = {
        "model":       model_name,
        "n_folds":     n_folds,
        "accuracy":    round(accuracy,    4),
        "auc":         round(auc,         4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4) if num_classes == 2 else "N/A",
        "f1_weighted": round(f1_w,        4),
    }
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    cm_path = os.path.join(result_dir, "ensemble_confusion_matrix.png")
    _plot_confusion_matrix(cm, class_names, cm_path)

    cm_norm_path = os.path.join(result_dir, "normalize_ensemble_confusion_matrix.png")
    _plot_normalized_confusion_matrix(cm, class_names, cm_norm_path)

    roc_path = os.path.join(result_dir, "ensemble_roc_curve.png")
    _plot_roc_curve(all_labels, probs_avg, class_names, auc, roc_path)

    print(f"\n[ensemble] Report       -> {report_path}")
    print(f"[ensemble] Metrics CSV  -> {metrics_csv}")
    print(f"[ensemble] Confusion mat-> {cm_path}")
    print(f"[ensemble] Norm. CM     -> {cm_norm_path}")
    print(f"[ensemble] ROC curve    -> {roc_path}")

    return {
        "accuracy":         accuracy,
        "auc":              auc,
        "sensitivity":      sensitivity,
        "specificity":      specificity if num_classes == 2 else None,
        "f1_weighted":      f1_w,
        "report_path":      report_path,
        "cm_path":          cm_path,
        "cm_norm_path":     cm_norm_path,
        "roc_path":         roc_path,
        "metrics_csv_path": metrics_csv,
    }


#==============================================================================
#  CLI entry-point
#==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble evaluation across all CV fold checkpoints."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    args = parser.parse_args()
    evaluate_cv_ensemble(config_path=args.config)
