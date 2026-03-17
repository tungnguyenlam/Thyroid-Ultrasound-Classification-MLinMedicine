"""
models/stacking/predict_meta_learner.py
Step 3 — Final test-set inference with the trained meta-learners.

Reads:
  models/stacking/intermediate/test_predictions.csv
  models/stacking/meta_learner_lr.pkl
  models/stacking/meta_learner_cv/fold*_best.pt

Outputs:
  results/stacking/lr/  — Metrics, report, plots for Logistic Regression
  results/stacking/cv/  — Metrics, report, plots for NN 5-Fold CV

CLI:
  python models/stacking/predict_meta_learner.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import pickle
from pathlib import Path
import glob

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.stacking.train_meta_learner import MetaLearner
from src.utils import load_config


# ===========================================================================
#  Plot helpers
# ===========================================================================

def _plot_confusion_matrix(cm, class_names, save_path, title):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_normalized_confusion_matrix(cm, class_names, save_path, title):
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
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2%}\n({cm[i, j]})",
                    ha="center", va="center", color=color, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_roc_curve(y_true, y_prob, auc_val, save_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set(xlim=[0, 1], ylim=[0, 1.02],
           xlabel="False Positive Rate",
           ylabel="True Positive Rate",
           title=title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ===========================================================================
#  Eval & Save block
# ===========================================================================

def evaluate_and_save(y_true, y_prob, out_dir, method_name, model_prefix, class_names) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy    = float((y_pred == y_true).mean()) * 100.0
    auc_val     = roc_auc_score(y_true, y_prob)
    f1_w        = f1_score(y_true, y_pred.astype(int), average="weighted", zero_division=0)
    cm          = confusion_matrix(y_true.astype(int), y_pred.astype(int))
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    report = classification_report(y_true.astype(int), y_pred.astype(int),
                                   target_names=class_names, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {method_name} — Final Test Results")
    print(f"{'='*60}")
    print(f"  Accuracy     : {accuracy:.2f}%")
    print(f"  AUC          : {auc_val:.4f}")
    print(f"  Sensitivity  : {sensitivity:.4f}")
    print(f"  Specificity  : {specificity:.4f}")
    print(f"  F1 (weighted): {f1_w:.4f}")
    print(f"{'='*60}")

    row = {
        "method":      method_name.replace(" ", "_").lower(),
        "accuracy":    round(accuracy, 4),
        "auc":         round(auc_val, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_weighted": round(f1_w, 4),
    }

    # Text report
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(f"{method_name} — Final Test Results\n\n")
        f.write(f"Accuracy     : {accuracy:.2f}%\n")
        f.write(f"AUC          : {auc_val:.4f}\n")
        f.write(f"Sensitivity  : {sensitivity:.4f}\n")
        f.write(f"Specificity  : {specificity:.4f}\n")
        f.write(f"F1 (weighted): {f1_w:.4f}\n\n")
        f.write(f"Classification Report:\n{report}")

    # Plots
    _plot_confusion_matrix(cm, class_names, os.path.join(out_dir, f"{model_prefix}_confusion_matrix.png"), f"{method_name} — Confusion Matrix")
    _plot_normalized_confusion_matrix(cm, class_names, os.path.join(out_dir, f"{model_prefix}_normalized_confusion_matrix.png"), f"{method_name} — Normalized Confusion Matrix")
    _plot_roc_curve(y_true, y_prob, auc_val, os.path.join(out_dir, f"{model_prefix}_roc_curve.png"), f"{method_name} — ROC Curve")

    return row

# ===========================================================================
#  Inference Functions
# ===========================================================================

def predict_lr(X_test, y_true, lr_path, out_dir, class_names):
    if not os.path.exists(lr_path):
        print(f"[ERROR] LR model missing at {lr_path}")
        return None

    with open(lr_path, "rb") as f:
        model_data = pickle.load(f)
    
    lr_model = model_data["model"]
    y_prob = lr_model.predict_proba(X_test)[:, 1]
    
    return evaluate_and_save(y_true, y_prob, out_dir, "Logistic Regression Ensemble", "lr", class_names)


def predict_nn_cv(X_test, y_true, cv_dir, out_dir, class_names, df_cols):
    fold_ckpts = glob.glob(os.path.join(cv_dir, "fold*_best.pt"))
    if not fold_ckpts:
        print(f"[ERROR] No NN fold checkpoints found in {cv_dir}")
        return None
    
    print(f"\n[predict] Found {len(fold_ckpts)} NN folds. Averaging predictions...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X_test).to(device)
    
    fold_probs = []
    
    for ckpt_path in fold_ckpts:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = MetaLearner(
            n_features=ckpt["n_features"], 
            hidden=ckpt["hidden"], 
            dropout=ckpt["dropout"]
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        with torch.no_grad():
            prob = model(X_t).cpu().numpy()
            fold_probs.append(prob)
            
    # Average probabilities across folds
    y_prob = np.mean(fold_probs, axis=0)
    
    return evaluate_and_save(y_true, y_prob, out_dir, "5-Fold NN Ensemble", "nn_cv", class_names)


# ===========================================================================
#  Main
# ===========================================================================

def main(config_path: str = "config/config.yaml"):
    cfg = load_config(config_path)
    class_names = cfg["data"].get("class_names", ["benign", "malignant"])

    stacking_dir     = os.path.join(str(_ROOT), "models", "stacking")
    intermediate_dir = os.path.join(stacking_dir, "intermediate")
    test_csv         = os.path.join(intermediate_dir, "test_predictions.csv")

    if not os.path.exists(test_csv):
        print(f"[ERROR] test predictions not found at {test_csv}. Run Step 1 first.")
        return

    df = pd.read_csv(test_csv)
    print(f"[predict] Loaded {len(df)} test samples from {test_csv}")

    prob_cols = [c for c in df.columns if c.endswith("_prob")]
    X_test = df[prob_cols].values.astype(np.float32)
    y_true = df["label"].values.astype(np.float32)

    results_dir = os.path.join(str(_ROOT), "results", "stacking")
    
    combined_rows = []

    # 1. Evaluate Logistic Regression
    lr_path = os.path.join(stacking_dir, "meta_learner_lr.pkl")
    lr_row = predict_lr(X_test, y_true, lr_path, os.path.join(results_dir, "lr"), class_names)
    if lr_row:
        combined_rows.append(lr_row)
    
    # 2. Evaluate NN 5-Fold CV
    cv_dir = os.path.join(stacking_dir, "meta_learner_cv")
    cv_row = predict_nn_cv(X_test, y_true, cv_dir, os.path.join(results_dir, "cv"), class_names, prob_cols)
    if cv_row:
        combined_rows.append(cv_row)
        
    # 3. Write combined summary to the root stacking results folder
    combined_csv_path = os.path.join(results_dir, "stacking_final_metrics.csv")
    if combined_rows:
        with open(combined_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
            w.writeheader()
            w.writerows(combined_rows)
        print(f"\n[predict] Combined test metrics saved to {combined_csv_path}")

    print(f"[predict] Detailed results and plots saved to {results_dir}/lr and {results_dir}/cv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3: Final test-set inference with both LR and NN meta-learners."
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
