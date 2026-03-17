"""
models/stacking/train_meta_learner.py
Step 2 — Train meta-learners on OOF predictions.

Supports two stacking approaches:
  1. NN_CV: 5-Fold Cross-Validation on a small Neural Network
  2. LR: Logistic Regression trained on 100% of the OOF data

Reads:
  models/stacking/intermediate/train_predictions.csv

Outputs:
  models/stacking/meta_learner_cv/       — NN checkpoints & logs for 5 folds
  models/stacking/meta_learner_lr.pkl    — Logistic Regression weights
  models/stacking/meta_training_log.csv  — Summary of both methods

CLI:
  python models/stacking/train_meta_learner.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import set_seed
from src.dataloader import BalancedEpochSampler


# ===========================================================================
#  Meta-learner NN Model
# ===========================================================================

class MetaLearner(nn.Module):
    """
    Small fully-connected network for stacking.
    """
    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)     # (B,)


# ===========================================================================
#  Logistic Regression
# ===========================================================================

def train_logistic_regression(X: np.ndarray, y: np.ndarray, prob_cols: list, seed: int):
    """Train a simple Logistic Regression on 100% of the data."""
    print(f"\n{'='*60}\n  Training Logistic Regression ({len(X)} samples)\n{'='*60}")
    
    # Class weight='balanced' achieves the same goal as our BalancedEpochSampler
    lr_model = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    lr_model.fit(X, y)
    
    # Eval on train set just for sanity check
    y_prob = lr_model.predict_proba(X)[:, 1]
    y_pred = lr_model.predict(X)
    
    acc = float((y_pred == y).mean()) * 100.0
    auc = roc_auc_score(y, y_prob)
    f1  = f1_score(y, y_pred, average="weighted", zero_division=0)
    
    print(f"  [LR] Train Acc: {acc:.2f}%  |  AUC: {auc:.4f}  |  F1: {f1:.4f}")
    
    # Coefficient importance
    print("\n  [LR] Feature Coefficients:")
    for col, coef in zip(prob_cols, lr_model.coef_[0]):
        print(f"    {col:>18s}: {coef:.4f}")
    print(f"    {"Intercept":>18s}: {lr_model.intercept_[0]:.4f}")
    
    # Save model
    stacking_dir = os.path.join(str(_ROOT), "models", "stacking")
    save_path = os.path.join(stacking_dir, "meta_learner_lr.pkl")
    
    model_data = {
        "model": lr_model,
        "prob_cols": prob_cols,
        "n_features": len(prob_cols),
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
        
    print(f"\n  [LR] Saved to {save_path}")
    return acc, auc, f1


# ===========================================================================
#  NN 5-Fold CV
# ===========================================================================

def _train_nn_fold(
    fold_k: int,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_features: int, prob_cols: list,
    epochs: int, patience: int, lr: float, weight_decay: float,
    hidden: int, dropout: float, seed: int, out_dir: str
):
    """Train one fold of the NN meta-learner."""
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    # Balance majority/minority classes
    train_sampler = BalancedEpochSampler(labels=y_train, seed=seed + fold_k)
    train_loader  = DataLoader(train_ds, batch_size=64, sampler=train_sampler)
    val_loader    = DataLoader(val_ds,   batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetaLearner(n_features=n_features, hidden=hidden, dropout=dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    ckpt_path = os.path.join(out_dir, f"fold{fold_k}_best.pt")
    plot_path = os.path.join(out_dir, f"fold{fold_k}_learning_curve.png")

    best_val_loss = float("inf")
    no_improve = 0
    history_train_loss, history_val_loss = [], []

    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss_sum, train_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(yb)
            train_n += len(yb)
        train_loss = train_loss_sum / train_n

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss_sum += loss.item() * len(yb)
                val_n += len(yb)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        val_loss = val_loss_sum / val_n
        
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            y_true = np.concatenate(all_labels)
            y_prob = np.concatenate(all_preds)
            y_pred = (y_prob >= 0.5).astype(int)
            
            val_acc = float((y_pred == y_true).mean()) * 100.0
            val_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
            val_f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_features": n_features,
                "hidden": hidden,
                "dropout": dropout,
                "prob_cols": prob_cols,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_f1": val_f1,
            }, ckpt_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # Plot learning curve
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(range(1, len(history_train_loss) + 1), history_train_loss, label="Train Loss")
    ax.plot(range(1, len(history_val_loss) + 1), history_val_loss, label="Val Loss")
    best_idx = np.argmin(history_val_loss)
    ax.plot(best_idx + 1, history_val_loss[best_idx], "ro", label="Best Val Loss")
    ax.set_title(f"NN Fold {fold_k}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  [NN Fold {fold_k}] Best Epoch: {best_idx + 1}  |  Val Acc: {ckpt['val_acc']:.2f}%  |  AUC: {ckpt['val_auc']:.4f}")
    return ckpt["val_acc"], ckpt["val_auc"], ckpt["val_f1"]


def train_nn_cv(X: np.ndarray, y: np.ndarray, prob_cols: list, args):
    """Train 5-Fold CV Neural Network Meta-Learner."""
    print(f"\n{'='*60}\n  Training NN Meta-Learner (5-Fold CV)\n{'='*60}")
    
    cv_dir = os.path.join(str(_ROOT), "models", "stacking", "meta_learner_cv")
    os.makedirs(cv_dir, exist_ok=True)
    
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    
    fold_metrics = []
    
    for fold_k, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val     = X[val_idx], y[val_idx]
        
        acc, auc, f1 = _train_nn_fold(
            fold_k=fold_k,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            n_features=len(prob_cols), prob_cols=prob_cols,
            epochs=args.epochs, patience=args.patience,
            lr=args.lr, weight_decay=args.weight_decay,
            hidden=args.hidden, dropout=args.dropout,
            seed=args.seed, out_dir=cv_dir
        )
        fold_metrics.append((acc, auc, f1))
        
    accs = [m[0] for m in fold_metrics]
    aucs = [m[1] for m in fold_metrics]
    
    print(f"\n  [NN CV] Mean Val Acc: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    print(f"  [NN CV] Mean Val AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  [NN CV] Models saved in {cv_dir}/")
    
    return np.mean(accs), np.mean(aucs), np.mean([m[2] for m in fold_metrics])


# ===========================================================================
#  Main Entry
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--patience",     type=int,   default=25)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden",       type=int,   default=32)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load data
    train_csv = os.path.join(str(_ROOT), "models", "stacking", "intermediate", "train_predictions.csv")
    if not os.path.exists(train_csv):
        print(f"[ERROR] {train_csv} not found.")
        return

    df = pd.read_csv(train_csv)
    prob_cols = [c for c in df.columns if c.endswith("_prob")]
    X = df[prob_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        X, y = X[~nan_mask], y[~nan_mask]

    # 1) Train Logistic Regression
    lr_acc, lr_auc, lr_f1 = train_logistic_regression(X, y, prob_cols, args.seed)

    # 2) Train NN 5-Fold CV
    nn_acc, nn_auc, nn_f1 = train_nn_cv(X, y, prob_cols, args)
    
    # Save overall summary
    log_path = os.path.join(str(_ROOT), "models", "stacking", "meta_training_summary.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "trainval_acc", "trainval_auc", "trainval_f1"])
        w.writerow(["LogisticRegression", round(lr_acc, 4), round(lr_auc, 4), round(lr_f1, 4)])
        w.writerow(["NeuralNetwork_5FoldCV", round(nn_acc, 4), round(nn_auc, 4), round(nn_f1, 4)])
        
    print(f"\n{'='*60}\n  Summary saved to {log_path}\n{'='*60}")


if __name__ == "__main__":
    main()
