"""
models/stacking/generate_predictions.py
Step 1 — Generate base-model probability predictions for stacking.

Strategy (Out-of-Fold to avoid data leakage):
  - For each architecture (res18, res50, densenet, efficientnet) and each
    fold (1–5), load the checkpoint and run inference.
  - TRAIN predictions (OOF): Each trainval sample appears in the *validation*
    set of exactly one fold.  We use that fold's model to predict on it,
    giving one clean P(malignant) per architecture per sample.
  - TEST predictions: All 5 fold models predict on the shared held-out test
    set; probabilities are averaged across folds per architecture.

Outputs (saved to models/stacking/intermediate/):
  - train_predictions.csv   — sample_idx, label, <arch>_prob …
  - test_predictions.csv    — sample_idx, label, <arch>_prob …

CLI:
  python models/stacking/generate_predictions.py
  python models/stacking/generate_predictions.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ---------------------------------------------------------------------------
#  Project root
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataloader import ThyroidDataset, get_transforms
from src.utils import load_config, set_seed
from src.train import build_model
from sklearn.model_selection import train_test_split, StratifiedKFold

# ---------------------------------------------------------------------------
#  Architectures to scan — auto-discovered from checkpoints/
# ---------------------------------------------------------------------------
ARCHITECTURES = ["res18", "res50", "densenet", "efficientnet"]


# ===========================================================================
#  Helpers
# ===========================================================================

def _get_splits(cfg: dict, n_folds: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple]]:
    """
    Reproduce the exact same data split used during training.

    Returns:
        trainval_idx : indices of the trainval pool
        test_idx     : indices of the held-out test set
        labels       : full label array for the dataset
        folds        : list of (rel_train_idx, rel_val_idx) tuples
    """
    data_cfg    = cfg["data"]
    data_dir    = data_cfg["data_dir"]
    seed        = int(data_cfg.get("seed", 42))
    train_ratio = float(data_cfg.get("train_ratio", 0.70))
    val_ratio   = float(data_cfg.get("val_ratio",   0.15))

    base_ds = ThyroidDataset(root=data_dir)
    labels  = np.array([label for _, label in base_ds.dataset.samples])
    indices = np.arange(len(base_ds))

    # Fixed test set (same as get_cv_fold_loaders)
    test_ratio = 1.0 - train_ratio - val_ratio
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # k-fold on trainval pool
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(trainval_idx, labels[trainval_idx]))

    return trainval_idx, test_idx, labels, folds


def _predict_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run inference and return softmax probabilities (N, C)."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


# ===========================================================================
#  Main
# ===========================================================================

def generate_predictions(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["data"]["seed"])

    data_cfg   = cfg["data"]
    data_dir   = data_cfg["data_dir"]
    img_size   = int(data_cfg.get("img_size", 512))
    seed       = int(data_cfg.get("seed", 42))
    n_folds    = int(cfg["training"].get("n_folds", 5))
    batch_size = int(cfg["training"].get("batch_size", 16))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate_predictions] Device: {device}")

    # ---- Reproduce splits ------------------------------------------------
    trainval_idx, test_idx, labels, folds = _get_splits(cfg, n_folds)

    # Build a non-augmented dataset for clean inference
    infer_ds = ThyroidDataset(
        root=data_dir,
        transform=get_transforms(img_size, "val"),  # no augmentation
    )

    # ---- Auto-discover architectures from checkpoints/ -------------------
    ckpt_root = os.path.join(str(_ROOT), "checkpoints")
    architectures = []
    for arch in sorted(os.listdir(ckpt_root)):
        arch_dir = os.path.join(ckpt_root, arch)
        if os.path.isdir(arch_dir):
            # Check it has fold subdirectories with best.pt
            has_folds = any(
                os.path.exists(os.path.join(arch_dir, f"fold{k}", "best.pt"))
                for k in range(1, n_folds + 1)
            )
            if has_folds:
                architectures.append(arch)

    if not architectures:
        print("[ERROR] No architectures found in checkpoints/. Exiting.")
        return

    print(f"[generate_predictions] Architectures found: {architectures}")
    print(f"[generate_predictions] Folds: {n_folds}")
    print(f"[generate_predictions] TrainVal samples: {len(trainval_idx)} | Test samples: {len(test_idx)}")

    # ---- Containers for predictions -------------------------------------
    # OOF: for each architecture, accumulate predictions for all trainval samples
    oof_probs = {arch: np.full(len(trainval_idx), np.nan, dtype=np.float64)
                 for arch in architectures}

    # Test: for each architecture, accumulate fold-averaged predictions
    test_probs = {arch: np.zeros(len(test_idx), dtype=np.float64)
                  for arch in architectures}

    # Lookup: absolute dataset index -> position in trainval_idx array
    abs_to_tv_pos = {int(abs_idx): pos for pos, abs_idx in enumerate(trainval_idx)}

    # ---- Test loader (shared across all folds) ---------------------------
    test_loader = DataLoader(
        Subset(infer_ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ---- Generate predictions -------------------------------------------
    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"  Architecture: {arch}")
        print(f"{'='*60}")

        for fold_k in range(n_folds):
            fold_num = fold_k + 1
            ckpt_path = os.path.join(ckpt_root, arch, f"fold{fold_num}", "best.pt")

            if not os.path.exists(ckpt_path):
                print(f"  [WARN] Missing checkpoint: {ckpt_path} — skipping fold {fold_num}")
                continue

            # Build model with the correct architecture
            cfg_copy = dict(cfg)
            cfg_copy["model"] = dict(cfg["model"])
            cfg_copy["model"]["name"] = arch
            model = build_model(cfg_copy).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()
            print(f"  Fold {fold_num}: loaded {ckpt_path}")

            # ---- OOF: predict on this fold's validation samples --------
            rel_train_idx, rel_val_idx = folds[fold_k]
            fold_val_idx = trainval_idx[rel_val_idx]  # absolute dataset indices

            val_loader = DataLoader(
                Subset(infer_ds, fold_val_idx),
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            val_probs = _predict_probs(model, val_loader, device)
            # P(malignant) — class index 1 for binary
            p_mal = val_probs[:, 1]

            # Map back to trainval positions
            for i, abs_idx in enumerate(fold_val_idx):
                tv_pos = abs_to_tv_pos[int(abs_idx)]
                oof_probs[arch][tv_pos] = p_mal[i]

            print(f"    OOF: {len(fold_val_idx)} val samples predicted")

            # ---- Test: predict on test set (accumulate for averaging) ---
            test_p = _predict_probs(model, test_loader, device)
            test_probs[arch] += test_p[:, 1]

            print(f"    Test: {len(test_idx)} samples predicted")

        # Average test predictions across folds
        test_probs[arch] /= n_folds

    # ---- Validate OOF completeness --------------------------------------
    for arch in architectures:
        nan_count = np.isnan(oof_probs[arch]).sum()
        if nan_count > 0:
            print(f"[WARN] {arch}: {nan_count} trainval samples have no OOF prediction!")

    # ---- Save to CSV ----------------------------------------------------
    out_dir = os.path.join(str(_ROOT), "models", "stacking", "intermediate")
    os.makedirs(out_dir, exist_ok=True)

    # Column names
    arch_cols = [f"{arch}_prob" for arch in architectures]
    header = ["sample_idx", "label"] + arch_cols

    # Train predictions (OOF)
    train_path = os.path.join(out_dir, "train_predictions.csv")
    with open(train_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, abs_idx in enumerate(trainval_idx):
            row = [int(abs_idx), int(labels[abs_idx])]
            row += [round(float(oof_probs[arch][i]), 6) for arch in architectures]
            writer.writerow(row)
    print(f"\n[generate_predictions] Train OOF saved -> {train_path}  ({len(trainval_idx)} rows)")

    # Test predictions (fold-averaged)
    test_path = os.path.join(out_dir, "test_predictions.csv")
    with open(test_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, abs_idx in enumerate(test_idx):
            row = [int(abs_idx), int(labels[abs_idx])]
            row += [round(float(test_probs[arch][i]), 6) for arch in architectures]
            writer.writerow(row)
    print(f"[generate_predictions] Test preds saved  -> {test_path}  ({len(test_idx)} rows)")

    print("\n[generate_predictions] Done! ✓")


# ===========================================================================
#  CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Generate base-model OOF + test predictions for stacking."
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    args = parser.parse_args()
    generate_predictions(config_path=args.config)
