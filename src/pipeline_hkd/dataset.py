"""
Dataset, transforms, balanced epoch sampler, and dataloader construction.
"""

import os
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import DataConfig


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transforms(cfg: DataConfig) -> transforms.Compose:
    """Aggressive augmentation pipeline for training."""
    return transforms.Compose([
        transforms.Resize(cfg.image_size + 32),  # 256 for image_size=224
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def get_val_transforms(cfg: DataConfig) -> transforms.Compose:
    """Deterministic transforms for validation / test."""
    return transforms.Compose([
        transforms.Resize(cfg.image_size + 32),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def get_tta_transforms(cfg: DataConfig) -> List[transforms.Compose]:
    """5 augmented views for Test-Time Augmentation."""
    base = [
        transforms.Resize(cfg.image_size + 32),
        transforms.CenterCrop(cfg.image_size),
    ]
    norm = [
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ]
    return [
        # 1. Original
        transforms.Compose(base + norm),
        # 2. Horizontal flip
        transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0)] + norm),
        # 3. Vertical flip
        transforms.Compose(base + [transforms.RandomVerticalFlip(p=1.0)] + norm),
        # 4. Small rotation +10
        transforms.Compose(base + [transforms.RandomRotation((10, 10))] + norm),
        # 5. Small rotation -10
        transforms.Compose(base + [transforms.RandomRotation((-10, -10))] + norm),
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ThyroidDataset(Dataset):
    """Thyroid ultrasound image dataset."""

    def __init__(self, image_paths: List[str], labels: List[int],
                 transform: Optional[transforms.Compose] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# ---------------------------------------------------------------------------
# Balanced Epoch Sampler
# ---------------------------------------------------------------------------

class BalancedEpochSampler(Sampler):
    """
    Epoch-level balanced subsampling.

    Every epoch:
      - Uses ALL samples from the minority class (positive / benign).
      - Randomly samples an equal number from the majority class (malignant).
      - The majority subset is re-sampled each epoch.
    """

    def __init__(self, labels: List[int], seed: int = 42):
        super().__init__()
        self.labels = np.array(labels)
        self.seed = seed
        self.epoch = 0

        # Identify minority / majority indices
        unique, counts = np.unique(self.labels, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]

        self.minority_indices = np.where(self.labels == minority_class)[0]
        self.majority_indices = np.where(self.labels == majority_class)[0]
        self.n_minority = len(self.minority_indices)

        print(f"  BalancedEpochSampler: minority={self.n_minority} "
              f"(class={minority_class}), majority={len(self.majority_indices)} "
              f"(class={majority_class})")
        print(f"  Each epoch will use {2 * self.n_minority} samples "
              f"({self.n_minority} per class)")

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        # All minority indices
        minority = self.minority_indices.copy()
        # Random subset of majority
        majority_subset = rng.choice(
            self.majority_indices, size=self.n_minority, replace=False
        )
        # Combine and shuffle
        indices = np.concatenate([minority, majority_subset])
        rng.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return 2 * self.n_minority

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch to re-sample majority class."""
        self.epoch = epoch


# ---------------------------------------------------------------------------
# Dataloader Construction
# ---------------------------------------------------------------------------

def _download_dataset() -> str:
    """Download dataset via kagglehub and return path."""
    import kagglehub
    path = kagglehub.dataset_download("sowmyaabirami/thyroid-ultrasound-dataset")
    print(f"  Dataset path: {path}")
    return path


def _collect_image_paths(dataset_root: str) -> Tuple[List[str], List[int]]:
    """Walk the dataset directory and collect image paths + labels."""
    all_paths = []
    all_labels = []

    # Benign = 0, Malignant = 1
    class_map = {"benign": 0, "malignant": 1}

    for class_name, label in class_map.items():
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing directory: {class_dir}")
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            if os.path.isfile(fpath):
                all_paths.append(fpath)
                all_labels.append(label)

    print(f"  Total images: {len(all_paths)}")
    for cls_name, cls_label in class_map.items():
        count = sum(1 for l in all_labels if l == cls_label)
        print(f"    {cls_name} (label={cls_label}): {count}")

    return all_paths, all_labels


def _count_classes(labels):
    """Count benign (0) and malignant (1) in a label list."""
    labels_np = np.array(labels)
    return {
        "benign": int((labels_np == 0).sum()),
        "malignant": int((labels_np == 1).sum()),
        "total": len(labels_np),
    }


def get_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader,
                                               BalancedEpochSampler, dict]:
    """
    Build train/val/test DataLoaders with stratified split.

    Returns:
        train_loader, val_loader, test_loader, train_sampler, split_info
        (train_sampler is returned so the training loop can call set_epoch)
        (split_info contains per-split class counts)
    """
    dataset_root = _download_dataset()
    all_paths, all_labels = _collect_image_paths(dataset_root)

    # --- Stratified split: 70/15/15 ---
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=cfg.test_size,
        stratify=all_labels,
        random_state=cfg.random_state,
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=cfg.val_size,  # 0.176 of 85% ≈ 15% of total
        stratify=train_val_labels,
        random_state=cfg.random_state,
    )

    print(f"  Split: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    # --- Split info ---
    split_info = {
        "total": _count_classes(all_labels),
        "train": _count_classes(train_labels),
        "val": _count_classes(val_labels),
        "test": _count_classes(test_labels),
    }

    # --- Build datasets ---
    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)

    train_dataset = ThyroidDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ThyroidDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = ThyroidDataset(test_paths, test_labels, transform=val_transform)

    # --- Balanced sampler for training ---
    train_sampler = BalancedEpochSampler(train_labels, seed=cfg.random_state)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_sampler, split_info
