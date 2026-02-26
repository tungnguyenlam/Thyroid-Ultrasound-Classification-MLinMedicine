"""
src/dataloader.py
Thyroid Ultrasound dataset — PyTorch DataLoader with a fixed seeded split.

Expects ImageFolder layout under data_dir:
    data/
    ├── benign/
    └── malignant/

The train / val / test split is deterministic: given the same dataset and
seed the same images always end up in the same split, regardless of run order.
"""

from typing import Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


###############################################################################
#  Transforms
###############################################################################

def get_transforms(img_size: int, split: str) -> transforms.Compose:
    """Train split uses augmentation; val/test use resize + normalise only."""
    mean = [0.485, 0.456, 0.406]   # ImageNet stats
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


###############################################################################
#  Dataset
###############################################################################

class ThyroidDataset(Dataset):
    """
    Wrapper around torchvision.datasets.ImageFolder that allows
    swapping the transform per split.
    """

    def __init__(self, root: str, transform=None):
        self.dataset = datasets.ImageFolder(root=root)
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        path, label = self.dataset.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


###############################################################################
#  Fixed seeded split → DataLoaders
###############################################################################

def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders with a fixed seeded stratified split.

    Given the same dataset + seed, split indices are identical across every run.

    Args:
        cfg: parsed config dict from config/config.yaml

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]

    data_dir    = data_cfg["data_dir"]
    img_size    = int(data_cfg.get("img_size", 224))
    seed        = int(data_cfg.get("seed", 42))
    train_ratio = float(data_cfg.get("train_ratio", 0.70))
    val_ratio   = float(data_cfg.get("val_ratio",   0.15))
    batch_size  = int(train_cfg.get("batch_size", 32))

    # Load full dataset (no transform yet) to get labels for stratification
    base_ds = ThyroidDataset(root=data_dir)
    labels  = np.array([label for _, label in base_ds.dataset.samples])
    indices = np.arange(len(base_ds))

    # Split 1: train vs (val + test)
    train_idx, valtest_idx = train_test_split(
        indices,
        test_size=1.0 - train_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Split 2: val vs test from the remaining pool
    val_size_relative = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=1.0 - val_size_relative,
        stratify=labels[valtest_idx],
        random_state=seed,
    )

    # Build a fresh dataset per split (transforms differ)
    train_ds = ThyroidDataset(root=data_dir, transform=get_transforms(img_size, "train"))
    val_ds   = ThyroidDataset(root=data_dir, transform=get_transforms(img_size, "val"))
    test_ds  = ThyroidDataset(root=data_dir, transform=get_transforms(img_size, "test"))

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds,   val_idx),   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(Subset(test_ds,  test_idx),  batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"[DataLoader] Split — train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}")
    print(f"[DataLoader] Classes: {base_ds.classes}")

    return train_loader, val_loader, test_loader
