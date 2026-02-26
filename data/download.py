"""
data/download.py
Downloads the Thyroid Ultrasound dataset from Kaggle using kagglehub
and saves it directly into data/.

Dataset:  sowmyaabirami/thyroid-ultrasound-dataset
Classes:  benign | malignant

Usage:
    python data/download.py

Requirements:
    pip install kagglehub
    Kaggle API credentials in ~/.kaggle/kaggle.json
"""

import os
import shutil
import kagglehub


def download_dataset(dest_dir: str = "data/") -> None:
    """Download and move the dataset directly into dest_dir."""
    print("Downloading Thyroid Ultrasound dataset from Kaggle ...")

    cached_path = kagglehub.dataset_download("sowmyaabirami/thyroid-ultrasound-dataset")
    print(f"Cached at: {cached_path}")

    os.makedirs(dest_dir, exist_ok=True)

    for item in os.listdir(cached_path):
        src = os.path.join(cached_path, item)
        dst = os.path.join(dest_dir, item)
        if os.path.exists(dst):
            print(f"  Already exists, skipping: {dst}")
            continue
        shutil.move(src, dst)
        print(f"  Moved: {item} -> {dst}")

    print(f"\nDataset ready in '{dest_dir}'")
    print("Expected layout:")
    print("  data/")
    print("  ├── benign/")
    print("  └── malignant/")


if __name__ == "__main__":
    download_dataset()
