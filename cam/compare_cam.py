"""
cam/compare_cam.py
==================
Side-by-side comparison of Grad-CAM vs Hi-Res CAM using the 5-fold ensemble.

Layout
------
    10 rows  x  3 columns
    col 0 : original image
    col 1 : Grad-CAM overlay
    col 2 : Hi-Res CAM overlay

Each row is one test-set image chosen to give 5 benign + 5 malignant examples
(or as balanced as possible given the test set size).

Output
------
    cam/compare_cam.png        — main comparison grid
    cam/compare_cam_<model>.png — same, model name in filename

Usage (run from project root)
------------------------------
    python cam/compare_cam.py
    python cam/compare_cam.py --config config/config.yaml \\
                               --n-images 10 \\
                               --target-class 1 \\
                               --alpha 0.45 \\
                               --out cam/compare_cam.png
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train import build_model
from src.utils import load_config, set_seed


# ===========================================================================
#  Target-layer resolver  (same as individual scripts)
# ===========================================================================

def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    name = model_name.lower()
    if name in ("res18", "res50"):
        return model.layer4
    elif name == "densenet":
        return model.features.denseblock4
    elif name == "efficientnet":
        return model.features[-1]
    else:
        raise ValueError(f"Unknown model name '{model_name}'.")


# ===========================================================================
#  Hook-based GradCAM
# ===========================================================================

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._acts = self._grads = None
        self._fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_acts", o.detach()))
        self._bh = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0].detach()))

    def __call__(self, x: torch.Tensor, cls: Optional[int] = None):
        self.model.zero_grad()
        t = x.clone().requires_grad_(True)
        logits = self.model(t)
        cls = cls if cls is not None else int(logits.argmax(1).item())
        logits[0, cls].backward()
        w   = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self._acts).sum(1)).squeeze().cpu().numpy().astype(np.float32)
        if cam.max() > 0:
            cam /= cam.max()
        return cam, cls

    def remove(self):
        self._fh.remove(); self._bh.remove()


# ===========================================================================
#  Hook-based HiResCAM
# ===========================================================================

class HiResCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._acts = self._grads = None
        self._fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_acts", o.detach()))
        self._bh = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0].detach()))

    def __call__(self, x: torch.Tensor, cls: Optional[int] = None):
        self.model.zero_grad()
        t = x.clone().requires_grad_(True)
        logits = self.model(t)
        cls = cls if cls is not None else int(logits.argmax(1).item())
        logits[0, cls].backward()
        cam = torch.relu((self._grads * self._acts).sum(1)).squeeze().cpu().numpy().astype(np.float32)
        if cam.max() > 0:
            cam /= cam.max()
        return cam, cls

    def remove(self):
        self._fh.remove(); self._bh.remove()


# ===========================================================================
#  Helpers
# ===========================================================================

def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w   = image.shape[:2]
    cam_u8 = (cv2.resize(cam, (w, h)) * 255).astype(np.uint8)
    heat   = cv2.cvtColor(cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * image).astype(np.uint8)


def get_test_paths_labels(cfg: dict):
    from sklearn.model_selection import train_test_split
    import torchvision.datasets as tvd

    full_ds = tvd.ImageFolder(root=cfg["data"]["data_dir"])
    labels  = np.array([l for _, l in full_ds.samples])
    idx     = np.arange(len(full_ds))
    seed    = int(cfg["data"].get("seed", 42))
    test_r  = 1 - float(cfg["data"].get("train_ratio", 0.70)) - float(cfg["data"].get("val_ratio", 0.15))

    _, test_idx = train_test_split(idx, test_size=test_r, stratify=labels, random_state=seed)
    paths = [full_ds.samples[i][0] for i in test_idx]
    lbls  = [full_ds.samples[i][1] for i in test_idx]
    return paths, lbls, full_ds.classes


def pick_balanced(paths, labels, n: int):
    """Pick ceil(n/2) from each class so the grid is balanced."""
    per_class = (n + 1) // 2
    chosen = []
    counts = {}
    for p, l in zip(paths, labels):
        counts[l] = counts.get(l, 0)
        if counts[l] < per_class and len(chosen) < n:
            chosen.append((p, l))
            counts[l] += 1
    return chosen


def ensemble_cam(models, cam_cls, input_t, model_name, target_cls):
    cam_sum = None
    used_cls = None
    for m in models:
        tgt   = get_target_layer(m, model_name)
        c_obj = cam_cls(m, tgt)
        cam_k, cls_k = c_obj(input_t.clone(), target_cls)
        c_obj.remove()
        if used_cls is None:
            used_cls = cls_k
        if cam_k.max() > 0:
            cam_k /= cam_k.max()
        cam_sum = cam_k if cam_sum is None else cam_sum + cam_k
    avg = cam_sum / len(models)
    if avg.max() > 0:
        avg /= avg.max()
    return avg, used_cls


# ===========================================================================
#  Main
# ===========================================================================

def compare_cam(
    config_path : str           = "config/config.yaml",
    n_folds     : int           = 5,
    n_images    : int           = 10,
    target_class: Optional[int] = None,
    alpha       : float         = 0.45,
    out_path    : str           = "cam/compare_cam.png",
) -> None:
    cfg         = load_config(config_path)
    set_seed(cfg["data"]["seed"])

    model_name  = cfg["model"]["name"]
    img_size    = int(cfg["data"].get("img_size", 512))
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[compare_cam] model={model_name} | device={device} | images={n_images}")

    # Test set
    paths, label_ints, classes = get_test_paths_labels(cfg)
    selected = pick_balanced(paths, label_ints, n_images)
    print(f"[compare_cam] Selected {len(selected)} images "
          f"({sum(l==0 for _,l in selected)} benign, "
          f"{sum(l==1 for _,l in selected)} malignant)")

    # Pre-processing
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load all fold models once
    models: List[nn.Module] = []
    for k in range(1, n_folds + 1):
        ckpt = os.path.join("checkpoints", model_name, f"fold{k}", "best.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        m = build_model(cfg).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        models.append(m)
    print(f"[compare_cam] Loaded {n_folds} fold checkpoints\n")

    # Build figure  —  n_rows x 3 cols
    n_rows = len(selected)

    # constrained_layout=True lets matplotlib pack titles/labels tightly
    # with no manual blank gaps.
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(16, 4.0 * n_rows),
        constrained_layout=True,
    )
    # Ensure axes is always 2-D (n_rows==1 gives a 1-D array)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles  = ["Original", "Grad-CAM", "Hi-Res CAM"]
    # Annotation colours: green for benign, red for malignant
    cls_color   = {0: "#27ae60", 1: "#e74c3c"}
    cls_label   = {0: "Benign",  1: "Malignant"}

    for row, (img_path, true_lbl) in enumerate(
        tqdm(selected, desc="[compare_cam]", unit="img")
    ):
        pil_img  = Image.open(img_path).convert("RGB")
        orig_arr = np.array(pil_img.resize((img_size, img_size)))
        input_t  = preprocess(pil_img).unsqueeze(0).to(device)

        # Ensemble Grad-CAM
        gcam_avg, used_cls = ensemble_cam(models, GradCAM,  input_t, model_name, target_class)
        # Ensemble Hi-Res CAM
        hcam_avg, _        = ensemble_cam(models, HiResCAM, input_t, model_name, target_class)

        gcam_ov = overlay(orig_arr, gcam_avg, alpha)
        hcam_ov = overlay(orig_arr, hcam_avg, alpha)

        panels = [orig_arr, gcam_ov, hcam_ov]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.axis("off")

            # Column headers on the first row only
            if row == 0:
                ax.set_title(col_titles[col], fontsize=15, fontweight="bold", pad=5)

            # Class badge — top-left corner of every panel
            ax.text(
                0.03, 0.97,
                cls_label[true_lbl],
                transform=ax.transAxes,
                fontsize=9, fontweight="bold", color="white",
                va="top", ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc=cls_color[true_lbl],
                    alpha=0.88,
                    ec="none",
                ),
            )

    fig.suptitle(
        f"Grad-CAM vs Hi-Res CAM  |  model: {model_name}  |  {n_folds}-fold ensemble",
        fontsize=13,
    )

    # Save with model name embedded in the filename
    out_stem  = Path(out_path)
    out_model = out_stem.parent / f"{out_stem.stem}_{model_name}{out_stem.suffix}"
    os.makedirs(out_stem.parent, exist_ok=True)
    fig.savefig(str(out_model), dpi=150)
    plt.close(fig)

    print(f"\n[compare_cam] Saved -> {out_model}")


# ===========================================================================
#  CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Grad-CAM vs Hi-Res CAM in a single grid figure.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config",       type=str,   default="config/config.yaml")
    parser.add_argument("--n-folds",      type=int,   default=5)
    parser.add_argument("--n-images",     type=int,   default=10,
                        help="Number of test images to show (default: 10)")
    parser.add_argument("--target-class", type=int,   default=None,
                        help="0=benign, 1=malignant  [default: predicted class]")
    parser.add_argument("--alpha",        type=float, default=0.45,
                        help="Overlay opacity [0,1] (default: 0.45)")
    parser.add_argument("--out",          type=str,   default="cam/compare_cam.png",
                        help="Output PNG path (default: cam/compare_cam.png)")
    args = parser.parse_args()
    compare_cam(
        config_path  = args.config,
        n_folds      = args.n_folds,
        n_images     = args.n_images,
        target_class = args.target_class,
        alpha        = args.alpha,
        out_path     = args.out,
    )
