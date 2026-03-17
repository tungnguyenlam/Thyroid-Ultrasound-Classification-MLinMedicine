"""
cam/hires_cam/run_hires_cam.py
==============================
Ensemble Hi-Res CAM visualisation for the Thyroid Ultrasound Classification project.

Works with ANY model defined in models/ and registered in build_model():
    res18 | res50 | densenet | efficientnet

What this script does
---------------------
1.  Reads config/config.yaml to know the active model name.
2.  Loads all 5 fold checkpoints (checkpoints/<model>/fold{k}/best.pt).
3.  For every image in the fixed test set it:
        a. Runs a forward + backward pass to get the element-wise product of
           *gradients x activations* at the target layer (HiRes-CAM formula).
        b. Sums over the channel axis and applies ReLU → per-fold heat-map.
        c. Normalises and averages the five heat-maps.
4.  Saves a 3-panel PNG per image under
        cam/hires_cam/outputs/<class>/<stem>_hirescam.png
    showing: original | Hi-Res CAM overlay | stand-alone heatmap.

Key difference from Grad-CAM
------------------------------
Grad-CAM   : weights = global_avg_pool(gradients)     (one scalar per channel)
             cam     = ReLU( Σ_c  weights_c x A_c )

HiRes-CAM  : cam     = ReLU( Σ_c  grad_c x A_c  )    (element-wise, spatially aware)

HiRes-CAM preserves more fine-grained spatial information because the gradient
is multiplied element-wise *before* summing, not reduced to a scalar first.

Target layer (selected automatically per architecture)
------------------------------------------------------
    res18       → model.layer4
    res50       → model.layer4
    densenet    → model.features.denseblock4
    efficientnet→ model.features[-1]

Usage (run from the project root)
----------------------------------
    python cam/hires_cam/run_hires_cam.py
    python cam/hires_cam/run_hires_cam.py --config config/config.yaml \\
                                           --target-class 1 \\
                                           --n-images 50 \\
                                           --alpha 0.45
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
# Add project root to sys.path so we can import src.* and models.*
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train import build_model
from src.utils import load_config, set_seed


# ===========================================================================
#  Automatic target-layer resolver  (shared logic with run_grad_cam.py)
# ===========================================================================

def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Return the last convolutional block of *model* based on *model_name*.

        res18 / res50   → model.layer4
        densenet        → model.features.denseblock4
        efficientnet    → model.features[-1]
    """
    name = model_name.lower()
    if name in ("res18", "res50"):
        return model.layer4
    elif name == "densenet":
        return model.features.denseblock4
    elif name == "efficientnet":
        return model.features[-1]
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            "Supported: res18 | res50 | densenet | efficientnet"
        )


# ===========================================================================
#  HiRes-CAM implementation  (pure PyTorch hooks)
# ===========================================================================

class HiResCAM:
    """
    Hook-based HiRes-CAM.

    Unlike Grad-CAM (which pools gradients to a single scalar per channel),
    HiRes-CAM computes the element-wise product  gradient x activation
    before summing over channels, retaining per-spatial-location sensitivity.

    Args:
        model        : PyTorch model in eval mode.
        target_layer : The Conv block to hook (e.g. model.layer4).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        self._fh = target_layer.register_forward_hook(self._save_activation)
        self._bh = target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    def _save_activation(self, _module, _inp, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    # ------------------------------------------------------------------
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Compute one Hi-Res CAM heat-map.

        Args:
            input_tensor : (1, C, H, W) pre-processed image on model's device.
            target_class : Class to explain. None → predicted class.

        Returns:
            cam          : (h, w) float32 heat-map in [0, 1].
            used_class   : The class index that was back-propagated.
        """
        self.model.zero_grad()
        x = input_tensor.clone().requires_grad_(True)
        logits = self.model(x)

        used_class = (
            target_class
            if target_class is not None
            else int(logits.argmax(dim=1).item())
        )

        logits[0, used_class].backward()

        # HiRes-CAM: element-wise product then sum over channels
        grads = self._gradients    # (1, C, h, w)
        acts  = self._activations  # (1, C, h, w)

        # Element-wise product → (1, C, h, w), then sum over C → (h, w)
        cam = torch.relu((grads * acts).sum(dim=1)).squeeze()   # (h, w)
        cam = cam.cpu().numpy().astype(np.float32)

        if cam.max() > 0:
            cam /= cam.max()

        return cam, used_class

    # ------------------------------------------------------------------
    def remove_hooks(self) -> None:
        self._fh.remove()
        self._bh.remove()


# ===========================================================================
#  Overlay helper
# ===========================================================================

def overlay_heatmap(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a [0,1] heat-map onto an (H, W, 3) uint8 RGB image.

    Returns an (H, W, 3) uint8 RGB array.
    """
    h, w    = image.shape[:2]
    cam_u8  = (cv2.resize(cam, (w, h)) * 255).astype(np.uint8)
    heatmap = cv2.cvtColor(cv2.applyColorMap(cam_u8, colormap), cv2.COLOR_BGR2RGB)
    return (alpha * heatmap + (1.0 - alpha) * image).astype(np.uint8)


# ===========================================================================
#  Reproduce the fixed test split  (identical to src/dataloader.py)
# ===========================================================================

def get_test_paths_labels(cfg: dict) -> tuple[list, list]:
    """
    Reproduce the deterministic test split and return
    (list_of_image_paths, list_of_integer_labels).
    """
    from sklearn.model_selection import train_test_split
    import torchvision.datasets as tvd

    data_dir    = cfg["data"]["data_dir"]
    seed        = int(cfg["data"].get("seed", 42))
    train_ratio = float(cfg["data"].get("train_ratio", 0.70))
    val_ratio   = float(cfg["data"].get("val_ratio",   0.15))
    test_ratio  = 1.0 - train_ratio - val_ratio

    full_ds = tvd.ImageFolder(root=data_dir)
    labels  = np.array([lbl for _, lbl in full_ds.samples])
    indices = np.arange(len(full_ds))

    _, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    paths = [full_ds.samples[i][0] for i in test_idx]
    lbls  = [full_ds.samples[i][1] for i in test_idx]
    return paths, lbls


# ===========================================================================
#  Main
# ===========================================================================

def run_ensemble_hirescam(
    config_path : str           = "config/config.yaml",
    n_folds     : int           = 5,
    target_class: Optional[int] = None,
    n_images    : Optional[int] = None,
    alpha       : float         = 0.45,
    out_dir     : str           = "cam/hires_cam/outputs",
) -> None:
    """
    Compute per-image ensemble HiRes-CAM over the shared test set and save
    3-panel PNGs (original | overlay | heatmap) to *out_dir/<class>/*.

    Args:
        config_path : Path to YAML config.
        n_folds     : Number of CV folds to ensemble.
        target_class: Class to explain (0=benign, 1=malignant, None=predicted).
        n_images    : Cap on number of images (None = whole test set).
        alpha       : Overlay opacity [0, 1].
        out_dir     : Root save directory.
    """
    cfg         = load_config(config_path)
    set_seed(cfg["data"]["seed"])

    model_name  = cfg["model"]["name"]
    img_size    = int(cfg["data"].get("img_size", 512))
    class_names = cfg["data"].get("class_names", ["benign", "malignant"])
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Ensemble Hi-Res CAM  |  model: {model_name}")
    print(f"  Device      : {device}")
    print(f"  Folds       : {n_folds}")
    print(f"  Target class: "
          f"{class_names[target_class] if target_class is not None else 'predicted'}")
    print(f"  Output dir  : {out_dir}")
    print(f"{'='*60}\n")

    # Output sub-dirs per class
    for cls in class_names:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    # Test set image paths (deterministic)
    test_paths, test_labels = get_test_paths_labels(cfg)
    total = len(test_paths) if n_images is None else min(n_images, len(test_paths))
    print(f"[Hi-Res CAM] Test images to process: {total}\n")

    # ImageNet pre-processing (same as dataloader "test" transform)
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Load all fold checkpoints
    models: List[nn.Module] = []
    for k in range(1, n_folds + 1):
        ckpt = os.path.join("checkpoints", model_name, f"fold{k}", "best.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        m = build_model(cfg).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        print(f"  Fold {k}: loaded  {ckpt}")
        models.append(m)

    print()

    # Main visualisation loop
    for idx in tqdm(range(total), desc="[Hi-Res CAM]", unit="img"):
        img_path   = test_paths[idx]
        true_label = test_labels[idx]

        pil_img  = Image.open(img_path).convert("RGB")
        orig_arr = np.array(pil_img.resize((img_size, img_size)))
        input_t  = preprocess(pil_img).unsqueeze(0).to(device)

        # ---- Ensemble CAM -----------------------------------------------
        cam_sum    = None
        used_class = None

        for m in models:
            tgt_layer = get_target_layer(m, model_name)
            hcam      = HiResCAM(m, tgt_layer)
            cam_k, cls_k = hcam(input_t.clone(), target_class=target_class)
            hcam.remove_hooks()

            if used_class is None:
                used_class = cls_k
            if cam_k.max() > 0:
                cam_k = cam_k / cam_k.max()              # per-fold normalise
            cam_sum = cam_k if cam_sum is None else cam_sum + cam_k

        cam_avg = cam_sum / n_folds
        if cam_avg.max() > 0:
            cam_avg /= cam_avg.max()                      # re-normalise ensemble

        # ---- Save figure ------------------------------------------------
        overlay  = overlay_heatmap(orig_arr, cam_avg, alpha=alpha)
        h, w     = img_size, img_size
        cam_full = cv2.resize(cam_avg, (w, h))
        hmap_bgr = cv2.applyColorMap((cam_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hmap_rgb = cv2.cvtColor(hmap_bgr, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Hi-Res CAM  ({model_name}, {n_folds}-fold ensemble)  |  "
            f"True: {class_names[true_label]}  |  "
            f"Explained: {class_names[used_class]}",
            fontsize=11,
        )
        axes[0].imshow(orig_arr);   axes[0].set_title("Original");              axes[0].axis("off")
        axes[1].imshow(overlay);    axes[1].set_title(f"Overlay (α={alpha})");  axes[1].axis("off")
        axes[2].imshow(hmap_rgb);   axes[2].set_title("Heatmap only");          axes[2].axis("off")
        plt.colorbar(
            plt.cm.ScalarMappable(cmap="jet"),
            ax=axes[2], fraction=0.046, pad=0.04,
        )
        plt.tight_layout()

        stem      = Path(img_path).stem
        save_path = os.path.join(out_dir, class_names[true_label], f"{stem}_hirescam.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\n[Hi-Res CAM] Done — {total} images saved under '{out_dir}'")


# ===========================================================================
#  CLI entry-point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Ensemble Hi-Res CAM for Thyroid Ultrasound Classification.\n"
            "Works with any model: res18 | res50 | densenet | efficientnet.\n"
            "Run from the project root."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str, default="config/config.yaml",
        help="Path to YAML config (default: config/config.yaml)",
    )
    parser.add_argument(
        "--n-folds",
        type=int, default=5,
        help="Number of CV fold checkpoints to load (default: 5)",
    )
    parser.add_argument(
        "--target-class",
        type=int, default=None,
        help="Class to visualise: 0=benign, 1=malignant  [default: predicted class]",
    )
    parser.add_argument(
        "--n-images",
        type=int, default=None,
        help="Max images to process (default: entire test set)",
    )
    parser.add_argument(
        "--alpha",
        type=float, default=0.45,
        help="Overlay opacity in [0,1] (default: 0.45)",
    )
    parser.add_argument(
        "--out-dir",
        type=str, default="cam/hires_cam/outputs",
        help="Root output directory (default: cam/hires_cam/outputs)",
    )
    args = parser.parse_args()
    run_ensemble_hirescam(
        config_path  = args.config,
        n_folds      = args.n_folds,
        target_class = args.target_class,
        n_images     = args.n_images,
        alpha        = args.alpha,
        out_dir      = args.out_dir,
    )
