"""
Grad-CAM visualization for model interpretability.
Shows which regions of the ultrasound image the model focuses on.

Usage:
    cd src
    python -m pipeline_hkd.gradcam --model efficientnet --checkpoint checkpoints/best_efficientnet.pt --num-images 10
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from .config import Config
from .dataset import get_dataloaders, get_val_transforms
from .models import get_model
from .utils import set_seed, get_device, load_checkpoint


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) input image tensor
            target_class: Not used for binary, kept for API consistency

        Returns:
            cam: (H, W) numpy array, values in [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad()
        output.backward()

        # Global average pool the gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def overlay_heatmap(image_np, cam, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        image_np: (H, W, 3) numpy array in [0, 1]
        cam: (h, w) numpy array in [0, 1]
        alpha: overlay transparency

    Returns:
        overlaid: (H, W, 3) numpy array
    """
    import cv2

    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = heatmap[:, :, ::-1] / 255.0  # BGR → RGB, normalize

    overlaid = (1 - alpha) * image_np + alpha * heatmap
    overlaid = np.clip(overlaid, 0, 1)
    return overlaid


def visualize_gradcam(cfg: Config, checkpoint_path: str, num_images: int = 10):
    """Generate and save Grad-CAM visualizations."""

    set_seed(cfg.seed)
    device = get_device()

    save_dir = os.path.join(cfg.paths.results_dir, "gradcam")
    os.makedirs(save_dir, exist_ok=True)

    # --- Data ---
    print("\n=== Loading Data ===")
    _, _, test_loader, _, _ = get_dataloaders(cfg.data)
    test_dataset = test_loader.dataset

    # --- Model ---
    print("\n=== Loading Model ===")
    model = get_model(cfg.model.name, pretrained=False, dropout=cfg.model.dropout)
    model = model.to(device)
    load_checkpoint(model, checkpoint_path, device)

    # --- Grad-CAM ---
    target_layer = model.get_feature_layer()
    gradcam = GradCAM(model, target_layer)

    # Transform for display (no normalization)
    display_transform = transforms.Compose([
        transforms.Resize(cfg.data.image_size + 32),
        transforms.CenterCrop(cfg.data.image_size),
        transforms.ToTensor(),
    ])

    print(f"\n=== Generating Grad-CAM for {num_images} images ===")

    n = min(num_images, len(test_dataset))
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    class_names = ["Benign", "Malignant"]

    for i in range(n):
        # Get image
        img_path = test_dataset.image_paths[i]
        true_label = test_dataset.labels[i]
        image = Image.open(img_path).convert("RGB")

        # Forward pass with Grad-CAM
        input_tensor = get_val_transforms(cfg.data)(image).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)
        cam = gradcam.generate(input_tensor)

        # Display image (no normalization)
        display_img = display_transform(image).permute(1, 2, 0).numpy()

        # Prediction
        with torch.no_grad():
            logit = model(get_val_transforms(cfg.data)(image).unsqueeze(0).to(device))
            prob = torch.sigmoid(logit).item()
            pred = int(prob >= 0.5)

        # Overlay
        overlaid = overlay_heatmap(display_img, cam)

        # Plot
        axes[i, 0].imshow(display_img)
        axes[i, 0].set_title(f"Original\nTrue: {class_names[true_label]}", fontsize=11)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cam, cmap="jet")
        axes[i, 1].set_title(f"Grad-CAM Heatmap", fontsize=11)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlaid)
        axes[i, 2].set_title(
            f"Overlay\nPred: {class_names[pred]} ({prob:.2f})",
            fontsize=11,
        )
        axes[i, 2].axis("off")

    plt.suptitle(f"Grad-CAM — {cfg.model.name}", fontsize=16, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{cfg.model.name}_gradcam.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    print("\n=== Grad-CAM Complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--model", type=str, required=True,
                        choices=["efficientnet", "resnet50", "densenet"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.model.name = args.model
    cfg.seed = args.seed

    visualize_gradcam(cfg, args.checkpoint, args.num_images)


if __name__ == "__main__":
    main()
