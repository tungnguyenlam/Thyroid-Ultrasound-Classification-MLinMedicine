"""
models/res50/model.py
ResNet-50 for Thyroid Ultrasound Binary Classification.

Architecture overview
---------------------
Backbone : torchvision ResNet-50 (pretrained on ImageNet by default)
Head     : single Linear(2048 -> num_classes) — replaces the original fc

Fine-tuning modes
-----------------
freeze_backbone=True  ->  only the final classifier is trained (linear probe).
freeze_backbone=False ->  all weights are updated (full fine-tune), default.

Usage
-----
    from models.res50.model import ResNet50Classifier, build_res50

    # From a config dict (matches config/config.yaml layout)
    model = build_res50(cfg)

    # Or directly
    model = ResNet50Classifier(num_classes=2, pretrained=True)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


__all__ = ["ResNet50Classifier", "build_res50"]


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 binary (or multi-class) classifier.

    Args:
        num_classes    : Number of output classes (≥ 2).
        pretrained     : Load ImageNet weights when True.
        freeze_backbone: If True, freeze all layers except the final head.
        dropout        : Dropout probability inserted before the head (0 = disabled).
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        # Remove the original fully-connected head; keep everything else
        in_features: int = backbone.fc.in_features          # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Optional regularisation before the head
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Classification head
        self.head = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor.
        Returns:
            logits: (B, num_classes) — raw, un-normalised class scores.
        """
        features = self.backbone(x)      # (B, 2048)
        features = self.dropout(features)
        return self.head(features)       # (B, num_classes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters (head remains trainable)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze the entire network (useful for stage-2 fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True

    def count_parameters(self) -> dict:
        """Return trainable and total parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
#  Factory — reads from config/config.yaml dict
# ---------------------------------------------------------------------------

def build_res50(cfg: dict) -> ResNet50Classifier:
    """
    Instantiate ResNet50Classifier from a config dict.

    Reads:
        cfg["data"]["num_classes"]         (int,   default 2)
        cfg["model"].get("pretrained")     (bool,  default True)
        cfg["model"].get("freeze_backbone")(bool,  default False)
        cfg["model"].get("dropout")        (float, default 0.0)

    Args:
        cfg: Parsed YAML config (from ``src.utils.load_config``).

    Returns:
        Fully constructed ResNet50Classifier.
    """
    num_classes     = int(cfg["data"].get("num_classes", 2))
    model_cfg       = cfg.get("model", {})
    pretrained      = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", False))
    dropout         = float(model_cfg.get("dropout", 0.0))

    return ResNet50Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
#  Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = ResNet50Classifier(num_classes=2, pretrained=False)
    dummy = torch.randn(4, 3, 512, 512)
    out   = model(dummy)
    info  = model.count_parameters()
    print(f"ResNet-50  |  output shape: {out.shape}  |  "
          f"total params: {info['total']:,}  |  "
          f"trainable: {info['trainable']:,}")
