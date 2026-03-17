"""
models/efficientnet/model.py
EfficientNet-B0 for Thyroid Ultrasound Binary Classification.

Architecture overview
---------------------
Backbone : torchvision EfficientNet-B0 (pretrained on ImageNet by default)
Head     : Linear(1280 -> num_classes) — replaces classifier[1] in the original
           Sequential([Dropout(0.2), Linear(1280, 1000)])

The torchvision EfficientNet-B0 exposes:
    model.features   – convolutional backbone
    model.avgpool    – AdaptiveAvgPool2d
    model.classifier – nn.Sequential([Dropout, Linear])

We keep features + avgpool as-is and replace classifier[1] (the Linear layer).

Fine-tuning modes
-----------------
freeze_backbone=True  ->  only the final classifier is trained (linear probe).
freeze_backbone=False ->  all weights are updated (full fine-tune), default.

Usage
-----
    from models.efficientnet.model import EfficientNetB0Classifier, build_efficientnet

    # From a config dict (matches config/config.yaml layout)
    model = build_efficientnet(cfg)

    # Or directly
    model = EfficientNetB0Classifier(num_classes=2, pretrained=True)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


__all__ = ["EfficientNetB0Classifier", "build_efficientnet"]


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 binary (or multi-class) classifier.

    Args:
        num_classes    : Number of output classes (≥ 2).
        pretrained     : Load ImageNet weights when True.
        freeze_backbone: If True, freeze all layers except the final head.
        dropout        : Dropout probability for the head (overrides the
                         default 0.2 in the original classifier block).
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = tvm.efficientnet_b0(weights=weights)

        # Feature extractor + global average pool
        self.features = backbone.features   # nn.Sequential of MBConv blocks
        self.avgpool  = backbone.avgpool    # AdaptiveAvgPool2d(1, 1)

        in_features: int = backbone.classifier[1].in_features  # 1280

        # Custom classifier head (Dropout -> Linear)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

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
        feat = self.features(x)      # (B, 1280, H', W')
        feat = self.avgpool(feat)    # (B, 1280, 1, 1)
        feat = feat.flatten(1)       # (B, 1280)
        return self.head(feat)       # (B, num_classes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        """Freeze all feature-extractor parameters (head remains trainable)."""
        for param in self.features.parameters():
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

def build_efficientnet(cfg: dict) -> EfficientNetB0Classifier:
    """
    Instantiate EfficientNetB0Classifier from a config dict.

    Reads:
        cfg["data"]["num_classes"]         (int,   default 2)
        cfg["model"].get("pretrained")     (bool,  default True)
        cfg["model"].get("freeze_backbone")(bool,  default False)
        cfg["model"].get("dropout")        (float, default 0.2)

    Args:
        cfg: Parsed YAML config (from ``src.utils.load_config``).

    Returns:
        Fully constructed EfficientNetB0Classifier.
    """
    num_classes     = int(cfg["data"].get("num_classes", 2))
    model_cfg       = cfg.get("model", {})
    pretrained      = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", False))
    dropout         = float(model_cfg.get("dropout", 0.2))

    return EfficientNetB0Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
#  Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = EfficientNetB0Classifier(num_classes=2, pretrained=False)
    dummy = torch.randn(4, 3, 512, 512)
    out   = model(dummy)
    info  = model.count_parameters()
    print(f"EfficientNet-B0  |  output shape: {out.shape}  |  "
          f"total params: {info['total']:,}  |  "
          f"trainable: {info['trainable']:,}")
