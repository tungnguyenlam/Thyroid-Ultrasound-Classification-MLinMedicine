"""
models/densenet/model.py
DenseNet-121 for Thyroid Ultrasound Binary Classification.

Architecture overview
---------------------
Backbone : torchvision DenseNet-121 (pretrained on ImageNet by default)
           features -> AdaptiveAvgPool2d -> flatten -> dropout -> Linear head
Head     : Linear(1024 -> num_classes) — replaces the original classifier

DenseNet-121 does not have a standard nn.Linear as its classifier; the final
layer is ``model.classifier`` (Linear(1024, 1000)).  We replace this with a
new Linear(1024, num_classes).

Fine-tuning modes
-----------------
freeze_backbone=True  ->  only the final classifier is trained (linear probe).
freeze_backbone=False ->  all weights are updated (full fine-tune), default.

Usage
-----
    from models.densenet.model import DenseNet121Classifier, build_densenet

    # From a config dict (matches config/config.yaml layout)
    model = build_densenet(cfg)

    # Or directly
    model = DenseNet121Classifier(num_classes=2, pretrained=True)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


__all__ = ["DenseNet121Classifier", "build_densenet"]


class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 binary (or multi-class) classifier.

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

        weights = tvm.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = tvm.densenet121(weights=weights)

        # DenseNet-121 feature extractor (everything before the classifier)
        self.features = backbone.features                # nn.Sequential

        # The adaptive pool + flatten are explicit so we can insert dropout
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        in_features: int = backbone.classifier.in_features  # 1024

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
        feat = self.features(x)           # (B, 1024, H', W')
        feat = self.pool(feat)            # (B, 1024, 1, 1)
        feat = feat.flatten(1)            # (B, 1024)
        feat = self.dropout(feat)
        return self.head(feat)            # (B, num_classes)

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

def build_densenet(cfg: dict) -> DenseNet121Classifier:
    """
    Instantiate DenseNet121Classifier from a config dict.

    Reads:
        cfg["data"]["num_classes"]         (int,   default 2)
        cfg["model"].get("pretrained")     (bool,  default True)
        cfg["model"].get("freeze_backbone")(bool,  default False)
        cfg["model"].get("dropout")        (float, default 0.0)

    Args:
        cfg: Parsed YAML config (from ``src.utils.load_config``).

    Returns:
        Fully constructed DenseNet121Classifier.
    """
    num_classes     = int(cfg["data"].get("num_classes", 2))
    model_cfg       = cfg.get("model", {})
    pretrained      = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", False))
    dropout         = float(model_cfg.get("dropout", 0.0))

    return DenseNet121Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
#  Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = DenseNet121Classifier(num_classes=2, pretrained=False)
    dummy = torch.randn(4, 3, 512, 512)
    out   = model(dummy)
    info  = model.count_parameters()
    print(f"DenseNet-121  |  output shape: {out.shape}  |  "
          f"total params: {info['total']:,}  |  "
          f"trainable: {info['trainable']:,}")
