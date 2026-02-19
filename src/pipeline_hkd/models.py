"""
Model architectures: EfficientNet-B0, ResNet50, DenseNet121.
All use ImageNet pretrained weights with a custom binary classification head.
"""

import torch.nn as nn
from torchvision import models


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 with custom binary classification head."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)

    def get_feature_layer(self):
        """Return the last convolutional layer for Grad-CAM."""
        return self.backbone.features[-1]


class ResNetClassifier(nn.Module):
    """ResNet50 with custom binary classification head."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)

    def get_feature_layer(self):
        """Return the last convolutional layer for Grad-CAM."""
        return self.backbone.layer4[-1]


class DenseNetClassifier(nn.Module):
    """DenseNet121 with custom binary classification head."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)

    def get_feature_layer(self):
        """Return the last convolutional layer for Grad-CAM."""
        return self.backbone.features.denseblock4


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "efficientnet": EfficientNetClassifier,
    "resnet50": ResNetClassifier,
    "densenet": DenseNetClassifier,
}


def get_model(name: str, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        name: One of 'efficientnet', 'resnet50', 'densenet'.
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate before the final FC layer.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[name](pretrained=pretrained, dropout=dropout)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {name} | params={n_params:,} | trainable={n_trainable:,}")
    return model
