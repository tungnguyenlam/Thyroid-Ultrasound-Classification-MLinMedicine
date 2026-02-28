"""
models/loss.py
General-purpose loss functions for image classification.

All losses work for any number of classes (binary OR multi-class) and are
consumed through a single factory function ``get_loss_fn`` that reads from
the project's config dict.

Supported loss types (set via config key  training.loss ):
  + "cross_entropy"            - standard PyTorch CrossEntropyLoss
  + "weighted_cross_entropy"   - CrossEntropyLoss with per-class weights
  + "label_smoothing"          - CrossEntropyLoss with label smoothing
  + "focal"                    - Focal Loss  (Lin et al., 2017)
  + "focal_ce"                 - α · FocalLoss + (1-α) · CrossEntropyLoss

Config snippet (config/config.yaml):
  training:
    loss: "focal"             # one of the keys above  [default: cross_entropy]
    loss_gamma: 2.0           # focal loss γ            [default: 2.0]
    loss_alpha: 0.25          # focal / focal_ce blend α [default: 0.25]
    loss_smoothing: 0.1       # label smoothing ε        [default: 0.1]
    loss_blend: 0.5           # focal_ce blend ratio     [default: 0.5]
    class_weights: null       # list of floats, one per class, or null

Usage:
  from models.loss import get_loss_fn
  criterion = get_loss_fn(cfg, device)
  loss = criterion(logits, targets)   # logits: (B, C), targets: (B,) long
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "CrossEntropyLoss",
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "FocalCELoss",
    "get_loss_fn",
]


#==============================================================================
#  1. Standard Cross-Entropy
#==============================================================================

class CrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy loss.

    Wraps ``torch.nn.CrossEntropyLoss`` so it has the same interface as the
    other loss classes in this module.

    Args:
        weight:     Optional 1-D tensor of per-class weights (length == num_classes).
        reduction:  'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw, un-normalised class scores.
            targets: (B,)   integer class indices in [0, C).
        Returns:
            Scalar loss value (or per-sample tensor when reduction='none').
        """
        return self.loss_fn(logits, targets)


#==============================================================================
#  2. Focal Loss
#==============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for dense classification (Lin et al., 2017).

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    This implementation works for arbitrary number of classes (including
    binary classification as a 2-class problem).

    Args:
        gamma:     Focusing exponent γ >= 0. γ=0 recovers cross-entropy.
        alpha:     Scalar weighting factor α ∈ [0, 1].  Set to None to
                   disable per-class weighting.
        weight:    Optional per-class weight tensor (overrides ``alpha``
                   when both are provided, combining multiplicatively).
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("weight", weight)          # may be None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw class scores.
            targets: (B,)   integer class indices.
        Returns:
            Focal loss scalar.
        """
        # log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=1)                  # (B, C)
        probs     = log_probs.exp()                                # (B, C)

        # Gather the probability assigned to the *true* class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt     = log_pt.exp()                                           # (B,)

        # Focal modulation factor
        focal_weight = (1.0 - pt) ** self.gamma                        # (B,)

        # Per-class weight (from weight buffer)
        if self.weight is not None:
            class_w = self.weight[targets]                              # (B,)
            focal_weight = focal_weight * class_w

        # Scalar alpha weighting
        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha

        loss = -focal_weight * log_pt                                   # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss   # 'none'


#==============================================================================
#  3. Label-Smoothing Cross-Entropy
#==============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing (Szegedy et al., 2016).

    Shifts probability mass from the true class to a uniform distribution
    over all classes:

        q(k) = (1 - ε) * δ(k == y) + ε / C

    where ε is the smoothing factor and C the number of classes.

    Args:
        smoothing:  ε ∈ [0, 1).  0 -> ordinary cross-entropy.
        weight:     Optional per-class weight tensor.
        reduction:  'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "smoothing must be in [0, 1)"
        self.smoothing = smoothing
        self.register_buffer("weight", weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw class scores.
            targets: (B,)   integer class indices.
        Returns:
            Label-smoothed cross-entropy loss.
        """
        num_classes = logits.size(1)
        log_probs   = F.log_softmax(logits, dim=1)               # (B, C)

        # Build smooth target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / num_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Per-sample loss:  -Σ_k  q(k) · log p(k)
        loss = -(smooth_targets * log_probs).sum(dim=1)           # (B,)

        # Optional per-class weighting (applied by true class)
        if self.weight is not None:
            class_w = self.weight[targets]                        # (B,)
            loss    = loss * class_w

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss   # 'none'


#==============================================================================
#  4. Focal + Cross-Entropy Blend
#==============================================================================

class FocalCELoss(nn.Module):
    """
    Convex combination of Focal Loss and Cross-Entropy:

        L = blend · FL + (1 - blend) · CE

    This is useful when you want the hard-example focusing effect of focal
    loss without discarding the calibration benefits of plain CE.

    Args:
        gamma:     Focal loss γ.
        alpha:     Focal loss α (scalar).
        blend:     Weight given to the focal component (0 -> pure CE,
                   1 -> pure focal).
        weight:    Optional per-class weight tensor (applied to both terms).
        smoothing: Label-smoothing ε for the CE component. Set to 0 to
                   use standard CE.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        blend: float = 0.5,
        weight: Optional[torch.Tensor] = None,
        smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.blend = blend
        self.focal = FocalLoss(
            gamma=gamma, alpha=alpha, weight=weight, reduction=reduction
        )
        if smoothing > 0.0:
            self.ce = LabelSmoothingCrossEntropy(
                smoothing=smoothing, weight=weight, reduction=reduction
            )
        else:
            self.ce = CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        fl = self.focal(logits, targets)
        ce = self.ce(logits, targets)
        return self.blend * fl + (1.0 - self.blend) * ce


#==============================================================================
#  5. Factory function
#==============================================================================

def get_loss_fn(
    cfg: dict,
    device: torch.device,
    num_classes: Optional[int] = None,
) -> nn.Module:
    """
    Build and return the loss function specified in ``cfg['training']['loss']``.

    Config keys consulted (all under ``training:``):
        loss              (str)   Name of the loss. Default: "cross_entropy".
        loss_gamma        (float) Focal γ.            Default: 2.0
        loss_alpha        (float) Focal α.            Default: 0.25
        loss_smoothing    (float) LS ε.               Default: 0.1
        loss_blend        (float) FocalCE blend.      Default: 0.5
        class_weights     (list)  Per-class floats.   Default: null / None

    Args:
        cfg:         Parsed config dict (from ``load_config()``).
        device:      torch.device to place weight tensors on.
        num_classes: Override for the number of classes.  Falls back to
                     ``cfg['data']['num_classes']`` if not supplied.

    Returns:
        An ``nn.Module`` with signature ``forward(logits, targets) -> Tensor``.

    Example::

        from models.loss import get_loss_fn
        criterion = get_loss_fn(cfg, device=torch.device("cuda"))
        loss = criterion(logits, targets)
    """
    train_cfg = cfg.get("training", {})
    data_cfg  = cfg.get("data", {})

    loss_name  = train_cfg.get("loss", "cross_entropy").lower().strip()
    gamma      = float(train_cfg.get("loss_gamma",     2.0))
    alpha      = float(train_cfg.get("loss_alpha",     0.25))
    smoothing  = float(train_cfg.get("loss_smoothing", 0.1))
    blend      = float(train_cfg.get("loss_blend",     0.5))

    if num_classes is None:
        num_classes = int(data_cfg.get("num_classes", 2))

    # Class Weights
    raw_weights: Optional[List[float]] = train_cfg.get("class_weights", None)
    weight_tensor: Optional[torch.Tensor] = None
    if raw_weights is not None:
        if len(raw_weights) != num_classes:
            raise ValueError(
                f"class_weights has {len(raw_weights)} entries but "
                f"num_classes={num_classes}."
            )
        weight_tensor = torch.tensor(raw_weights, dtype=torch.float32, device=device)

    # Build Loss
    if loss_name == "cross_entropy":
        return CrossEntropyLoss(weight=weight_tensor).to(device)

    elif loss_name == "weighted_cross_entropy":
        if weight_tensor is None:
            raise ValueError(
                "loss='weighted_cross_entropy' requires 'class_weights' in config."
            )
        return CrossEntropyLoss(weight=weight_tensor).to(device)

    elif loss_name == "label_smoothing":
        return LabelSmoothingCrossEntropy(
            smoothing=smoothing,
            weight=weight_tensor,
        ).to(device)

    elif loss_name == "focal":
        return FocalLoss(
            gamma=gamma,
            alpha=alpha,
            weight=weight_tensor,
        ).to(device)

    elif loss_name == "focal_ce":
        return FocalCELoss(
            gamma=gamma,
            alpha=alpha,
            blend=blend,
            weight=weight_tensor,
            smoothing=0.0,
        ).to(device)

    else:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            "Choose from: cross_entropy | weighted_cross_entropy | "
            "label_smoothing | focal | focal_ce"
        )
