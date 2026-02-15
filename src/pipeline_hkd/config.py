"""
Centralized configuration for the thyroid ultrasound classification pipeline.
All hyperparameters are defined here using dataclasses.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    test_size: float = 0.15
    val_size: float = 0.176  # 0.176 of remaining 85% ≈ 15% of total
    random_state: int = 42
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    # Scheduler
    scheduler_T0: int = 10
    scheduler_Tmult: int = 2
    # Early stopping
    patience: int = 10
    # Fine-tuning
    freeze_epochs: int = 5
    unfreeze_lr_factor: float = 0.1
    # SWA
    swa_start_epoch: int = -5  # negative means "last N epochs"
    swa_lr: float = 1e-5


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "efficientnet"  # efficientnet | resnet50 | densenet
    dropout: float = 0.3
    pretrained: bool = True


@dataclass
class PathConfig:
    """Output paths."""
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    training_log: str = "results/training_log.csv"


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    seed: int = 42
