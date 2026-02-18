from torch import nn
import torchvision.models as models


class ResNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Replace first conv: ResNet expects 3-channel input; adapt to 1-channel grayscale
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace final FC: binary classification
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        self.model = backbone

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def get_param_groups(self, head_lr: float, backbone_lr: float) -> list:
        backbone_params = [p for name, p in self.model.named_parameters() if not name.startswith("fc")]
        head_params = list(self.model.fc.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
