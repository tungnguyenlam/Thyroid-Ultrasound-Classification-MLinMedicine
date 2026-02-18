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
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 1),
        )

        self.model = backbone

    def forward(self, x):
        return self.model(x)
