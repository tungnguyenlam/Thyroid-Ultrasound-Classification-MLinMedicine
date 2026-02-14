
from torch import nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Simple2DConvNN(nn.Module):
    def __init__(self, input_shape=(1, 500, 700)):
        super().__init__()
        self.conv1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 16)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()

        self.flattened_size = self._get_flatten_size(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def _get_flatten_size(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            output = self.conv1(dummy)
            output = self.pool1(output)
            output = self.conv2(output)
            output = self.pool2(output)
            return output.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits
    
