"""ResNet50 adapted for CIFAR-10, trained from scratch.

Written from scratch rather than reusing torchvision.models.resnet50: its
Bottleneck.forward does `out += identity`, an in-place op on the output of
a hooked BatchNorm2d module, which crashes AdaFisherBackbone's full
backward hooks. Full control here guarantees hook-safety by construction.

Stem is a single 3x3 stride-1 conv with no initial maxpool (the standard
"CIFAR ResNet" stem) instead of the ImageNet 7x7 stride-2 conv + maxpool,
which would collapse a 32x32 image to 8x8 before the first residual block.
"""

from typing import Optional, Tuple

import torch.nn as nn

_DEFAULT_LAYERS: Tuple[int, int, int, int] = (3, 4, 6, 3)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=False)

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion, kernel_size=1, stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        layers: Tuple[int, int, int, int] = _DEFAULT_LAYERS,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)

        self._in_channels = 64
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        blocks = [Bottleneck(self._in_channels, out_channels, stride=stride)]
        self._in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, num_blocks):
            blocks.append(Bottleneck(self._in_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def build_resnet50_cifar(
    num_classes: int = 10,
    layers: Tuple[int, int, int, int] = _DEFAULT_LAYERS,
) -> ResNetCIFAR:
    return ResNetCIFAR(num_classes=num_classes, layers=layers)
