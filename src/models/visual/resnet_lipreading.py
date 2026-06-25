"""Lip-reading ResNet trunk (2D), ported to match the pretrained LRW checkpoint.

This is a faithful port of the ResNet trunk from
``mpc001/Lipreading_using_Temporal_Convolutional_Networks`` (``lipreading/models/resnet.py``),
kept byte-compatible at the ``state_dict`` level so the published ``resnet18`` lip-reading
weights load by key. Differences from torchvision ResNet18:

- NO stem (``conv1``/``bn1``/``maxpool``): the spatial+temporal stem is the 3D frontend
  (see :class:`LipReadingFrontend`). This trunk starts at ``layer1`` with 64 input channels.
- Per-block activation is selectable (``relu`` / ``prelu`` / ``swish``); ``prelu`` adds
  per-channel weight params, so ``relu_type`` MUST match the checkpoint.
- ``AdaptiveAvgPool2d(1)`` → 512-D per frame.

Upstream license is research/non-commercial (comparative/benchmark use only); this thesis
use complies. Cite the source repo. See plan ``recursive-hatching-haven.md`` (Phase A1).
"""
import math
from typing import List, Type

import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def downsample_basic_block(inplanes: int, outplanes: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def downsample_basic_block_v2(inplanes: int, outplanes: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def _make_activation(relu_type: str, num_parameters: int) -> nn.Module:
    """Activation matching the upstream choice. ``prelu`` is per-channel (adds params)."""
    if relu_type == "relu":
        return nn.ReLU(inplace=True)
    if relu_type == "prelu":
        return nn.PReLU(num_parameters=num_parameters)
    if relu_type == "swish":
        return nn.SiLU(inplace=True)  # x * sigmoid(x); paramless, == upstream Swish
    raise ValueError(f"relu type not implemented: {relu_type}")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        relu_type: str = "prelu",
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = _make_activation(relu_type, planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = _make_activation(relu_type, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    """Stem-less ResNet trunk (layer1..4 + AdaptiveAvgPool2d) → 512-D per frame.

    Input is the 3D-frontend feature map ``[N, 64, H', W']`` (N = batch*time).
    """

    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        relu_type: str = "prelu",
        gamma_zero: bool = False,
        avg_pool_downsample: bool = False,
    ):
        super().__init__()
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes, outplanes=planes * block.expansion, stride=stride
            )
        layers = [block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)
