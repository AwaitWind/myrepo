"""U-Net 导线分割模型。

论文用的是 "the original U-Net architecture with default settings"，所以默认
就是原版：4 次下采样、双卷积块、转置卷积上采样、跳连拼接。

另提供 resnet18 编码器变体，对应 `赛题六_模型设计.md` §3.0 的
"编码器用 ImageNet 权重"。ResNet18 的 ImageNet 权重本机实测可下
（46.8 MB / 19.8 MB/s），不像 YOLO 和 PaddleOCR 那样卡网络。

输出单通道 logits，不含 sigmoid —— 损失函数里用 BCEWithLogits 更稳。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """原版 U-Net。base=64 是论文口径；显存紧张时可降到 32。"""

    def __init__(self, in_ch: int = 3, base: int = 64, depth: int = 4):
        super().__init__()
        self.depth = depth
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_ch
        for i in range(depth):
            self.downs.append(double_conv(c, chs[i]))
            c = chs[i]
        self.bottleneck = double_conv(c, chs[depth])
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in reversed(range(depth)):
            self.ups.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2))
            self.up_convs.append(double_conv(chs[i] * 2, chs[i]))
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for block in self.downs:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = conv(torch.cat([skip, x], dim=1))
        return self.head(x)


class ResNet18UNet(nn.Module):
    """ResNet18 编码器 + U-Net 解码器。pretrained=True 时会联网下载权重。"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=w)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)  # /2, 64
        self.pool = net.maxpool                                   # /4
        self.layer1, self.layer2 = net.layer1, net.layer2         # /4 64, /8 128
        self.layer3, self.layer4 = net.layer3, net.layer4         # /16 256, /32 512
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = double_conv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = double_conv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = double_conv(128, 64)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    @staticmethod
    def _fit(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="nearest")
        return x

    def forward(self, x):
        s0 = self.stem(x)            # /2
        c1 = self.layer1(self.pool(s0))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        d4 = self.dec4(torch.cat([self._fit(self.up4(c4), c3), c3], 1))
        d3 = self.dec3(torch.cat([self._fit(self.up3(d4), c2), c2], 1))
        d2 = self.dec2(torch.cat([self._fit(self.up2(d3), c1), c1], 1))
        d1 = self.dec1(torch.cat([self._fit(self.up1(d2), s0), s0], 1))
        return self.head(d1)


def build(arch: str = "unet", base: int = 64, pretrained: bool = True) -> nn.Module:
    if arch == "unet":
        return UNet(base=base)
    if arch == "resnet18_unet":
        return ResNet18UNet(pretrained=pretrained)
    raise SystemExit(f"未知架构 {arch}，可选: unet / resnet18_unet")
