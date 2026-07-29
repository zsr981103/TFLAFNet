# -*- coding: utf-8 -*-
"""
Res-Lap 混合网络：编码器与 UNet_DE_Res 相同（ResBlock1D + MaxPool 下采样），
解码器采用 LUNet 的 LapFusionUp（时域/拉普拉斯高频四路 + 通道注意力 + 1×1 融合 + double_conv 细化）。
"""

import torch
import torch.nn as nn

from LUNet import LapFusionUp
from UNet_DE_Res import Down, ResBlock1D


class ResEncLapDecNet(nn.Module):
    """ResBlock 编码器 + 拉普拉斯四分支注意力解码器。"""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        bilinear: bool = True,
        base_c: int = 64,
        gauss_kernel: int = 5,
        ca_reduction: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.in_conv = ResBlock1D(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = LapFusionUp(
            base_c * 16 // factor,
            base_c * 8,
            base_c * 8 // factor,
            bilinear=bilinear,
            gauss_kernel=gauss_kernel,
            ca_reduction=ca_reduction,
        )
        self.up2 = LapFusionUp(
            base_c * 8 // factor,
            base_c * 4,
            base_c * 4 // factor,
            bilinear=bilinear,
            gauss_kernel=gauss_kernel,
            ca_reduction=ca_reduction,
        )
        self.up3 = LapFusionUp(
            base_c * 4 // factor,
            base_c * 2,
            base_c * 2 // factor,
            bilinear=bilinear,
            gauss_kernel=gauss_kernel,
            ca_reduction=ca_reduction,
        )
        self.up4 = LapFusionUp(
            base_c * 2 // factor,
            base_c,
            base_c,
            bilinear=bilinear,
            gauss_kernel=gauss_kernel,
            ca_reduction=ca_reduction,
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(base_c, num_classes, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.out_conv(x)
        return logits + x if logits.shape[1] == x.shape[1] else logits


if __name__ == "__main__":
    net = ResEncLapDecNet(
        in_channels=1,
        num_classes=1,
        bilinear=True,
        base_c=64,
        gauss_kernel=5,
        ca_reduction=4,
    )
    dummy = torch.zeros((2, 1, 256))
    with torch.no_grad():
        out = net(dummy)
    print("Input:", dummy.shape, "Output:", out.shape)
    print("ResEncLapDecNet OK")
