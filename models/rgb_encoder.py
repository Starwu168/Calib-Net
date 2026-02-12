# models/rgb_encoder.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class RGBPyramidEncoder(nn.Module):
    """
    输入: RGB (B,3,H,W)
    输出: 5个尺度特征 fout_list[l]，空间尺寸依次 /2
    """
    def __init__(self, channels: list[int]):
        super().__init__()
        assert len(channels) == 5
        c0, c1, c2, c3, c4 = channels

        self.stem = nn.Sequential(
            ConvBNAct(3, c0, 3, 2, 1),   # /2
            ConvBNAct(c0, c0, 3, 1, 1),
        )
        self.l1 = nn.Sequential(ConvBNAct(c0, c1, 3, 2, 1), ConvBNAct(c1, c1))
        self.l2 = nn.Sequential(ConvBNAct(c1, c2, 3, 2, 1), ConvBNAct(c2, c2))
        self.l3 = nn.Sequential(ConvBNAct(c2, c3, 3, 2, 1), ConvBNAct(c3, c3))
        self.l4 = nn.Sequential(ConvBNAct(c3, c4, 3, 2, 1), ConvBNAct(c4, c4))

    def forward(self, rgb):
        f0 = self.stem(rgb)   # /2
        f1 = self.l1(f0)      # /4
        f2 = self.l2(f1)      # /8
        f3 = self.l3(f2)      # /16
        f4 = self.l4(f3)      # /32
        return [f0, f1, f2, f3, f4]
