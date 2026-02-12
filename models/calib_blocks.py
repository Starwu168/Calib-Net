# models/calib_blocks.py
from __future__ import annotations
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, c: int, reduction: int = 8):
        super().__init__()
        mid = max(4, c // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, mid, 1),
            nn.GELU(),
            nn.Conv2d(mid, c, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.net(x))
