# models/radar_wpool.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class RadarWPool(nn.Module):
    """
    用每个尺度的 RGB 特征 fout^l 生成权重 W^l，
    将原始 S/M 在该尺度下对齐成 S_l, M_l。
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def weighted_pool(S: torch.Tensor, M: torch.Tensor, W: torch.Tensor, out_hw: tuple[int, int], eps: float = 1e-6):
        """
        S,M: (B,1,H,W), W: (B,1,H,W) 同分辨率
        out_hw: (Hl, Wl)
        """
        Hl, Wl = out_hw
        # 下采样到 Hl,Wl：用 area 聚合做 window pooling（等价均值池化窗口）
        # 先把加权和 / 权重和分别 area downsample
        SMW = S * M * W
        MW = M * W
        num = F.interpolate(SMW, size=(Hl, Wl), mode="area")
        den = F.interpolate(MW,  size=(Hl, Wl), mode="area")
        S_l = num / (den + eps)
        # 有效 mask：只要窗口里有有效点，就认为有效
        M_l = (F.interpolate(M, size=(Hl, Wl), mode="area") > 0).float()
        return S_l, M_l

    def forward(self, S: torch.Tensor, fout_l: torch.Tensor):
        """
        S: (B,1,H,W)
        fout_l: (B,C,Hl,Wl)
        返回: S_l, M_l (B,1,Hl,Wl)
        """
        B, _, H, W = S.shape
        Hl, Wl = fout_l.shape[-2], fout_l.shape[-1]
        M = (S > 0).float()

        # 生成权重：先把 fout_l 上采样到原分辨率以引导 pooling（更贴“RGB结构引导”）
        g = F.interpolate(fout_l, size=(H, W), mode="bilinear", align_corners=False)
        Wg = torch.exp(g.mean(dim=1, keepdim=True))  # (B,1,H,W) 正权重

        S_l, M_l = self.weighted_pool(S, M, Wg, (Hl, Wl))
        return S_l, M_l
