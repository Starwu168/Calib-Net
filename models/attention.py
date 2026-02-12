# models/attention.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowSelfAttention2D(nn.Module):
    """
    局部窗口 SA：每个位置只对邻域 window×window 做注意力。
    这里用 unfold 实现，窗口小（5×5）可接受。
    """
    def __init__(self, dim: int, heads: int, dim_head: int, window: int):
        super().__init__()
        assert window % 2 == 1, "window must be odd"
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.window = window
        inner = heads * dim_head

        self.to_q = nn.Conv2d(dim, inner, 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner, 1, bias=False)
        self.to_v = nn.Conv2d(dim, inner, 1, bias=False)
        self.proj = nn.Conv2d(inner, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)
        return: (B,C,H,W)
        """
        B, C, H, W = x.shape
        h = self.heads
        d = self.dim_head
        win = self.window
        pad = win // 2

        q = self.to_q(x)  # (B, h*d, H, W)
        k = self.to_k(x)
        v = self.to_v(x)

        # unfold K,V 成局部邻域 tokens: (B, h*d*win*win, H*W)
        k_unf = F.unfold(k, kernel_size=win, padding=pad)  # (B, h*d*win*win, HW)
        v_unf = F.unfold(v, kernel_size=win, padding=pad)

        # reshape
        # q: (B,h,d,HW)
        q = q.view(B, h, d, H*W)
        # k_unf: (B,h,d,win*win,HW)
        k_unf = k_unf.view(B, h, d, win*win, H*W)
        v_unf = v_unf.view(B, h, d, win*win, H*W)

        # attention: (B,h,win*win,HW)
        # score_j = q·k_j
        attn = (q.unsqueeze(3) * k_unf).sum(dim=2) / (d ** 0.5)
        attn = attn.softmax(dim=3)

        # weighted sum: (B,h,d,HW)
        out = (attn.unsqueeze(2) * v_unf).sum(dim=3)
        out = out.view(B, h*d, H, W)
        return self.proj(out)

class CrossFusionSamePos(nn.Module):
    """
    你定义的 CA：Radar token 作 Q，同位置 RGB 特征作 KV。
    由于 KV 只有一个 token，本质是跨模态投影+融合。
    """
    def __init__(self, radar_dim: int, rgb_dim: int, out_dim: int):
        super().__init__()
        self.r = nn.Conv2d(radar_dim, out_dim, 1, bias=False)
        self.i = nn.Conv2d(rgb_dim,   out_dim, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, radar_feat: torch.Tensor, rgb_feat: torch.Tensor) -> torch.Tensor:
        """
        radar_feat: (B,Cr,H,W)
        rgb_feat:   (B,Ci,H,W)
        out:        (B,out_dim,H,W)
        """
        r = self.r(radar_feat)
        i = self.i(rgb_feat)
        return self.fuse(torch.cat([r, i], dim=1))
