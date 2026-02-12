# models/attn_dilated.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedWindowSelfAttention2D(nn.Module):
    """
    window SA with dilation + mask-aware keys
    - x: (B,C,H,W)
    - m: (B,1,H,W) valid mask (0/1)
    """
    def __init__(self, dim: int, heads: int, dim_head: int, window: int, dilation: int = 2):
        super().__init__()
        assert window % 2 == 1
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.window = window
        self.dilation = dilation
        inner = heads * dim_head

        self.to_q = nn.Conv2d(dim, inner, 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner, 1, bias=False)
        self.to_v = nn.Conv2d(dim, inner, 1, bias=False)
        self.proj = nn.Conv2d(inner, dim, 1, bias=False)

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        B, C, H, W = x.shape
        h = self.heads
        d = self.dim_head
        win = self.window
        pad = (win // 2) * self.dilation

        q = self.to_q(x)  # (B,h*d,H,W)
        k = self.to_k(x)
        v = self.to_v(x)

        # unfold with dilation
        k_unf = F.unfold(k, kernel_size=win, padding=pad, dilation=self.dilation)  # (B,h*d*win*win,HW)
        v_unf = F.unfold(v, kernel_size=win, padding=pad, dilation=self.dilation)

        # unfold mask（同样 dilation），用于 key masking
        m_unf = F.unfold(m, kernel_size=win, padding=pad, dilation=self.dilation)  # (B,1*win*win,HW)
        m_unf = (m_unf > 0).float()

        q = q.view(B, h, d, H * W)
        k_unf = k_unf.view(B, h, d, win * win, H * W)
        v_unf = v_unf.view(B, h, d, win * win, H * W)

        # attn logits: (B,h,win*win,HW)
        logits = (q.unsqueeze(3) * k_unf).sum(dim=2) / (d ** 0.5)

        # mask invalid keys: set -inf where m_unf==0
        # m_unf: (B,win*win,HW) -> broadcast to (B,1,win*win,HW)
        logits = logits + (m_unf.unsqueeze(1) - 1.0) * 1e9

        attn = logits.softmax(dim=2)  # softmax over window positions

        out = (attn.unsqueeze(2) * v_unf).sum(dim=3)  # (B,h,d,HW)
        out = out.view(B, h * d, H, W)
        return self.proj(out)
