# models/token_sparse.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_maxpool2d(S: torch.Tensor, M: torch.Tensor, k: int = 2, s: int = 2):
    """
    S: (B,1,H,W) depth (meters), invalid=0
    M: (B,1,H,W) {0,1}
    returns:
      S_d: (B,1,H/2,W/2) masked max pooled (invalid->0)
      M_d: (B,1,H/2,W/2) pooled mask
    """
    # invalid -> -inf for max
    neg_inf = torch.tensor(-1e6, device=S.device, dtype=S.dtype)
    S_masked = torch.where(M > 0, S, neg_inf)

    S_d = F.max_pool2d(S_masked, kernel_size=k, stride=s)  # (B,1,H/2,W/2)
    M_d = (F.max_pool2d(M, kernel_size=k, stride=s) > 0).float()

    # 把 -inf 还原为 0
    S_d = torch.where(M_d > 0, S_d, torch.zeros_like(S_d))
    return S_d, M_d

class SparseTokenEncoder(nn.Module):
    """
    per-scale 内部 token encoder：
    - 用 masked maxpool downsample，防止稀疏点消失
    - 再用 1x1/3x3 conv 提升通道
    """
    def __init__(self, out_c: int, down_ratio: int = 2):
        super().__init__()
        assert down_ratio in (1, 2, 4)
        self.down_ratio = down_ratio
        self.embed = nn.Sequential(
            nn.Conv2d(1, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, S_l: torch.Tensor, M_l: torch.Tensor):
        S_t, M_t = S_l, M_l
        if self.down_ratio == 2:
            S_t, M_t = masked_maxpool2d(S_t, M_t, 2, 2)
        elif self.down_ratio == 4:
            S_t, M_t = masked_maxpool2d(S_t, M_t, 2, 2)
            S_t, M_t = masked_maxpool2d(S_t, M_t, 2, 2)

        R_t = self.embed(S_t)  # (B,out_c,Ht,Wt)
        return R_t, S_t, M_t
