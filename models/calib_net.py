# models/calib_net.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rgb_encoder import RGBPyramidEncoder, ConvBNAct
from .radar_wpool import RadarWPool
from .attention import WindowSelfAttention2D, CrossFusionSamePos
from .calib_blocks import SEBlock, ResBlock
from .token_sparse import SparseTokenEncoder
from .attn_dilated import DilatedWindowSelfAttention2D

class TokenEncoder(nn.Module):
    """
    在每个尺度内部，把 (B,1,Hl,Wl) 的稀疏深度再下采样到 /4 得到 token 特征。
    """
    def __init__(self, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(1, out_c, 3, 2, 1),   # /2
            ConvBNAct(out_c, out_c, 3, 2, 1) # /4
        )
    def forward(self, s_l):
        return self.net(s_l)

class TokenDecoder(nn.Module):
    """
    token 特征上采样回尺度 Hl,Wl 并输出 S_l' (B,1,Hl,Wl)
    """
    def __init__(self, in_c: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.GELU(),
            nn.Conv2d(in_c, 1, 1)
        )

    def forward(self, x, out_hw):
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        s = self.proj(x)
        return F.relu(s)  # 深度非负

class PerScaleCalib(nn.Module):
    def __init__(self, rgb_c: int, radar_c: int, sa_cfg: dict, ca_out_dim: int, refine_cfg: dict, token_down_ratio: int):
        super().__init__()
        self.token_down_ratio = token_down_ratio

        self.token_enc = SparseTokenEncoder(out_c=radar_c, down_ratio=token_down_ratio)

        self.sa = DilatedWindowSelfAttention2D(
            dim=radar_c,
            heads=sa_cfg["heads"],
            dim_head=sa_cfg["dim_head"],
            window=sa_cfg["window"],
            dilation=sa_cfg.get("dilation", 2),
        )

        self.ca = CrossFusionSamePos(
            radar_dim=radar_c,
            rgb_dim=rgb_c,
            out_dim=ca_out_dim,
        )

        self.fuse_proj = nn.Sequential(
            nn.Conv2d(radar_c + ca_out_dim, radar_c, 1, bias=False),
            nn.BatchNorm2d(radar_c),
            nn.GELU(),
        )

        blocks = [ResBlock(radar_c) for _ in range(refine_cfg["blocks"])]
        self.refine = nn.Sequential(*blocks)
        self.se = SEBlock(radar_c, refine_cfg.get("se_reduction", 8)) if refine_cfg.get("use_channel_attn", True) else nn.Identity()

        self.dec = TokenDecoder(radar_c)

    def forward(self, S_l: torch.Tensor, M_l: torch.Tensor, fout_l: torch.Tensor):
        """
        S_l, M_l: (B,1,Hl,Wl) 该尺度 sparse depth + mask
        fout_l:   (B,C,Hl,Wl) 该尺度 rgb features
        输出: S'_l (B,1,Hl,Wl)
        """
        Hl, Wl = S_l.shape[-2], S_l.shape[-1]

        # token encoder（masked maxpool 保点）
        R_t, S_t, M_t = self.token_enc(S_l, M_l)  # R_t: (B,Cr,Ht,Wt)

        # SA（mask-aware + dilated）
        R_t = R_t + self.sa(R_t, M_t)

        # CA（同位置融合）：把 RGB 插值到 token 分辨率
        F_tok = F.interpolate(fout_l, size=R_t.shape[-2:], mode="bilinear", align_corners=False)
        Z = self.ca(R_t, F_tok)  # (B,ca_out,Ht,Wt)

        # 融合 + residual refine
        Fprime = self.fuse_proj(torch.cat([R_t, Z], dim=1))
        delta = self.se(self.refine(Fprime))
        F2 = Fprime + delta

        # decode to S'_l
        S_l_pred = self.dec(F2, out_hw=(Hl, Wl))
        return S_l_pred

class CalibOnlyNet(nn.Module):
    """
    calib-only 训练网络：
    - 自己带 RGB pyramid encoder（后面接 BPNet 时可替换成 BPNet encoder 输出）
    - per-scale wpool 对齐 S -> S_l
    - per-scale calib 得到 S_l'
    """
    def __init__(self, cfg_model: dict):
        super().__init__()
        self.num_scales = cfg_model["num_scales"]
        assert self.num_scales == 5, "按 BPNet 5尺度"

        self.rgb_enc = RGBPyramidEncoder(cfg_model["rgb_channels"])
        self.wpool = RadarWPool()

        self.calibs = nn.ModuleList()
        token_down_ratio = cfg_model["radar_token_down_ratio"]
        for l in range(self.num_scales):
            self.calibs.append(
                PerScaleCalib(
                    rgb_c=cfg_model["rgb_channels"][l],
                    radar_c=cfg_model["radar_channels"][l],
                    sa_cfg=cfg_model["sa"],
                    ca_out_dim=cfg_model["ca"]["out_dim"],
                    refine_cfg=cfg_model["refine"],
                    token_down_ratio=token_down_ratio,
                )
            )

    def forward(self, rgb: torch.Tensor, dep_sp: torch.Tensor):
        """
        rgb:    (B,3,H,W)
        dep_sp: (B,1,H,W) sparse radar depth
        return:
          s_pred_list: list of per-scale S_l' (B,1,Hl,Wl)
        """
        fout_list = self.rgb_enc(rgb)
        s_pred_list = []

        for l in range(self.num_scales):
            fout_l = fout_list[l]
            s_l, m_l = self.wpool(dep_sp, fout_l)  # 对齐到该尺度
            s_l_pred = self.calibs[l](s_l,m_l,fout_l)
            s_pred_list.append(s_l_pred)

        return s_pred_list
