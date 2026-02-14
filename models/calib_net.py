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
    def __init__(self, rgb_c, radar_c, sa_cfg, ca_out_dim, refine_cfg, token_down_ratio):
        super().__init__()

        self.token_enc = SparseTokenEncoder(
            out_c=radar_c,
            down_ratio=token_down_ratio
        )

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
        self.se = SEBlock(radar_c, refine_cfg.get("se_reduction", 8))

        # 关键改动：输出 ΔS 而不是直接输出 S
        self.delta_head = nn.Sequential(
            nn.Conv2d(radar_c, radar_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(radar_c),
            nn.GELU(),
            nn.Conv2d(radar_c, 1, 1)
        )

    def forward(self, S_l, M_l, fout_l):
        Hl, Wl = S_l.shape[-2:]

        # === token encode ===
        R_t, S_t, M_t = self.token_enc(S_l, M_l)

        # === SA ===
        R_t = R_t + self.sa(R_t, M_t)

        # === CA ===
        F_tok = F.interpolate(fout_l, size=R_t.shape[-2:], mode="bilinear", align_corners=False)
        Z = self.ca(R_t, F_tok)

        Fprime = self.fuse_proj(torch.cat([R_t, Z], dim=1))
        delta_feat = self.se(self.refine(Fprime))

        # === 预测 ΔS (token 分辨率) ===
        delta = self.delta_head(delta_feat)

        # === 上采样回 Hl,Wl ===
        delta = F.interpolate(delta, size=(Hl, Wl), mode="bilinear", align_corners=False)

        # === 膨胀 mask（允许 3x3 邻域修正）===
        M_dil = F.max_pool2d(M_l, kernel_size=3, stride=1, padding=1)

        # === 残差修正 ===
        S_pred = torch.relu(S_l + delta * M_dil)

        return S_pred

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
