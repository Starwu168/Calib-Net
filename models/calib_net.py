# models/calib_net.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rgb_encoder import RGBPyramidEncoder, ConvBNAct
from .radar_wpool import RadarWPool
from .attention import CrossFusionSamePos
from .calib_blocks import SEBlock, ResBlock
from .token_sparse import SparseTokenEncoder
from .attn_dilated import DilatedWindowSelfAttention2D


def _scale_intrinsics(K: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    K_scaled = K.clone()
    K_scaled[:, 0, 0] = K_scaled[:, 0, 0] * sx
    K_scaled[:, 1, 1] = K_scaled[:, 1, 1] * sy
    K_scaled[:, 0, 2] = K_scaled[:, 0, 2] * sx
    K_scaled[:, 1, 2] = K_scaled[:, 1, 2] * sy
    return K_scaled


def _make_xy_grid(height: int, width: int, device, dtype):
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return xx.view(1, 1, height, width), yy.view(1, 1, height, width)


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
    def __init__(
        self,
        rgb_c,
        radar_c,
        sa_cfg,
        ca_out_dim,
        refine_cfg,
        token_down_ratio,
        predict_xyz=False,
        range_min_xyz=(0.0, 0.0, 0.0),
        range_max_xyz=(1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.predict_xyz = bool(predict_xyz)
        range_min = torch.tensor(range_min_xyz, dtype=torch.float32).view(1, 3, 1, 1)
        range_max = torch.tensor(range_max_xyz, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("range_min_xyz", range_min, persistent=False)
        self.register_buffer("range_max_xyz", range_max, persistent=False)

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
        self.offset_adapter = ResBlock(radar_c)
        self.conf_adapter = ResBlock(radar_c)
        self.range_adapter = ResBlock(radar_c) if self.predict_xyz else None

        self.offset_head = nn.Sequential(
            nn.Conv2d(radar_c, radar_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(radar_c),
            nn.GELU(),
            nn.Conv2d(radar_c, 3 if self.predict_xyz else 1, 1)
        )
        self.conf_head = nn.Sequential(
            nn.Conv2d(radar_c, radar_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(radar_c),
            nn.GELU(),
            nn.Conv2d(radar_c, 1, 1)
        )
        self.range_head = None
        if self.predict_xyz:
            self.range_head = nn.Sequential(
                nn.Conv2d(radar_c, radar_c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(radar_c),
                nn.GELU(),
                nn.Conv2d(radar_c, 3, 1)
            )

    @staticmethod
    def _splat_average(u_proj: torch.Tensor, v_proj: torch.Tensor, value: torch.Tensor, valid: torch.Tensor):
        B, _, H, W = value.shape
        num = value.new_zeros((B, H * W))
        den = value.new_zeros((B, H * W))

        u = u_proj.reshape(B, -1)
        v = v_proj.reshape(B, -1)
        val = value.reshape(B, -1)
        valid_flat = valid.reshape(B, -1)

        u0 = torch.floor(u)
        v0 = torch.floor(v)
        u1 = u0 + 1.0
        v1 = v0 + 1.0

        neighbors = (
            (u0, v0, (u1 - u) * (v1 - v)),
            (u1, v0, (u - u0) * (v1 - v)),
            (u0, v1, (u1 - u) * (v - v0)),
            (u1, v1, (u - u0) * (v - v0)),
        )

        for uu, vv, ww in neighbors:
            inside = valid_flat & (uu >= 0) & (uu <= (W - 1)) & (vv >= 0) & (vv <= (H - 1))
            weight = ww * inside.float()
            idx = vv.clamp(0, H - 1).long() * W + uu.clamp(0, W - 1).long()
            num.scatter_add_(1, idx, val * weight)
            den.scatter_add_(1, idx, weight)

        out = (num / (den + 1e-6)).view(B, 1, H, W)
        return torch.where(den.view(B, 1, H, W) > 1e-6, out, torch.zeros_like(out))

    def _apply_xyz_delta(
        self,
        S_l: torch.Tensor,
        M_l: torch.Tensor,
        K_l: torch.Tensor,
        delta_xyz: torch.Tensor,
        conf_map: torch.Tensor,
    ):
        _, _, H, W = S_l.shape
        xx, yy = _make_xy_grid(H, W, device=S_l.device, dtype=S_l.dtype)
        xx = xx.expand(S_l.shape[0], -1, -1, -1)
        yy = yy.expand(S_l.shape[0], -1, -1, -1)

        fx = K_l[:, 0:1, 0:1].view(-1, 1, 1, 1)
        fy = K_l[:, 1:2, 1:2].view(-1, 1, 1, 1)
        cx = K_l[:, 0:1, 2:3].view(-1, 1, 1, 1)
        cy = K_l[:, 1:2, 2:3].view(-1, 1, 1, 1)

        z = S_l
        x = z * (xx - cx) / (fx + 1e-6)
        y = z * (yy - cy) / (fy + 1e-6)

        delta_xyz = delta_xyz * M_l
        x_new = x + delta_xyz[:, 0:1]
        y_new = y + delta_xyz[:, 1:2]
        z_new = torch.relu(z + delta_xyz[:, 2:3])

        valid = (M_l > 0) & (z_new > 1e-6)
        z_safe = torch.clamp(z_new, min=1e-6)
        u_new = fx * x_new / z_safe + cx
        v_new = fy * y_new / z_safe + cy
        proj_conf = conf_map * M_l
        s_pred = self._splat_average(u_new, v_new, z_new, valid)
        c_pred = self._splat_average(u_new, v_new, proj_conf, valid)
        info = {
            "proj_u": u_new,
            "proj_v": v_new,
            "proj_z": z_new,
            "proj_conf": proj_conf,
            "valid": valid.float(),
            "delta_xyz": delta_xyz,
        }
        return s_pred, c_pred, info

    def forward(self, S_l, M_l, fout_l, K_l=None):
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
        offset_feat = self.offset_adapter(delta_feat)
        conf_feat = self.conf_adapter(delta_feat)

        # === 预测 ΔS (token 分辨率) ===
        offset = self.offset_head(offset_feat)
        conf = torch.sigmoid(self.conf_head(conf_feat))

        # === 上采样回 Hl,Wl ===
        offset = F.interpolate(offset, size=(Hl, Wl), mode="nearest")
        conf = F.interpolate(conf, size=(Hl, Wl), mode="nearest") * M_l

        if self.predict_xyz:
            if K_l is None:
                raise ValueError("K_l is required when predict_xyz=True")
            range_feat = self.range_adapter(delta_feat)
            range_raw = self.range_head(range_feat)
            range_raw = F.interpolate(range_raw, size=(Hl, Wl), mode="nearest")
            range_xyz = self.range_min_xyz + (self.range_max_xyz - self.range_min_xyz) * torch.sigmoid(range_raw)
            delta_xyz = range_xyz * torch.tanh(offset)
            S_pred, C_pred, aux = self._apply_xyz_delta(S_l, M_l, K_l, delta_xyz, conf)
            aux["range_xyz"] = range_xyz * M_l
        else:
            # === 膨胀 mask（允许 n*n 邻域修正）===
            M_dil = F.max_pool2d(M_l, kernel_size=3, stride=1, padding=1)

            # === 残差修正 ===
            S_pred = torch.relu(S_l + offset * M_dil)
            C_pred = conf * M_l
            aux = {
                "delta_xyz": torch.cat(
                    [torch.zeros_like(S_pred), torch.zeros_like(S_pred), (S_pred - S_l) * M_dil],
                    dim=1,
                ),
                "range_xyz": torch.zeros(S_pred.shape[0], 3, Hl, Wl, device=S_pred.device, dtype=S_pred.dtype),
                "proj_u": None,
                "proj_v": None,
                "proj_z": S_pred,
                "proj_conf": C_pred,
                "valid": M_l.float(),
            }

        return S_pred, C_pred, aux

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
        self.predict_xyz = bool(cfg_model.get("predict_xyz", False))

        self.rgb_enc = RGBPyramidEncoder(cfg_model["rgb_channels"])
        self.wpool = RadarWPool()
        range_min_xyz = tuple(cfg_model.get("range_min_xyz", (0.0, 0.0, 0.0)))
        range_max_xyz = tuple(cfg_model.get("range_max_xyz", (1.0, 1.0, 1.0)))

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
                    predict_xyz=self.predict_xyz,
                    range_min_xyz=range_min_xyz,
                    range_max_xyz=range_max_xyz,
                )
            )

    def forward(self, rgb: torch.Tensor, dep_sp: torch.Tensor, K: torch.Tensor | None = None):
        """
        rgb:    (B,3,H,W)
        dep_sp: (B,1,H,W) sparse radar depth
        return:
          s_pred_list: list of per-scale S_l' (B,1,Hl,Wl)
        """
        fout_list = self.rgb_enc(rgb)
        s_pred_list = []
        c_pred_list = []
        calib_aux_list = []
        H, W = dep_sp.shape[-2:]

        for l in range(self.num_scales):
            fout_l = fout_list[l]
            s_l, m_l = self.wpool(dep_sp, fout_l)  # 对齐到该尺度
            K_l = None
            if K is not None:
                Hl, Wl = s_l.shape[-2:]
                K_l = _scale_intrinsics(K, sx=Wl / float(W), sy=Hl / float(H))
            s_l_pred, c_l_pred, calib_aux = self.calibs[l](s_l, m_l, fout_l, K_l=K_l)
            calib_aux["scale_hw"] = (s_l.shape[-2], s_l.shape[-1])
            s_pred_list.append(s_l_pred)
            c_pred_list.append(c_l_pred)
            calib_aux_list.append(calib_aux)

        return s_pred_list, c_pred_list, calib_aux_list
