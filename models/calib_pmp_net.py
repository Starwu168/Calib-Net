# models/calib_pmp_net.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.calib_net import CalibOnlyNet
from bp.BPNet import Pre_MF_Post


class CalibPMPNet(nn.Module):
    """
    End-to-end depth completion:
      Calib: (RGB, S) -> S'  (保持稀疏，不扩散)
      PMP:   (RGB, S', K) -> list of dense depth outputs (BP-Net 原生 list)

    forward returns:
      pmp_out_list: list[Tensor]  (B,1,H,W) each, output[-1] is final
      s_pred_list:  list[Tensor]  calib multi-scale S'
      s_prime:      Tensor (B,1,H,W) used as PMP input
      calib_aux_list: list[dict]  per-scale projected point info for calib loss
    """
    def __init__(self, cfg_model: dict):
        super().__init__()
        # 兼容：如果你不想改 yaml，也允许 cfg_model 直接就是 calib 的 cfg
        calib_cfg = cfg_model.get("calib", cfg_model)
        self.calib = CalibOnlyNet(calib_cfg)

        # BP-Net PMP 主体：写死版
        self.pmp = Pre_MF_Post()

    @staticmethod
    def _build_sprime(s_pred_list, dep_sp: torch.Tensor) -> torch.Tensor:
        """
        将 calib 的 scale0 输出上采样到输入分辨率。
        当前 calib 可能已经在图像平面内移动点位，因此这里不再强行乘原始输入 mask。
        """
        s0 = s_pred_list[0]  # (B,1,h,w)
        H, W = dep_sp.shape[-2:]
        return F.interpolate(s0, size=(H, W), mode="nearest")

    def forward(self, rgb: torch.Tensor, dep_sp: torch.Tensor, K: torch.Tensor):
        s_pred_list, calib_aux_list = self.calib(rgb, dep_sp, K)
        s_prime = self._build_sprime(s_pred_list, dep_sp)
        with torch.amp.autocast("cuda", enabled=False):
            pmp_out_list = self.pmp(rgb.float(), s_prime.float(), K.float()) # BP-Net 原生输出 list
        return pmp_out_list, s_pred_list, s_prime, calib_aux_list
