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
        将 calib 的 scale0 输出上采样到输入分辨率，并硬 mask 限制稀疏性：
          S' = up_nearest(S0) * 1(dep_sp>0)
        这样不会出现 S' 非零点膨胀的问题。
        """
        s0 = s_pred_list[0]  # (B,1,h,w)
        H, W = dep_sp.shape[-2:]
        s_up = F.interpolate(s0, size=(H, W), mode="nearest")
        mask = (dep_sp > 0).float()
        return s_up * mask

    def forward(self, rgb: torch.Tensor, dep_sp: torch.Tensor, K: torch.Tensor):
        s_pred_list = self.calib(rgb, dep_sp)
        s_prime = self._build_sprime(s_pred_list, dep_sp)
        with torch.cuda.amp.autocast(enabled=False):
            pmp_out_list = self.pmp(rgb.float(), s_prime.float(), K.float()) # BP-Net 原生输出 list
        return pmp_out_list, s_pred_list, s_prime
