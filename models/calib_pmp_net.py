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
      Calib: (RGB, S) -> S', C'
      PMP:   (RGB, S', C', K) -> list of dense depth outputs

    forward returns:
      pmp_out_list: list[Tensor]  (B,1,H,W) each, output[-1] is final
      s_pred_list:  list[Tensor]  calib multi-scale corrected sparse depth
      s_prime:      Tensor (B,1,H,W) used as PMP input
      c_prime:      Tensor (B,1,H,W) confidence used by PMP
      calib_aux_list: list[dict]  per-scale projected point info for calib loss
    """

    def __init__(self, cfg_model: dict):
        super().__init__()
        calib_cfg = cfg_model.get("calib", cfg_model)
        self.calib = CalibOnlyNet(calib_cfg)
        self.pmp = Pre_MF_Post(cfg_model.get("pmp", {}))

    @staticmethod
    def _build_sprime(s_pred_list, dep_sp: torch.Tensor) -> torch.Tensor:
        s0 = s_pred_list[0]
        H, W = dep_sp.shape[-2:]
        return F.interpolate(s0, size=(H, W), mode="nearest")

    @staticmethod
    def _build_cprime(c_pred_list, dep_sp: torch.Tensor) -> torch.Tensor:
        c0 = c_pred_list[0]
        H, W = dep_sp.shape[-2:]
        return F.interpolate(c0, size=(H, W), mode="nearest").clamp_(0.0, 1.0)

    def forward(self, rgb: torch.Tensor, dep_sp: torch.Tensor, K: torch.Tensor):
        s_pred_list, c_pred_list, calib_aux_list = self.calib(rgb, dep_sp, K)
        s_prime = self._build_sprime(s_pred_list, dep_sp)
        c_prime = self._build_cprime(c_pred_list, dep_sp)
        with torch.amp.autocast("cuda", enabled=False):
            pmp_out_list = self.pmp(rgb.float(), s_prime.float(), c_prime.float(), K.float())
        return pmp_out_list, s_pred_list, s_prime, c_prime, calib_aux_list
