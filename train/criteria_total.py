# train/criteria_total.py
from __future__ import annotations
import torch
import torch.nn as nn

from train.criteria import compute_losses as compute_calib_losses
from train.criteria_pmp import PMPCompletionLoss


class TotalCriterion(nn.Module):
    """
    L = lambda_calib * L_calib + lambda_pmp * L_pmp
    - L_calib: your original compute_losses(...)
    - L_pmp: multi-scale masked supervision on BP-Net outputs
    """
    def __init__(self, cfg_loss: dict):
        super().__init__()
        self.lam_calib = float(cfg_loss.get("lambda_calib", 1.0))
        self.lam_pmp = float(cfg_loss.get("lambda_pmp", 1.0))

        self.pmp_loss = PMPCompletionLoss(
            ms_weights=tuple(cfg_loss.get("pmp_ms_weights", (0.03, 0.06, 0.12, 0.25, 0.5, 1.0))),
            t_valid=float(cfg_loss.get("t_valid", 1e-3)),
        )

    def forward(self, pmp_out_list, s_pred_list, dep_sp, dep_gt, cfg_loss_calib: dict):
        loss_calib, parts_calib = compute_calib_losses(s_pred_list, dep_sp, dep_gt, cfg_loss_calib)
        loss_pmp, parts_pmp = self.pmp_loss(pmp_out_list, dep_gt)

        loss = self.lam_calib * loss_calib + self.lam_pmp * loss_pmp

        parts = {}
        parts.update(parts_calib)
        parts.update(parts_pmp)
        parts["loss_calib"] = float(loss_calib.detach().cpu())
        parts["loss_pmp"] = float(loss_pmp.detach().cpu())
        parts["loss"] = float(loss.detach().cpu())
        return loss, parts
