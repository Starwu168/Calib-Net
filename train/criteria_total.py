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
        self.cfg_loss = cfg_loss

        self.pmp_loss = PMPCompletionLoss(
            ms_weights=tuple(cfg_loss.get("pmp_ms_weights", (0.03, 0.06, 0.12, 0.25, 0.5, 1.0))),
            t_valid=float(cfg_loss.get("t_valid", 1e-3)),
            depth_min=cfg_loss.get("depth_min", None),
            depth_max=cfg_loss.get("depth_max", None),
            sparse_lidar_weight=float(cfg_loss.get("pmp_sparse_weight", 0.0)),
            l2_weight=float(cfg_loss.get("pmp_l2_weight", 0.0)),
            gt_outlier_kernel_size=int(cfg_loss.get("gt_outlier_kernel_size", -1)),
            gt_outlier_threshold=float(cfg_loss.get("gt_outlier_threshold", -1.0)),
        )

    def forward(self, pmp_out_list, s_pred_list, dep_sp, dep_gt, cfg_loss_calib: dict, dep_sparse_gt=None):
        cfg_calib = dict(cfg_loss_calib)
        cfg_calib.setdefault("t_valid", float(self.cfg_loss.get("t_valid", 1e-3)))
        cfg_calib.setdefault("depth_min", self.cfg_loss.get("depth_min", None))
        cfg_calib.setdefault("depth_max", self.cfg_loss.get("depth_max", None))
        cfg_calib.setdefault("use_sparse_gt_mask", bool(self.cfg_loss.get("use_sparse_gt_mask", True)))
        cfg_calib.setdefault("gt_outlier_kernel_size", int(self.cfg_loss.get("gt_outlier_kernel_size", -1)))
        cfg_calib.setdefault("gt_outlier_threshold", float(self.cfg_loss.get("gt_outlier_threshold", -1.0)))

        loss_calib, parts_calib = compute_calib_losses(
            s_pred_list, dep_sp, dep_gt, cfg_calib, dep_sparse_gt=dep_sparse_gt
        )
        loss_pmp, parts_pmp = self.pmp_loss(pmp_out_list, dep_gt, dep_sparse_gt=dep_sparse_gt)

        loss = self.lam_calib * loss_calib + self.lam_pmp * loss_pmp

        parts = {}
        parts.update(parts_calib)
        parts.update(parts_pmp)
        parts["loss_calib"] = float(loss_calib.detach().cpu())
        parts["loss_pmp"] = float(loss_pmp.detach().cpu())
        parts["loss"] = float(loss.detach().cpu())
        return loss, parts
