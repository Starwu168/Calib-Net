# train/criteria_pmp.py
from __future__ import annotations
import torch
import torch.nn as nn


def _depth_valid_mask(
    depth: torch.Tensor,
    t_valid: float,
    depth_min: float | None,
    depth_max: float | None,
    ignore_top_rows: int = 0,
) -> torch.Tensor:
    m = depth > t_valid
    if depth_min is not None:
        m = m & (depth > float(depth_min))
    if depth_max is not None:
        m = m & (depth < float(depth_max))
    if ignore_top_rows > 0:
        m[..., :ignore_top_rows, :] = False
    return m


def _remove_outlier_mask(depth: torch.Tensor, valid: torch.Tensor, kernel_size: int, threshold: float):
    if kernel_size <= 1 or threshold <= 0:
        return valid

    max_value = 10.0 * torch.clamp(depth.max(), min=1.0)
    depth_filled = torch.where(valid, depth, torch.full_like(depth, max_value))
    padding = kernel_size // 2
    depth_pad = torch.nn.functional.pad(
        depth_filled, (padding, padding, padding, padding), mode="constant", value=max_value
    )
    min_values = -torch.nn.functional.max_pool2d(-depth_pad, kernel_size=kernel_size, stride=1, padding=0)

    not_outlier = ~(min_values < (depth - threshold))
    return valid & not_outlier


class PMPCompletionLoss(nn.Module):
    """
    Multi-scale masked L1 for BP-Net outputs.
    - dense supervision on dep_gt
    - optional sparse supervision on dep_sparse_gt (RadarCam-Depth style)
    """

    def __init__(
        self,
        ms_weights=(0.03, 0.06, 0.12, 0.25, 0.5, 1.0),
        t_valid: float = 1e-3,
        depth_min: float | None = None,
        depth_max: float | None = None,
        dense_weight: float = 1.0,
        sparse_lidar_weight: float = 0.0,
        sparse_huber_weight: float = 0.0,
        sparse_huber_beta: float = 1.0,
        gt_outlier_kernel_size: int = -1,
        gt_outlier_threshold: float = -1.0,
    ):
        super().__init__()
        self.ms_weights = list(ms_weights)
        self.t_valid = float(t_valid)
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.dense_weight = float(dense_weight)
        self.sparse_lidar_weight = float(sparse_lidar_weight)
        self.sparse_huber_weight = float(sparse_huber_weight)
        self.sparse_huber_beta = float(sparse_huber_beta)
        self.gt_outlier_kernel_size = int(gt_outlier_kernel_size)
        self.gt_outlier_threshold = float(gt_outlier_threshold)
        self.ignore_top_rows = 0

    @staticmethod
    def _masked_l1(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(valid.sum(), min=1.0)
        return ((pred - gt).abs() * valid).sum() / denom

    def _masked_huber(self, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(valid.sum(), min=1.0)
        loss = torch.nn.functional.smooth_l1_loss(
            pred,
            gt,
            reduction="none",
            beta=self.sparse_huber_beta,
        )
        return (loss * valid).sum() / denom

    def _single_scale_loss(self, pred, dep_gt, dep_sparse_gt=None):
        valid_dense = _depth_valid_mask(
            dep_gt,
            self.t_valid,
            self.depth_min,
            self.depth_max,
            ignore_top_rows=self.ignore_top_rows,
        )
        valid_dense = _remove_outlier_mask(
            dep_gt, valid_dense, self.gt_outlier_kernel_size, self.gt_outlier_threshold
        ).float()
        dense = self._masked_l1(pred, dep_gt, valid_dense) if self.dense_weight > 0.0 else pred.new_zeros(())

        sparse = pred.new_zeros(())
        sparse_huber = pred.new_zeros(())
        if dep_sparse_gt is not None and (self.sparse_lidar_weight > 0.0 or self.sparse_huber_weight > 0.0):
            valid_sparse = (dep_sparse_gt > self.t_valid).float()
            if self.sparse_lidar_weight > 0.0:
                sparse = self._masked_l1(pred, dep_sparse_gt, valid_sparse)
            if self.sparse_huber_weight > 0.0:
                sparse_huber = self._masked_huber(pred, dep_sparse_gt, valid_sparse)

        total = (
            self.dense_weight * dense
            + self.sparse_lidar_weight * sparse
            + self.sparse_huber_weight * sparse_huber
        )
        return total, dense.detach(), sparse.detach(), sparse_huber.detach()

    def forward(self, outputs, dep_gt, dep_sparse_gt=None):
        if isinstance(outputs, (list, tuple)):
            assert len(outputs) == len(self.ms_weights), (
                f"len(outputs)={len(outputs)} != len(ms_weights)={len(self.ms_weights)}"
            )
            loss = outputs[0].new_zeros(())
            loss_dense = outputs[0].new_zeros(())
            loss_sparse = outputs[0].new_zeros(())
            loss_sparse_huber = outputs[0].new_zeros(())
            for o, w in zip(outputs, self.ms_weights):
                ls, ld, lsp, lsh = self._single_scale_loss(o, dep_gt, dep_sparse_gt=dep_sparse_gt)
                loss = loss + float(w) * ls
                loss_dense = loss_dense + float(w) * ld
                loss_sparse = loss_sparse + float(w) * lsp
                loss_sparse_huber = loss_sparse_huber + float(w) * lsh
        else:
            loss, loss_dense, loss_sparse, loss_sparse_huber = self._single_scale_loss(
                outputs, dep_gt, dep_sparse_gt=dep_sparse_gt
            )

        return loss, {
            "loss_pmp": float(loss.detach().cpu()),
            "loss_pmp_dense": float(loss_dense.detach().cpu()),
            "loss_pmp_sparse": float(loss_sparse.detach().cpu()),
            "loss_pmp_sparse_huber": float(loss_sparse_huber.detach().cpu()),
        }
