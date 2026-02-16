# train/criteria_pmp.py
from __future__ import annotations
import torch
import torch.nn as nn


class PMPCompletionLoss(nn.Module):
    """
    Multi-scale masked L1 for BP-Net outputs (list).
    outputs: list of (B,1,H,W)
    gt:      (B,1,H,W)
    """
    def __init__(self, ms_weights=(0.03, 0.06, 0.12, 0.25, 0.5, 1.0), t_valid: float = 1e-3):
        super().__init__()
        self.ms_weights = list(ms_weights)
        self.t_valid = float(t_valid)

    @staticmethod
    def _masked_l1(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return ((pred - gt).abs() * valid).mean()

    def forward(self, outputs, gt):
        valid = (gt > self.t_valid).float()
        if isinstance(outputs, (list, tuple)):
            assert len(outputs) == len(self.ms_weights), \
                f"len(outputs)={len(outputs)} != len(ms_weights)={len(self.ms_weights)}"
            loss = 0.0
            for o, w in zip(outputs, self.ms_weights):
                loss = loss + float(w) * self._masked_l1(o, gt, valid)
            return loss, {"loss_pmp": float(loss.detach().cpu())}
        else:
            loss = self._masked_l1(outputs, gt, valid)
            return loss, {"loss_pmp": float(loss.detach().cpu())}
