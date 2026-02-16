# utils/metrics_dc.py
from __future__ import annotations
import torch
import torch.distributed as dist


class DCMetrics:
    """
    Metrics for depth completion:
      MAE, RMSE, iMAE, iRMSE, AbsRel, SqRel
    computed on mask: gt > t_valid
    """
    def __init__(self, t_valid: float = 1e-3, eps: float = 1e-6):
        self.t_valid = float(t_valid)
        self.eps = float(eps)
        self.reset()

    def reset(self):
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_abs_inv = 0.0
        self.sum_sq_inv = 0.0
        self.sum_absrel = 0.0
        self.sum_sqrel = 0.0
        self.count = 0.0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        # (B,1,H,W)
        valid = (gt > self.t_valid)
        if valid.sum() == 0:
            return

        p = pred[valid].double()
        g = gt[valid].double()
        diff = p - g

        self.sum_abs += diff.abs().sum().item()
        self.sum_sq += (diff * diff).sum().item()

        p_inv = 1.0 / (p + self.eps)
        g_inv = 1.0 / (g + self.eps)
        diff_inv = p_inv - g_inv

        self.sum_abs_inv += diff_inv.abs().sum().item()
        self.sum_sq_inv += (diff_inv * diff_inv).sum().item()

        self.sum_absrel += (diff.abs() / (g + self.eps)).sum().item()
        self.sum_sqrel += ((diff * diff) / (g + self.eps)).sum().item()

        self.count += float(g.numel())

    def all_reduce_(self):
        if not (dist.is_available() and dist.is_initialized()):
            return
        t = torch.tensor([
            self.sum_abs, self.sum_sq,
            self.sum_abs_inv, self.sum_sq_inv,
            self.sum_absrel, self.sum_sqrel,
            self.count
        ], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        (self.sum_abs, self.sum_sq,
         self.sum_abs_inv, self.sum_sq_inv,
         self.sum_absrel, self.sum_sqrel,
         self.count) = t.tolist()

    def compute(self) -> dict:
        c = max(self.count, 1.0)
        mae = self.sum_abs / c
        rmse = (self.sum_sq / c) ** 0.5
        imae = self.sum_abs_inv / c
        irmse = (self.sum_sq_inv / c) ** 0.5
        absrel = self.sum_absrel / c
        sqrel = self.sum_sqrel / c
        return {"MAE": mae, "RMSE": rmse, "iMAE": imae, "iRMSE": irmse, "AbsRel": absrel, "SqRel": sqrel}


def fmt_metrics(m: dict) -> str:
    return (f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
            f"iMAE={m['iMAE']:.4f}  iRMSE={m['iRMSE']:.4f}  "
            f"AbsRel={m['AbsRel']:.4f}  SqRel={m['SqRel']:.4f}")
