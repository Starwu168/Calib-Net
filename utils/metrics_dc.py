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
        self.t_valid = t_valid
        self.eps = eps
        self.reset()

    def reset(self):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sum_abs = torch.zeros(1, device=dev)
        self.sum_sq = torch.zeros(1, device=dev)
        self.sum_abs_rel = torch.zeros(1, device=dev)
        self.sum_sq_rel = torch.zeros(1, device=dev)
        self.sum_iabs = torch.zeros(1, device=dev)
        self.sum_isq = torch.zeros(1, device=dev)
        self.count = torch.zeros(1, device=dev)

        # ✅ delta counts
        self.sum_d1 = torch.zeros(1, device=dev)
        self.sum_d2 = torch.zeros(1, device=dev)
        self.sum_d3 = torch.zeros(1, device=dev)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        # pred/gt: (B,1,H,W) or (B,H,W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)

        m = (gt > self.t_valid)
        if m.sum() == 0:
            return

        p = pred[m].float()
        g = gt[m].float()

        diff = p - g
        abs_diff = diff.abs()

        self.sum_abs += abs_diff.sum()
        self.sum_sq += (diff * diff).sum()

        self.sum_abs_rel += (abs_diff / (g + self.eps)).sum()
        self.sum_sq_rel += ((diff * diff) / (g + self.eps)).sum()

        ip = 1.0 / (p + self.eps)
        ig = 1.0 / (g + self.eps)
        idiff = (ip - ig).abs()
        self.sum_iabs += idiff.sum()
        self.sum_isq += (idiff * idiff).sum()

        # ✅ delta metrics
        ratio = torch.maximum(p / (g + self.eps), g / (p + self.eps))
        self.sum_d1 += (ratio < 1.25).float().sum()
        self.sum_d2 += (ratio < (1.25 ** 2)).float().sum()
        self.sum_d3 += (ratio < (1.25 ** 3)).float().sum()

        self.count += torch.tensor([p.numel()], device=self.count.device, dtype=self.count.dtype)

    def all_reduce_(self):
        if not (dist.is_available() and dist.is_initialized()):
            return
        for t in [self.sum_abs, self.sum_sq, self.sum_abs_rel, self.sum_sq_rel,
                  self.sum_iabs, self.sum_isq, self.count,
                  self.sum_d1, self.sum_d2, self.sum_d3]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    def compute(self):
        c = torch.clamp(self.count, min=1.0)
        mae = (self.sum_abs / c).item()
        rmse = torch.sqrt(self.sum_sq / c).item()
        absrel = (self.sum_abs_rel / c).item()
        sqrel = (self.sum_sq_rel / c).item()
        imae = (self.sum_iabs / c).item()
        irmse = torch.sqrt(self.sum_isq / c).item()

        d1 = (self.sum_d1 / c).item()
        d2 = (self.sum_d2 / c).item()
        d3 = (self.sum_d3 / c).item()

        return {
            "MAE": mae, "RMSE": rmse, "iMAE": imae, "iRMSE": irmse,
            "AbsRel": absrel, "SqRel": sqrel,
            "d1": d1, "d2": d2, "d3": d3,
        }

def fmt_metrics(m: dict) -> str:
    # ✅ 把 d1 打出来（你只要求 d1，我也一起把 d2/d3带上，方便对照）
    return (f"MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  "
            f"iMAE={m['iMAE']:.3f}  iRMSE={m['iRMSE']:.3f}  "
            f"AbsRel={m['AbsRel']:.3f}  SqRel={m['SqRel']:.3f}  "
            f"d1={m['d1']:.3f}  d2={m['d2']:.3f}  d3={m['d3']:.3f}")
