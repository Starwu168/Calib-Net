from __future__ import annotations

import torch
import torch.distributed as dist


class DCMetrics:
    """
    Metrics for depth completion:
      MAE, RMSE, iMAE, iRMSE, AbsRel, SqRel
    computed on mask: gt > t_valid
    aggregated as the mean of per-image metrics.
    """

    def __init__(self, t_valid: float = 1e-3, eps: float = 1e-6, protocol: str = "dc"):
        self.t_valid = t_valid
        self.eps = eps
        self.protocol = str(protocol).strip().lower()
        if self.protocol not in {"dc", "radarcam"}:
            raise ValueError(f"Unsupported metrics protocol: {protocol}")
        self.reset()

    def reset(self):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sum_mae = torch.zeros(1, device=dev)
        self.sum_rmse = torch.zeros(1, device=dev)
        self.sum_abs_rel = torch.zeros(1, device=dev)
        self.sum_sq_rel = torch.zeros(1, device=dev)
        self.sum_imae = torch.zeros(1, device=dev)
        self.sum_irmse = torch.zeros(1, device=dev)
        self.sum_d1 = torch.zeros(1, device=dev)
        self.sum_d2 = torch.zeros(1, device=dev)
        self.sum_d3 = torch.zeros(1, device=dev)
        self.count = torch.zeros(1, device=dev)

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        depth_min: float | None = None,
        depth_max: float | None = None,
    ):
        # pred/gt: (B,1,H,W) or (B,H,W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        if valid_mask is not None and valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)

        for b in range(pred.shape[0]):
            p_b = pred[b:b + 1]
            g_b = gt[b:b + 1]

            m = g_b > self.t_valid
            if depth_min is not None:
                m = m & (g_b > float(depth_min))
            if depth_max is not None:
                m = m & (g_b < float(depth_max))
            if valid_mask is not None:
                m = m & (valid_mask[b:b + 1] > 0)
            if m.sum() == 0:
                continue

            p = p_b[m].float()
            g = g_b[m].float()

            if self.protocol == "radarcam":
                p_mm = p * 1000.0
                g_mm = g * 1000.0
                diff_mm = p_mm - g_mm
                abs_diff_mm = diff_mm.abs()
                mae = abs_diff_mm.mean()
                rmse = torch.sqrt((diff_mm * diff_mm).mean())
                abs_rel = (abs_diff_mm / (g_mm + self.eps)).mean()
                sq_rel = ((diff_mm * diff_mm) / (g_mm + self.eps)).mean()

                p_km = p * 0.001
                g_km = g * 0.001
                idiff = (1.0 / (p_km + self.eps) - 1.0 / (g_km + self.eps)).abs()
                imae = idiff.mean()
                irmse = torch.sqrt((idiff * idiff).mean())
            else:
                diff = p - g
                abs_diff = diff.abs()
                mae = abs_diff.mean()
                rmse = torch.sqrt((diff * diff).mean())
                abs_rel = (abs_diff / (g + self.eps)).mean()
                sq_rel = ((diff * diff) / (g + self.eps)).mean()

                ip = 1.0 / (p + self.eps)
                ig = 1.0 / (g + self.eps)
                idiff = (ip - ig).abs()
                imae = idiff.mean()
                irmse = torch.sqrt((idiff * idiff).mean())

            ratio = torch.maximum(p / (g + self.eps), g / (p + self.eps))
            d1 = (ratio < 1.25).float().mean()
            d2 = (ratio < (1.25 ** 2)).float().mean()
            d3 = (ratio < (1.25 ** 3)).float().mean()

            self.sum_mae += mae
            self.sum_rmse += rmse
            self.sum_abs_rel += abs_rel
            self.sum_sq_rel += sq_rel
            self.sum_imae += imae
            self.sum_irmse += irmse
            self.sum_d1 += d1
            self.sum_d2 += d2
            self.sum_d3 += d3
            self.count += 1.0

    def all_reduce_(self):
        if not (dist.is_available() and dist.is_initialized()):
            return
        for t in [
            self.sum_mae,
            self.sum_rmse,
            self.sum_abs_rel,
            self.sum_sq_rel,
            self.sum_imae,
            self.sum_irmse,
            self.sum_d1,
            self.sum_d2,
            self.sum_d3,
            self.count,
        ]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    def compute(self):
        c = torch.clamp(self.count, min=1.0)
        mae = (self.sum_mae / c).item()
        rmse = (self.sum_rmse / c).item()
        absrel = (self.sum_abs_rel / c).item()
        sqrel = (self.sum_sq_rel / c).item()
        imae = (self.sum_imae / c).item()
        irmse = (self.sum_irmse / c).item()
        d1 = (self.sum_d1 / c).item()
        d2 = (self.sum_d2 / c).item()
        d3 = (self.sum_d3 / c).item()

        return {
            "MAE": mae,
            "RMSE": rmse,
            "iMAE": imae,
            "iRMSE": irmse,
            "AbsRel": absrel,
            "SqRel": sqrel,
            "d1": d1,
            "d2": d2,
            "d3": d3,
        }


def fmt_metrics(m: dict) -> str:
    return (
        f"MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  "
        f"iMAE={m['iMAE']:.3f}  iRMSE={m['iRMSE']:.3f}  "
        f"AbsRel={m['AbsRel']:.3f}  SqRel={m['SqRel']:.3f}  "
        f"d1={m['d1']:.3f}  d2={m['d2']:.3f}  d3={m['d3']:.3f}"
    )
