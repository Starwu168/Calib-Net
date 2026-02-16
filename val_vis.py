# val_vis.py
from __future__ import annotations
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.config import load_yaml
from datasets.builder import build_dataset
from models.calib_pmp_net import CalibPMPNet
from utils.metrics_dc import DCMetrics, fmt_metrics


def _robust_clip(d: np.ndarray):
    m = d > 1e-3
    if m.sum() == 0:
        return d
    lo = np.percentile(d[m], 5)
    hi = np.percentile(d[m], 95)
    return np.clip(d, lo, hi)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./vis_out")
    ap.add_argument("--num", type=int, default=50)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(cfg["exp"]["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # dataset
    val_kwargs = dict(cfg["data"]["dataset_kwargs"]); val_kwargs["mode"] = "val"
    val_ds = build_dataset(cfg["data"]["dataset_class"], val_kwargs)

    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # model
    model = CalibPMPNet(cfg["model"]).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    met = DCMetrics(t_valid=cfg["loss"].get("t_valid", 1e-3))

    saved = 0
    for i, batch in enumerate(val_loader):
        rgb, dep_sp, Kcam, dep = batch
        rgb = rgb.to(device)
        dep_sp = dep_sp.to(device)
        Kcam = Kcam.to(device)
        dep = dep.to(device)

        pmp_out_list, _s_pred_list, _sprime = model(rgb, dep_sp, Kcam)
        pred = pmp_out_list[-1] if isinstance(pmp_out_list, (list, tuple)) else pmp_out_list

        met.update(pred, dep)

        gt_np = dep.squeeze().detach().cpu().numpy()
        pr_np = pred.squeeze().detach().cpu().numpy()

        gt_vis = _robust_clip(gt_np)
        pr_vis = _robust_clip(pr_np)

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(gt_vis, cmap="magma")
        ax1.set_title("GT")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(pr_vis, cmap="magma")
        ax2.set_title("Pred (Calib+PMP)")
        ax2.axis("off")

        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"vis_{i:05d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        saved += 1
        if saved >= args.num:
            break

    m = met.compute()
    print(f"[val_vis] {fmt_metrics(m)}")
    print(f"[done] saved {saved} images to {args.out_dir}")


if __name__ == "__main__":
    main()
