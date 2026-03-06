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


def _unpack_batch(batch):
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Unexpected batch type: {type(batch)}")
    if len(batch) < 4:
        raise ValueError(f"Batch must contain at least 4 items, got {len(batch)}")
    rgb, dep_sp, Kcam, dep_interp = batch[:4]
    dep_sparse = batch[4] if len(batch) >= 5 else None
    return rgb, dep_sp, Kcam, dep_interp, dep_sparse


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="/data00/wsx/code/calibnet/configs/calib_train.yaml")
    ap.add_argument("--ckpt", type=str, default="/data00/wsx/code/calibnet/runs/calib_pmp_zju/best.pth")
    ap.add_argument("--out_dir", type=str, default="/data00/wsx/code/calibnet/vis_out")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument(
        "--depth_max_list",
        type=float,
        nargs="+",
        default=None,
        help="Evaluate metrics for multiple depth_max values in one run, e.g. --depth_max_list 50 70 80",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(cfg["exp"]["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # dataset (fixed to official test split)
    test_kwargs = dict(cfg["data"]["dataset_kwargs"]); test_kwargs["mode"] = "test"
    test_ds = build_dataset(cfg["data"]["dataset_class"], test_kwargs)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # model
    model = CalibPMPNet(cfg["model"]).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    if args.depth_max_list:
        depth_max_list = [float(x) for x in args.depth_max_list]
    else:
        depth_max_list = [cfg["loss"].get("depth_max", None)]

    metrics_map = {
        dm: DCMetrics(t_valid=cfg["loss"].get("t_valid", 1e-3))
        for dm in depth_max_list
    }

    saved = 0
    for i, batch in enumerate(test_loader):
        rgb, dep_sp, Kcam, dep_interp, dep_sparse = _unpack_batch(batch)
        rgb = rgb.to(device)
        dep_sp = dep_sp.to(device)
        Kcam = Kcam.to(device)
        dep_interp = dep_interp.to(device)
        dep_sparse = dep_sparse.to(device) if dep_sparse is not None else None

        pmp_out_list, _s_pred_list, _sprime = model(rgb, dep_sp, Kcam)
        pred = pmp_out_list[-1] if isinstance(pmp_out_list, (list, tuple)) else pmp_out_list

        if dep_sparse is not None and cfg["loss"].get("metrics_use_sparse_gt", True):
            eval_tgt = dep_sparse
            eval_mask = dep_sparse > float(cfg["loss"].get("t_valid", 1e-3))
        else:
            eval_tgt = dep_interp
            eval_mask = None

        for depth_max, met in metrics_map.items():
            met.update(
                pred,
                eval_tgt,
                valid_mask=eval_mask,
                depth_min=cfg["loss"].get("depth_min", None),
                depth_max=depth_max,
            )

        sparse_np = dep_sparse.squeeze().detach().cpu().numpy() if dep_sparse is not None else np.zeros_like(dep_interp.squeeze().detach().cpu().numpy())
        interp_np = dep_interp.squeeze().detach().cpu().numpy()
        pr_np = pred.squeeze().detach().cpu().numpy()

        sparse_vis = _robust_clip(sparse_np)
        interp_vis = _robust_clip(interp_np)
        pr_vis = _robust_clip(pr_np)

        fig = plt.figure(figsize=(15, 4))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(sparse_vis, cmap="magma")
        ax1.set_title("Sparse GT")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(interp_vis, cmap="magma")
        ax2.set_title("GT Interp")
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(pr_vis, cmap="magma")
        ax3.set_title("Pred (Calib+PMP)")
        ax3.axis("off")

        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"vis_{i:05d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        saved += 1
        if saved >= args.num:
            break

    for depth_max, met in metrics_map.items():
        m = met.compute()
        print(f"[val_vis][depth_max={depth_max}] {fmt_metrics(m)}")
    print(f"[done] saved {saved} images to {args.out_dir}")


if __name__ == "__main__":
    main()
