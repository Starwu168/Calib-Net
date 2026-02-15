# val_vis.py
from __future__ import annotations
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.config import load_yaml
from datasets.builder import build_dataset, build_loader
from models.calib_net import CalibOnlyNet

import numpy as np

def masked_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-6):
    """
    pred/gt/mask: (H,W)
    mask: 0/1
    return dict metrics on mask
    """
    m = mask.astype(bool)
    if m.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "AbsRel": np.nan, "d1": np.nan, "d2": np.nan, "d3": np.nan}

    p = pred[m].astype(np.float64)
    g = gt[m].astype(np.float64)

    mae = np.mean(np.abs(p - g))
    rmse = np.sqrt(np.mean((p - g) ** 2))
    absrel = np.mean(np.abs(p - g) / (g + eps))

    # delta accuracy
    ratio = np.maximum(p / (g + eps), g / (p + eps))
    d1 = np.mean(ratio < 1.25)
    d2 = np.mean(ratio < 1.25 ** 2)
    d3 = np.mean(ratio < 1.25 ** 3)

    return {"MAE": mae, "RMSE": rmse, "AbsRel": absrel, "d1": d1, "d2": d2, "d3": d3}


def fmt_metrics(m: dict):
    return (f"MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  AbsRel={m['AbsRel']:.3f}  "
            f"d1={m['d1']:.3f} d2={m['d2']:.3f} d3={m['d3']:.3f}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="path to best.pth or epoch_xxx.pth")
    ap.add_argument("--out_dir", type=str, default="./vis_out")
    ap.add_argument("--num", type=int, default=20)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(cfg["exp"]["device"] if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    # dataset/loader
    val_kwargs = dict(cfg["data"]["dataset_kwargs"])
    val_kwargs["mode"] = "val"
    val_ds = build_dataset(cfg["data"]["dataset_class"], val_kwargs)
    val_loader = build_loader(val_ds, batch_size=1, num_workers=0, shuffle=False)

    # model
    model = CalibOnlyNet(cfg["model"]).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    def denorm_rgb(x):
        # inverse imagenet norm for visualization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        return (x * std + mean).clamp(0, 1)

    saved = 0
    for i, batch in enumerate(val_loader):
        rgb, dep_sp, K, dep = batch
        rgb = rgb.to(device)
        dep_sp = dep_sp.to(device)
        dep = dep.to(device)

        # forward: list of per-scale S'_l
        s_pred_list = model(rgb, dep_sp)
        # 取 scale0 输出并 upsample 回原图尺寸
        s0 = s_pred_list[0]
        s_up = F.interpolate(s0, size=dep_sp.shape[-2:], mode="nearest")
        # 强制只保留输入点位置（硬写死）
        s_up = s_up * (dep_sp > 0).float()

        # to numpy
        rgb_vis = denorm_rgb(rgb).squeeze(0).permute(1,2,0).detach().cpu().numpy()
        s_in = dep_sp.squeeze().detach().cpu().numpy()
        s_out = s_up.squeeze().detach().cpu().numpy()
        gt = dep.squeeze().detach().cpu().numpy()

        thr = 1e-4
        nz_in = int((s_in > thr).sum())
        nz_out = int((s_out > thr).sum())

        ratio = nz_out / (nz_in + 1e-6)

        print(f"[{i:05d}] nonzero count: input={nz_in}, calib={nz_out}, ratio={ratio:.3f}")

        # 只在 radar mask 处看误差（可选）
        mask = (s_in > 0).astype(np.float32)
        # baseline error map + metrics
        err_in = np.abs(s_in - gt) * mask
        met_in = masked_metrics(s_in, gt, mask)

        # calibrated error map + metrics
        err_out = np.abs(s_out - gt) * mask
        err_all = np.abs(s_out - gt)
        met_out = masked_metrics(s_out, gt, mask)

        print(f"[{i:05d}]  input: {fmt_metrics(met_in)}")
        print(f"[{i:05d}]  calib: {fmt_metrics(met_out)}")
        err = np.abs(s_out - gt) * mask

        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(2, 3, 1);
        ax1.imshow(rgb_vis);
        ax1.set_title("RGB");
        ax1.axis("off")
        ax2 = fig.add_subplot(2, 3, 2);
        ax2.imshow(s_in, cmap="magma");
        ax2.set_title("Sparse S (input)");
        ax2.axis("off")
        ax3 = fig.add_subplot(2, 3, 3);
        ax3.imshow(s_out, cmap="magma");
        ax3.set_title("Sparse S' (calibrated)");
        ax3.axis("off")
        ax4 = fig.add_subplot(2, 3, 4);
        ax4.imshow(err_in, cmap="magma");
        ax4.set_title("|S - GT| on mask");
        ax4.axis("off")
        ax5 = fig.add_subplot(2, 3, 5);
        ax5.imshow(err_out, cmap="magma");
        ax5.set_title("|S' - GT| on mask");
        ax5.axis("off")
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.imshow(err_all, cmap="magma")
        ax6.set_title("|S' - GT| (full)")
        ax6.axis("off")

        fig.tight_layout()

        out_path = os.path.join(args.out_dir, f"vis_{i:05d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        saved += 1
        if saved >= args.num:
            break

    print(f"[done] saved {saved} visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()
