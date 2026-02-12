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
    val_ds = build_dataset(cfg["data"]["dataset_class"], cfg["data"]["dataset_kwargs"] | {"mode": "val"})
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
        s_up = F.interpolate(s0, size=dep_sp.shape[-2:], mode="bilinear", align_corners=False)

        # to numpy
        rgb_vis = denorm_rgb(rgb).squeeze(0).permute(1,2,0).detach().cpu().numpy()
        s_in = dep_sp.squeeze().detach().cpu().numpy()
        s_out = s_up.squeeze().detach().cpu().numpy()
        gt = dep.squeeze().detach().cpu().numpy()

        # 只在 radar mask 处看误差（可选）
        mask = (s_in > 0).astype(np.float32)
        err = np.abs(s_out - gt) * mask

        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(2,2,1); ax1.imshow(rgb_vis); ax1.set_title("RGB"); ax1.axis("off")
        ax2 = fig.add_subplot(2,2,2); ax2.imshow(s_in,  cmap="magma"); ax2.set_title("Sparse S (input)"); ax2.axis("off")
        ax3 = fig.add_subplot(2,2,3); ax3.imshow(s_out, cmap="magma"); ax3.set_title("Sparse S' (calibrated)"); ax3.axis("off")
        ax4 = fig.add_subplot(2,2,4); ax4.imshow(err,  cmap="magma"); ax4.set_title("|S' - GT| on mask"); ax4.axis("off")
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
