from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np
from PIL import Image

from tools.depth_io import ensure_dir, load_depth_png_16bit, load_mask_png
from tools.nuscenes_geometry import colorize_depth


def overlay_radar(rgb: np.ndarray, radar_uvd: np.ndarray) -> np.ndarray:
    vis = rgb.copy()
    if radar_uvd.size == 0:
        return vis
    for u, v, d in radar_uvd[:, :3]:
        x = int(np.floor(u))
        y = int(np.floor(v))
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
    return vis


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="preprocessed output_root")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_vis", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    base = os.path.join(args.root, "data")
    debug_dir = os.path.join(args.root, "debug_vis")
    ensure_dir(debug_dir)

    split_file = os.path.join(base, f"{args.split}.txt")
    with open(split_file, "r", encoding="utf-8") as f:
        names = [x.strip() for x in f.readlines() if x.strip()]

    random.seed(args.seed)
    if len(names) > args.num_vis:
        names = random.sample(names, args.num_vis)

    radar_counts = []
    gt_counts = []
    gti_counts = []

    for name in names:
        img = np.array(Image.open(os.path.join(base, "image", f"{name}.png")).convert("RGB"))
        radar = np.load(os.path.join(base, "radar", f"{name}.npy"))
        gt = load_depth_png_16bit(os.path.join(base, "gt", f"{name}.png"))
        gti = load_depth_png_16bit(os.path.join(base, "gt_interp", f"{name}.png"))
        mr = load_mask_png(os.path.join(base, "valid_mask_raw", f"{name}.png"))
        mi = load_mask_png(os.path.join(base, "valid_mask_interp", f"{name}.png"))

        radar_counts.append(int(radar.shape[0]))
        gt_counts.append(int((gt > 0).sum()))
        gti_counts.append(int((gti > 0).sum()))

        rgb_radar = overlay_radar(img, radar)
        gt_vis = colorize_depth(gt)
        gti_vis = colorize_depth(gti)

        inc = ((mi > 0) & (mr == 0)).astype(np.uint8) * 255
        inc_vis = cv2.applyColorMap(inc, cv2.COLORMAP_HOT)

        row1 = np.concatenate([img[:, :, ::-1], rgb_radar[:, :, ::-1]], axis=1)
        row2 = np.concatenate([gt_vis, gti_vis], axis=1)
        row3 = np.concatenate([cv2.cvtColor(mr * 255, cv2.COLOR_GRAY2BGR), inc_vis], axis=1)
        canvas = np.concatenate([row1, row2, row3], axis=0)

        out_path = os.path.join(debug_dir, f"{name}.jpg")
        cv2.imwrite(out_path, canvas)

    print(f"samples={len(names)}")
    if len(names) > 0:
        print(f"avg_radar_points={np.mean(radar_counts):.2f}")
        print(f"avg_gt_valid={np.mean(gt_counts):.2f}")
        print(f"avg_gt_interp_valid={np.mean(gti_counts):.2f}")
        print(f"interp_gain_ratio={(np.mean(gti_counts) / max(np.mean(gt_counts), 1.0)):.4f}")
    print(f"debug_dir={debug_dir}")


if __name__ == "__main__":
    main()