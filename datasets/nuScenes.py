from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from datasets.ZJU import (
    BaseDataset,
    ceil_to_multiple,
    choose_biased_crop_box,
    read_depth_png_16bit,
    rasterize_radar_to_sparse,
    update_K_for_crop,
    update_K_for_resize,
)
from tools.depth_io import load_meta_npz


class NuScenesDepthCompletion(BaseDataset):
    """
    Calib-Net style nuScenes dataset from offline preprocessed static files.

    Directory:
        <root>/data/
            image/
            radar/
            gt/
            gt_interp/
            meta/
            train.txt
            val.txt
            test.txt

    Return:
        rgb_t, dep_sp_t, K, dep_t, dep_sparse_t
    """

    def __init__(
        self,
        mode: str,
        path: str,
        num_sample: int = 500,
        mul_factor: float = 1.0,
        num_mask: int = 1,
        rand_scale: bool = True,
        enable_color_aug: bool = True,
        enable_hflip: bool = True,
        target_h: int = 288,
        target_w: int = 800,
        sky_bias_ratio: float = 0.05,
        sparse_gt_dirname: str = "gt",
        return_sparse_gt: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(path, mode)
        if mode not in ("train", "val", "test"):
            raise NotImplementedError

        self.mode = mode
        self.num_sample = int(num_sample)
        self.mul_factor = float(mul_factor)
        self.num_mask = int(num_mask)

        self.rand_scale = bool(rand_scale)
        self.enable_color_aug = bool(enable_color_aug)
        self.enable_hflip = bool(enable_hflip)

        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.sky_bias_ratio = float(sky_bias_ratio)
        self.return_sparse_gt = bool(return_sparse_gt)
        self.sparse_gt_dirname = str(sparse_gt_dirname)

        self.base_dir = os.path.join(path, "data")
        txt_path = os.path.join(self.base_dir, f"{mode}.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"missing split file: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f.readlines() if ln.strip()]

        self.sample_list = []
        for name in names:
            pure = os.path.splitext(name)[0]
            img_path = os.path.join(self.base_dir, "image", f"{pure}.png")
            radar_path = os.path.join(self.base_dir, "radar", f"{pure}.npy")
            gt_path = os.path.join(self.base_dir, "gt_interp", f"{pure}.png")
            sparse_gt_path = os.path.join(self.base_dir, self.sparse_gt_dirname, f"{pure}.png")
            meta_path = os.path.join(self.base_dir, "meta", f"{pure}.npz")

            if os.path.exists(img_path) and os.path.exists(radar_path) and os.path.exists(meta_path):
                if mode == "test":
                    gt_ok = True
                else:
                    gt_ok = os.path.exists(gt_path)

                if gt_ok:
                    self.sample_list.append(
                        {
                            "name": pure,
                            "img": img_path,
                            "radar": radar_path,
                            "gt": gt_path if os.path.exists(gt_path) else None,
                            "sparse_gt": sparse_gt_path if os.path.exists(sparse_gt_path) else None,
                            "meta": meta_path,
                        }
                    )

        print(f"[NuScenesDepthCompletion] mode={mode}, samples={len(self.sample_list)}")

        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)

    def __len__(self):
        if self.mode == "val":
            return len(self.sample_list) * self.num_mask
        return len(self.sample_list)

    def _rng_for_val(self, idx: int) -> random.Random:
        return random.Random(2027 + idx)

    def __getitem__(self, idx: int):
        if self.mode == "val":
            base_idx = idx // self.num_mask
            rng = self._rng_for_val(idx)
        else:
            base_idx = idx
            rng = random

        sample = self.sample_list[base_idx]

        rgb = Image.open(sample["img"]).convert("RGB")
        radar = np.load(sample["radar"]).astype(np.float32)
        meta = load_meta_npz(sample["meta"])

        K = torch.tensor(np.asarray(meta["K"], dtype=np.float32), dtype=torch.float32)

        if sample["gt"] is not None:
            gt = read_depth_png_16bit(sample["gt"])
        else:
            H0 = int(meta["height"])
            W0 = int(meta["width"])
            gt = np.zeros((H0, W0), dtype=np.float32)

        if sample["sparse_gt"] is not None:
            sparse_gt = read_depth_png_16bit(sample["sparse_gt"])
        else:
            sparse_gt = np.zeros_like(gt, dtype=np.float32)

        H, W = gt.shape
        assert rgb.size == (W, H), f"RGB/GT size mismatch: rgb={rgb.size}, gt={(W, H)}"

        if radar.ndim != 2 or radar.shape[1] < 3:
            raise ValueError(f"radar npy must be (N,3+) with [u,v,d], got {radar.shape}")

        radar_uvd = radar[:, :3].astype(np.float32)
        N = radar_uvd.shape[0]
        if N > self.num_sample:
            if self.mode == "val":
                ids = rng.sample(range(N), self.num_sample)
                radar_uvd = radar_uvd[ids]
            else:
                ids = np.random.choice(N, self.num_sample, replace=False)
                radar_uvd = radar_uvd[ids]

        cx0 = float(K[0, 2].item())
        cy0 = float(K[1, 2].item())

        top, left = choose_biased_crop_box(
            H=H,
            W=W,
            target_h=self.target_h,
            target_w=self.target_w,
            cx=cx0,
            cy=cy0,
            sky_bias_ratio=self.sky_bias_ratio,
        )

        rgb = TF.crop(rgb, top=top, left=left, height=self.target_h, width=self.target_w)
        gt = gt[top:top + self.target_h, left:left + self.target_w]
        sparse_gt = sparse_gt[top:top + self.target_h, left:left + self.target_w]
        K = update_K_for_crop(K, left=left, top=top)

        dep_sp = rasterize_radar_to_sparse(
            radar_uvd=radar_uvd,
            top=top,
            left=left,
            target_h=self.target_h,
            target_w=self.target_w,
        )

        dep = gt.astype(np.float32)
        dep_sparse_gt = sparse_gt.astype(np.float32)

        if self.mode == "train":
            if self.enable_hflip and rng.random() < 0.5:
                rgb = TF.hflip(rgb)
                dep = np.ascontiguousarray(np.fliplr(dep))
                dep_sp = np.ascontiguousarray(np.fliplr(dep_sp))
                dep_sparse_gt = np.ascontiguousarray(np.fliplr(dep_sparse_gt))

                Wc = self.target_w
                K[0, 2] = (Wc - 1.0) - K[0, 2]

            if self.enable_color_aug and rng.random() < 0.5:
                rgb = self.color_jitter(rgb)

            if self.rand_scale:
                scale = rng.uniform(1.0, 1.2)
                new_h = ceil_to_multiple(int(round(self.target_h * scale)), 16)
                new_w = ceil_to_multiple(int(round(self.target_w * scale)), 16)
                new_h = max(new_h, self.target_h)
                new_w = max(new_w, self.target_w)

                rgb = TF.resize(rgb, size=[new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
                dep = np.array(Image.fromarray(dep).resize((new_w, new_h), resample=Image.NEAREST), dtype=np.float32)
                dep_sp = np.array(Image.fromarray(dep_sp).resize((new_w, new_h), resample=Image.NEAREST), dtype=np.float32)
                dep_sparse_gt = np.array(
                    Image.fromarray(dep_sparse_gt).resize((new_w, new_h), resample=Image.NEAREST),
                    dtype=np.float32,
                )

                sx = new_w / self.target_w
                sy = new_h / self.target_h
                K = update_K_for_resize(K, sx=sx, sy=sy)

                left2 = (new_w - self.target_w) // 2
                top2 = (new_h - self.target_h) // 2

                rgb = TF.crop(rgb, top=top2, left=left2, height=self.target_h, width=self.target_w)
                dep = dep[top2:top2 + self.target_h, left2:left2 + self.target_w]
                dep_sp = dep_sp[top2:top2 + self.target_h, left2:left2 + self.target_w]
                dep_sparse_gt = dep_sparse_gt[top2:top2 + self.target_h, left2:left2 + self.target_w]
                K = update_K_for_crop(K, left=left2, top=top2)

        rgb_t = self.norm(self.to_tensor(rgb))
        dep_sp_t = torch.from_numpy(dep_sp).unsqueeze(0)
        dep_t = torch.from_numpy(dep).unsqueeze(0)
        dep_sparse_t = torch.from_numpy(dep_sparse_gt).unsqueeze(0)

        dep_sp_t = dep_sp_t * self.mul_factor
        dep_t = dep_t * self.mul_factor
        dep_sparse_t = dep_sparse_t * self.mul_factor

        if self.return_sparse_gt:
            return rgb_t, dep_sp_t, K, dep_t, dep_sparse_t
        return rgb_t, dep_sp_t, K, dep_t