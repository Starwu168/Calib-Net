# datasets/zju.py
from __future__ import annotations
import os
import math
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class BaseDataset(torch.utils.data.Dataset):
    """
    简化版 NYU/BPNet BaseDataset 风格：
    - 提供 ToNumpy transform
    - 提供 mode 管理
    """
    def __init__(self, root, mode: str):
        super().__init__()
        self.root = root
        self.mode = mode

    class ToNumpy:
        def __call__(self, pic):
            # pic 可能是 PIL 或 np.ndarray
            if isinstance(pic, np.ndarray):
                return pic
            # PIL Image -> np
            return np.array(pic, dtype=np.float32)

def clamp_int(x, lo, hi):
    return int(max(lo, min(hi, x)))
def read_depth_png_16bit(path: str) -> np.ndarray:
    """ZJU 的深度图通常是 16-bit PNG，除以 256 得到米。"""
    d = np.array(Image.open(path), dtype=np.float32)
    return d / 256.0


def update_K_for_crop(K: torch.Tensor, left: int, top: int) -> torch.Tensor:
    """
    严谨：crop 后像素原点移动 => cx,cy 减去 left/top。
    右/下裁剪不影响 cx/cy。:contentReference[oaicite:2]{index=2}
    """
    K2 = K.clone()
    K2[0, 2] -= float(left)
    K2[1, 2] -= float(top)
    return K2


def update_K_for_resize(K: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    """
    严谨：resize 后 fx,fy,cx,cy 按缩放比例缩放。:contentReference[oaicite:3]{index=3}
    """
    K2 = K.clone()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


def choose_centered_crop_box(
    H: int, W: int,
    target_h: int, target_w: int,
    cx: float, cy: float
):
    """
    以 principal point (cx,cy) 尽量为中心的裁剪窗口（整数像素）。
    若超出边界则 clamp。
    返回: top, left
    """
    # 以 (cx,cy) 为中心，left = cx - target_w/2
    left = int(round(cx - target_w / 2.0))
    top = int(round(cy - target_h / 2.0))

    left = clamp_int(left, 0, W - target_w)
    top = clamp_int(top, 0, H - target_h)
    return top, left


def rasterize_radar_to_sparse(
    radar_uvd: np.ndarray,
    H: int, W: int,
    top: int, left: int,
    target_h: int, target_w: int
) -> np.ndarray:
    """
    将 radar 点 (u,v,d) 栅格化到裁剪后的 sparse depth 图 S。
    规则：落在裁剪框内则写入；同一像素多点取更近(更小深度)。
    radar_uvd: (N,3)  u,v 以原图像素坐标为基准。
    """
    S = np.zeros((target_h, target_w), dtype=np.float32)
    if radar_uvd.size == 0:
        return S

    u = radar_uvd[:, 0]
    v = radar_uvd[:, 1]
    d = radar_uvd[:, 2]

    # 只保留深度>0
    valid = d > 0
    u = u[valid]; v = v[valid]; d = d[valid]
    if d.size == 0:
        return S

    # 落在 crop box 内
    inside = (u >= left) & (u < left + target_w) & (v >= top) & (v < top + target_h)
    u = u[inside]; v = v[inside]; d = d[inside]
    if d.size == 0:
        return S

    # 转到裁剪后的坐标系
    uu = np.floor(u - left).astype(np.int32)
    vv = np.floor(v - top).astype(np.int32)

    # 写入：同一像素取 min depth
    for x, y, z in zip(uu, vv, d):
        cur = S[y, x]
        if cur <= 0:
            S[y, x] = z
        else:
            S[y, x] = min(cur, z)
    return S


class ZJU4DRadarCam(BaseDataset):
    """
    BPNet/NYU 风格输出：
      rgb:   (3,H,W) float tensor normalized
      dep_sp:(1,H,W) sparse radar depth
      Kcam:  (3,3)   float tensor
      dep:   (1,H,W) dense/interp GT depth (meters)

    目录结构：
    ZJU-4DRadarCam/
      data/
        image/
        radar/        *.npy (N,3) [u,v,d] in pixel coords of original image
        gt_interp/    *.png 16bit depth
        train.txt val.txt ...
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
        # 你要求 H in [300,500], multiple of 16
        target_h: int = 384,
        target_w: int = 960,
        *args, **kwargs
    ):
        super().__init__(path, mode)
        if mode not in ("train", "val", "test"):
            raise NotImplementedError

        assert 300 <= target_h <= 500 and target_h % 16 == 0
        assert target_w % 16 == 0

        self.mode = mode
        self.num_sample = int(num_sample)
        self.mul_factor = float(mul_factor)
        self.num_mask = int(num_mask)
        self.rand_scale = bool(rand_scale)
        self.enable_color_aug = bool(enable_color_aug)
        self.enable_hflip = bool(enable_hflip)

        self.target_h = int(target_h)
        self.target_w = int(target_w)

        self.base_dir = os.path.join(path, "data")
        txt_path = os.path.join(self.base_dir, f"{mode}.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"missing split file: {txt_path}")

        # === 相机内参（你示例给的）===
        self.Kcam0 = torch.tensor(
            [
                [6.4367089874925102e+02, 0.0, 6.3134981290491908e+02],
                [0.0, 6.4354259444006982e+02, 3.6731040913213513e+02],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32
        )

        with open(txt_path, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f.readlines() if ln.strip()]

        self.sample_list = []
        for name in names:
            pure = os.path.splitext(name)[0]
            img_path = os.path.join(self.base_dir, "image", f"{pure}.png")
            radar_path = os.path.join(self.base_dir, "radar", f"{pure}.npy")
            gt_path = os.path.join(self.base_dir, "gt_interp", f"{pure}.png")  # 用插值 gt

            if os.path.exists(img_path) and os.path.exists(radar_path) and os.path.exists(gt_path):
                self.sample_list.append({"img": img_path, "radar": radar_path, "gt": gt_path})

        print(f"[ZJU4DRadarCam] mode={mode}, samples={len(self.sample_list)}")

        # === transforms ===
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # color aug 参考 BPNet 常见写法（强度可调）
        self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)

    def __len__(self):
        if self.mode == "train":
            return len(self.sample_list)
        if self.mode == "val":
            return len(self.sample_list) * self.num_mask
        return len(self.sample_list)

    def _rng_for_val(self, idx: int) -> random.Random:
        # 让 val 的 mask/采样可复现
        return random.Random(2027 + idx)

    def __getitem__(self, idx: int):
        if self.mode == "val":
            seed = idx % self.num_mask
            base_idx = idx // self.num_mask
            rng = self._rng_for_val(idx)
        else:
            seed = 0
            base_idx = idx
            rng = random

        sample = self.sample_list[base_idx]
        rgb = Image.open(sample["img"]).convert("RGB")
        gt = read_depth_png_16bit(sample["gt"])  # (H,W) meters
        radar = np.load(sample["radar"])         # 期望 (N,3) [u,v,d]

        H, W = gt.shape[0], gt.shape[1]
        assert rgb.size == (W, H), f"RGB/GT size mismatch: rgb={rgb.size}, gt={(W,H)}"

        # --------------------------
        # 1) radar subsample (num_sample)
        # --------------------------
        if radar.ndim != 2 or radar.shape[1] < 3:
            raise ValueError(f"radar npy must be (N,3+) with [u,v,d], got {radar.shape}")

        radar_uvd = radar[:, :3].astype(np.float32)
        N = radar_uvd.shape[0]
        if N > self.num_sample:
            # train: random; val: deterministic
            if self.mode == "val":
                ids = rng.sample(range(N), self.num_sample)
            else:
                ids = np.random.choice(N, self.num_sample, replace=False)
            radar_uvd = radar_uvd[ids]

        # --------------------------
        # 2) 选择裁剪窗口：以 principal point 为中心
        #    这会“尽量保持中心点不动”，但 K 仍需更新（cx,cy 减 left/top）:contentReference[oaicite:4]{index=4}
        # --------------------------
        K = self.Kcam0.clone()
        cx0 = float(K[0, 2].item())
        cy0 = float(K[1, 2].item())

        # 在 ZJU 原“上下裁剪”基础上做轻微变化：
        #   仍然倾向保留中间区域，但最终裁剪窗口是 (target_h,target_w)，并在左右也裁剪。
        top, left = choose_centered_crop_box(
            H=H, W=W,
            target_h=self.target_h, target_w=self.target_w,
            cx=cx0, cy=cy0
        )

        # crop RGB / GT
        rgb = TF.crop(rgb, top=top, left=left, height=self.target_h, width=self.target_w)
        gt = gt[top:top + self.target_h, left:left + self.target_w]

        # update K for crop
        K = update_K_for_crop(K, left=left, top=top)

        # --------------------------
        # 3) rasterize radar to sparse depth S (after crop)
        # --------------------------
        dep_sp_np = rasterize_radar_to_sparse(
            radar_uvd=radar_uvd,
            H=H, W=W,
            top=top, left=left,
            target_h=self.target_h, target_w=self.target_w
        )

        # dep / dep_sp 都是 meters
        dep = gt.astype(np.float32)
        dep_sp = dep_sp_np.astype(np.float32)

        # --------------------------
        # 4) BPNet 风格增强（轻量 + K 严谨更新）
        #    - hflip: 需要更新 cx = (W-1) - cx
        #    - resize(scale): 需要更新 fx,fy,cx,cy 乘以 scale :contentReference[oaicite:5]{index=5}
        # --------------------------
        if self.mode == "train":
            # (a) random horizontal flip
            if self.enable_hflip and rng.random() < 0.5:
                rgb = TF.hflip(rgb)
                dep = np.ascontiguousarray(np.fliplr(dep))
                dep_sp = np.ascontiguousarray(np.fliplr(dep_sp))

                # K update for hflip (pixel coords): cx' = (W-1) - cx
                Wc = self.target_w
                K[0, 2] = (Wc - 1.0) - K[0, 2]

            # (b) random color jitter (RGB only)
            if self.enable_color_aug and rng.random() < 0.5:
                rgb = self.color_jitter(rgb)

            # (c) optional random resize scale (keep aspect)
            # 说明：resize 会改变像素尺寸，所以 K 要按比例缩放；之后我们再 center-crop 回 target size
            if self.rand_scale:
                scale = rng.uniform(1.0, 1.2)  # 可按 BPNet 调
                new_h = int(round(self.target_h * scale))
                new_w = int(round(self.target_w * scale))
                # 保证仍为 16 的倍数（可选）
                new_h = (new_h // 16) * 16
                new_w = (new_w // 16) * 16
                new_h = max(new_h, self.target_h)
                new_w = max(new_w, self.target_w)

                # resize
                rgb = TF.resize(rgb, size=[new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
                dep = np.array(Image.fromarray(dep).resize((new_w, new_h), resample=Image.NEAREST), dtype=np.float32)
                dep_sp = np.array(Image.fromarray(dep_sp).resize((new_w, new_h), resample=Image.NEAREST), dtype=np.float32)

                # update K for resize
                sx = new_w / self.target_w
                sy = new_h / self.target_h
                K = update_K_for_resize(K, sx=sx, sy=sy)

                # center crop back to target
                left2 = (new_w - self.target_w) // 2
                top2 = (new_h - self.target_h) // 2

                rgb = TF.crop(rgb, top=top2, left=left2, height=self.target_h, width=self.target_w)
                dep = dep[top2:top2 + self.target_h, left2:left2 + self.target_w]
                dep_sp = dep_sp[top2:top2 + self.target_h, left2:left2 + self.target_w]

                # update K for crop2
                K = update_K_for_crop(K, left=left2, top=top2)

        # --------------------------
        # 5) 转 tensor + 升维 dep_sp/dep 为 (1,H,W)
        # --------------------------
        rgb_t = self.norm(self.to_tensor(rgb))  # (3,H,W)

        dep_sp_t = torch.from_numpy(dep_sp).unsqueeze(0)  # (1,H,W)
        dep_t = torch.from_numpy(dep).unsqueeze(0)        # (1,H,W)

        # 深度缩放（mul_factor）
        dep_sp_t = dep_sp_t * self.mul_factor
        dep_t = dep_t * self.mul_factor

        return rgb_t, dep_sp_t, K, dep_t
