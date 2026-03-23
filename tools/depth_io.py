from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
from PIL import Image


DEPTH_SCALE = 256.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_depth_png_16bit(path: str, depth_m: np.ndarray) -> None:
    """
    Save depth map in meters to uint16 PNG with scale 256.
    0 means invalid.
    """
    if depth_m.ndim != 2:
        raise ValueError(f"depth_m must be HxW, got {depth_m.shape}")

    depth = np.asarray(depth_m, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0.0

    depth_u16 = np.clip(np.round(depth * DEPTH_SCALE), 0, 65535).astype(np.uint16)
    Image.fromarray(depth_u16).save(path)


def load_depth_png_16bit(path: str) -> np.ndarray:
    d = np.array(Image.open(path), dtype=np.float32)
    return d / DEPTH_SCALE


def save_mask_png(path: str, mask: np.ndarray) -> None:
    """
    Save bool/0-1 mask as uint8 PNG.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    Image.fromarray(m).save(path)


def load_mask_png(path: str) -> np.ndarray:
    m = np.array(Image.open(path), dtype=np.uint8)
    return (m > 0).astype(np.uint8)


def save_meta_npz(path: str, meta_dict: Dict[str, Any]) -> None:
    np.savez_compressed(path, **meta_dict)


def load_meta_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in data.files:
        v = data[k]
        if isinstance(v, np.ndarray) and v.shape == ():
            try:
                out[k] = v.item()
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out