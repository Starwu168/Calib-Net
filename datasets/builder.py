# datasets/builder.py
from __future__ import annotations
import importlib
from typing import Any, Dict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.ddp import ddp_is_initialized


def import_from_path(class_path: str):
    module_path, cls_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)

def build_dataset(class_path: str, kwargs: Dict[str, Any]):
    cls = import_from_path(class_path)
    return cls(**kwargs)

def build_loader(ds, batch_size: int, num_workers: int, shuffle: bool):
    sampler = None
    if ddp_is_initialized():
        sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=shuffle)
        shuffle = False  # DDP 下 shuffle 交给 sampler

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
