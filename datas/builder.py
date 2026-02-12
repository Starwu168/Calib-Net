# datasets/builder.py
from __future__ import annotations
import importlib
from typing import Any, Dict
from torch.utils.data import DataLoader

def import_from_path(class_path: str):
    """
    class_path: "pkg.subpkg.module.ClassName"
    """
    module_path, cls_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)

def build_dataset(class_path: str, kwargs: Dict[str, Any]):
    cls = import_from_path(class_path)
    return cls(**kwargs)

def build_loader(ds, batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )
