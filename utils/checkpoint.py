# utils/checkpoint.py
from __future__ import annotations
from pathlib import Path
import torch

def save_checkpoint(out_dir: Path, epoch: int, model, optimizer, scaler, best: bool, extra=None):
    out_dir = Path(out_dir)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "extra": extra or {},
    }
    path = out_dir / f"epoch_{epoch:03d}.pth"
    torch.save(ckpt, path)
    if best:
        torch.save(ckpt, out_dir / "best.pth")
    print(f"[ckpt] saved: {path} (best={best})")
