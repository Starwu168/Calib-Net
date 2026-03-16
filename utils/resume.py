# utils/resume.py
from __future__ import annotations
import re
from pathlib import Path

import torch
import torch.distributed as dist

from utils.ddp import ddp_is_initialized, is_main_process, barrier


_EPOCH_RE = re.compile(r"epoch_(\d+)\.pth$")


def _find_latest_epoch_ckpt(out_dir: Path) -> Path | None:
    if not out_dir.exists():
        return None
    ckpts = list(out_dir.glob("epoch_*.pth"))
    if not ckpts:
        return None
    best_p = None
    best_e = -1
    for p in ckpts:
        m = _EPOCH_RE.search(p.name)
        if not m:
            continue
        e = int(m.group(1))
        if e > best_e:
            best_e = e
            best_p = p
    return best_p


def _broadcast_str(s: str, device: torch.device) -> str:
    """Broadcast a python string from rank0 to all ranks (DDP safe)."""
    if not ddp_is_initialized():
        return s

    if is_main_process():
        b = s.encode("utf-8")
        n = torch.tensor([len(b)], device=device, dtype=torch.int64)
    else:
        n = torch.tensor([0], device=device, dtype=torch.int64)

    dist.broadcast(n, src=0)

    buf = torch.empty((int(n.item()),), device=device, dtype=torch.uint8)
    if is_main_process():
        buf[:] = torch.tensor(list(b), device=device, dtype=torch.uint8)

    dist.broadcast(buf, src=0)
    return bytes(buf.tolist()).decode("utf-8")


def auto_resume(
    out_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    prefer: str = "latest",  # "latest" or "best"
    ckpt_path: str | Path | None = None,
) -> tuple[int, float | None, str | None]:
    """
    Load model/optim/scaler from checkpoint in out_dir.

    Returns:
      start_epoch: next epoch index (ckpt_epoch + 1), or 1 if none found
      best_rmse: float|None (from ckpt["extra"]["best_rmse"] if exists)
      ckpt_path: str|None
    """
    out_dir = Path(out_dir)

    # rank0 selects path
    if is_main_process():
        selected_ckpt: Path | None = None
        if ckpt_path is not None:
            p = Path(ckpt_path)
            selected_ckpt = p if p.exists() else None
        elif prefer == "best":
            p = out_dir / "best.pth"
            selected_ckpt = p if p.exists() else None
        else:
            selected_ckpt = _find_latest_epoch_ckpt(out_dir)
            if selected_ckpt is None:
                p = out_dir / "best.pth"
                selected_ckpt = p if p.exists() else None

        ckpt_str = str(selected_ckpt) if selected_ckpt is not None else ""
    else:
        ckpt_str = ""

    # broadcast to all ranks
    ckpt_str = _broadcast_str(ckpt_str, device=device)
    barrier()

    if ckpt_str == "":
        return 1, None, None

    ckpt = torch.load(ckpt_str, map_location="cpu")

    # --- load model ---
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError:
        # if model is DDP wrapper but load called on DDP; fallback
        if hasattr(model, "module"):
            model.module.load_state_dict(ckpt["model"], strict=True)
        else:
            raise

    # --- load optimizer/scaler (NOTE: key is "optim" in your save_checkpoint) ---
    if optimizer is not None and "optim" in ckpt and ckpt["optim"] is not None:
        optimizer.load_state_dict(ckpt["optim"])

    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])

    epoch = int(ckpt.get("epoch", 0))
    extra = ckpt.get("extra", {}) or {}
    best_rmse = extra.get("best_rmse", None)
    best_rmse = float(best_rmse) if best_rmse is not None else None

    return epoch + 1, best_rmse, ckpt_str
