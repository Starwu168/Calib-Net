# main.py
from __future__ import annotations
import os
import time
import argparse
from pathlib import Path

import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.config import load_yaml
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
from utils.meters import AvgMeter
from datasets.builder import build_dataset
from models.calib_net import CalibOnlyNet
from train.criteria import compute_losses
from utils.ddp import setup_ddp, ddp_cleanup


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch):
    model.train()
    meter = AvgMeter()
    start_time = time.time()

    log_every = cfg["train"]["log_every"]
    amp = cfg["exp"]["amp"]

    rank = dist.get_rank() if dist.is_initialized() else 0
    pbar = tqdm(
        loader,
        total=len(loader),
        dynamic_ncols=True,
        desc=f"train e{epoch}",
        disable=(rank != 0)
    )

    for step, batch in enumerate(pbar):
        rgb, dep_sp, Kcam, dep = batch
        rgb = rgb.to(device, non_blocking=True)
        dep_sp = dep_sp.to(device, non_blocking=True)
        dep = dep.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            s_pred_list = model(rgb, dep_sp)
            loss, parts = compute_losses(s_pred_list, dep_sp, dep, cfg["loss"])

        scaler.scale(loss).backward()

        if cfg["train"]["grad_clip_norm"] is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])

        scaler.step(optimizer)
        scaler.update()

        elapsed = time.time() - start_time

        meter.update(loss=float(loss.detach().cpu()), n=rgb.size(0), parts=parts)

        # tqdm显示（只在rank0）
        pbar.set_postfix({"time": f"{elapsed/60:.1f}m"})

        # 原有print保留，但只让rank0打印（否则多进程疯狂刷屏很慢）
        if rank == 0 and (step + 1) % log_every == 0:
            print(
                f"[train] step={step+1}/{len(loader)} "
                f"loss={meter.avg('loss'):.6f} "
                f"sparse={meter.avg('loss_sparse'):.6f} "
                f"consis={meter.avg('loss_consis'):.6f} "
                f"energy={meter.avg('loss_energy'):.6f}"
            )

    return meter


@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    meter = AvgMeter()
    rank = dist.get_rank() if dist.is_initialized() else 0

    for batch in loader:
        rgb, dep_sp, Kcam, dep = batch
        rgb = rgb.to(device, non_blocking=True)
        dep_sp = dep_sp.to(device, non_blocking=True)
        dep = dep.to(device, non_blocking=True)

        with autocast(enabled=cfg["exp"]["amp"]):
            s_pred_list = model(rgb, dep_sp)
            loss, parts = compute_losses(s_pred_list, dep_sp, dep, cfg["loss"])

        meter.update(loss=float(loss.detach().cpu()), n=rgb.size(0), parts=parts)

    if rank == 0:
        print(
            f"[val] loss={meter.avg('loss'):.6f} "
            f"sparse={meter.avg('loss_sparse'):.6f} "
            f"consis={meter.avg('loss_consis'):.6f} "
            f"energy={meter.avg('loss_energy'):.6f}"
        )
    return meter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to yaml config")
    args = ap.parse_args()

    # ---- DDP init (torchrun会注入LOCAL_RANK等) ----
    local_rank = setup_ddp()  # 必须返回local_rank，并在内部 torch.cuda.set_device(local_rank)

    cfg = load_yaml(args.config)

    # ---- seed：每个rank不同更安全 ----
    rank = dist.get_rank() if dist.is_initialized() else 0
    set_seed(cfg["exp"]["seed"] + rank)

    # ---- speed flags ----
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # ✅ 关键：device必须绑定到local_rank
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # ---- out dir: 只在rank0创建/打印 ----
    out_dir = Path(cfg["exp"]["out_dir"]) / cfg["exp"]["name"]
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] out_dir={out_dir}")

    # ---- dataset ----
    train_kwargs = dict(cfg["data"]["dataset_kwargs"])
    train_kwargs["mode"] = "train"
    train_ds = build_dataset(cfg["data"]["dataset_class"], train_kwargs)

    val_kwargs = dict(cfg["data"]["dataset_kwargs"])
    val_kwargs["mode"] = "val"
    val_ds = build_dataset(cfg["data"]["dataset_class"], val_kwargs)

    # ✅ DDP: sampler
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["train_batch_size"],
        sampler=train_sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=(cfg["data"]["num_workers"] > 0),
        prefetch_factor=4 if cfg["data"]["num_workers"] > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["val_batch_size"],
        sampler=val_sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=(cfg["data"]["num_workers"] > 0),
        prefetch_factor=4 if cfg["data"]["num_workers"] > 0 else None,
    )

    # ---- model ----
    model = CalibOnlyNet(cfg["model"]).to(device)

    # ✅ DDP wrapper：每个进程只用自己的GPU
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    # ---- optim ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    scaler = GradScaler(enabled=cfg["exp"]["amp"])

    best_val = 1e18
    epochs = cfg["train"]["epochs"]

    for epoch in range(1, epochs + 1):
        # ✅ DDP: 每个epoch需要set_epoch保证shuffle正确
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n===== epoch {epoch}/{epochs} =====")

        train_meter = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)

        if epoch % cfg["train"]["val_every"] == 0:
            val_meter = validate(model, val_loader, device, cfg)
            val_loss = val_meter.avg("loss")

            if rank == 0:
                is_best = val_loss < best_val
                best_val = min(best_val, val_loss)

                if epoch % cfg["train"]["save_every"] == 0 or is_best:
                    save_checkpoint(
                        out_dir=out_dir,
                        epoch=epoch,
                        # ✅ 保存真实模型参数
                        model=model.module,
                        optimizer=optimizer,
                        scaler=scaler,
                        best=is_best,
                        extra={"val_loss": val_loss}
                    )

    if rank == 0:
        print("[done]")

    ddp_cleanup()


if __name__ == "__main__":
    main()
