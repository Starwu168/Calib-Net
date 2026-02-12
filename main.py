# main.py
from __future__ import annotations
import os
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from utils.config import load_yaml, get
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
from utils.meters import AvgMeter
from datasets.builder import build_dataset, build_loader
from models.calib_net import CalibOnlyNet


def compute_losses(s_pred_list, dep_sp, dep_gt, cfg_loss: dict):
    """
    s_pred_list: list of S_l' (B,1,Hl,Wl)
    dep_sp:      (B,1,H,W)
    dep_gt:      (B,1,H,W)
    """
    ms_w = cfg_loss["ms_weights"]
    assert len(ms_w) == len(s_pred_list)

    B, _, H, W = dep_sp.shape
    mask = (dep_sp > 0).float()  # 监督只在 radar 有效处
    # 稀疏监督 target：GT
    target = dep_gt

    loss_sparse = 0.0
    loss_consis = 0.0
    loss_energy = 0.0

    for l, s_l in enumerate(s_pred_list):
        # upsample 回原图做监督
        s_up = F.interpolate(s_l, size=(H, W), mode="bilinear", align_corners=False)

        # 1) 稀疏监督：S'在mask处逼近GT
        # smooth L1 更稳
        ls = F.smooth_l1_loss(s_up * mask, target * mask, reduction="sum") / (mask.sum() + 1e-6)

        # 2) 保守一致性：不要离原 dep_sp 太远（同样只在 mask）
        lc = F.smooth_l1_loss(s_up * mask, dep_sp * mask, reduction="sum") / (mask.sum() + 1e-6)

        # 3) 能量约束：鼓励输出在mask区域有一定能量，避免全0塌缩
        # 让 mean(s_up on mask) 接近 dep_sp 的均值（弱约束）
        mean_pred = (s_up * mask).sum() / (mask.sum() + 1e-6)
        mean_src  = (dep_sp * mask).sum() / (mask.sum() + 1e-6)
        le = (mean_pred - mean_src).abs()

        loss_sparse += ms_w[l] * ls
        loss_consis += ms_w[l] * lc
        loss_energy += ms_w[l] * le

    w_sparse = cfg_loss["w_sparse"]
    w_consis = cfg_loss["w_consistency"]
    w_energy = cfg_loss["w_energy"]

    total = w_sparse * loss_sparse + w_consis * loss_consis + w_energy * loss_energy
    return total, {
        "loss_sparse": float(loss_sparse.detach().cpu()),
        "loss_consis": float(loss_consis.detach().cpu()),
        "loss_energy": float(loss_energy.detach().cpu()),
    }


def train_one_epoch(model, loader, optimizer, scaler, device, cfg):
    model.train()
    meter = AvgMeter()

    log_every = cfg["train"]["log_every"]
    amp = cfg["exp"]["amp"]

    for step, batch in enumerate(loader):
        rgb, dep_sp, Kcam, dep = batch
        rgb = rgb.to(device, non_blocking=True)
        dep_sp = dep_sp.to(device, non_blocking=True)
        dep = dep.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            s_pred_list = model(rgb, dep_sp)
            loss, parts = compute_losses(s_pred_list, dep_sp, dep, cfg["loss"])

        scaler.scale(loss).backward()
        # 梯度裁剪
        if cfg["train"]["grad_clip_norm"] is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])

        scaler.step(optimizer)
        scaler.update()

        meter.update(loss=float(loss.detach().cpu()), n=rgb.size(0), parts=parts)

        if (step + 1) % log_every == 0:
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

    for batch in loader:
        rgb, dep_sp, Kcam, dep = batch
        rgb = rgb.to(device, non_blocking=True)
        dep_sp = dep_sp.to(device, non_blocking=True)
        dep = dep.to(device, non_blocking=True)

        s_pred_list = model(rgb, dep_sp)
        loss, parts = compute_losses(s_pred_list, dep_sp, dep, cfg["loss"])
        meter.update(loss=float(loss.detach().cpu()), n=rgb.size(0), parts=parts)

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

    cfg = load_yaml(args.config)

    set_seed(cfg["exp"]["seed"])
    device = torch.device(cfg["exp"]["device"] if torch.cuda.is_available() else "cpu")

    # output dir
    out_dir = Path(cfg["exp"]["out_dir"]) / cfg["exp"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] out_dir={out_dir}")

    # dataset
    train_kwargs = dict(cfg["data"]["dataset_kwargs"])
    train_kwargs["mode"] = "train"
    train_ds = build_dataset(cfg["data"]["dataset_class"], train_kwargs)

    val_kwargs = dict(cfg["data"]["dataset_kwargs"])
    val_kwargs["mode"] = "val"
    val_ds = build_dataset(cfg["data"]["dataset_class"], val_kwargs)

    train_loader = build_loader(
        train_ds,
        batch_size=cfg["data"]["train_batch_size"],
        num_workers=cfg["data"]["num_workers"],
        shuffle=True
    )
    val_loader = build_loader(
        val_ds,
        batch_size=cfg["data"]["val_batch_size"],
        num_workers=cfg["data"]["num_workers"],
        shuffle=False
    )

    # model
    model = CalibOnlyNet(cfg["model"]).to(device)

    # optim
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    scaler = GradScaler(enabled=cfg["exp"]["amp"])

    best_val = 1e18
    epochs = cfg["train"]["epochs"]

    for epoch in range(1, epochs + 1):
        print(f"\n===== epoch {epoch}/{epochs} =====")
        train_meter = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg)

        if epoch % cfg["train"]["val_every"] == 0:
            val_meter = validate(model, val_loader, device, cfg)
            val_loss = val_meter.avg("loss")

            is_best = val_loss < best_val
            best_val = min(best_val, val_loss)

            if epoch % cfg["train"]["save_every"] == 0 or is_best:
                save_checkpoint(
                    out_dir=out_dir,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    best=is_best,
                    extra={"val_loss": val_loss}
                )

    print("[done]")


if __name__ == "__main__":
    main()
