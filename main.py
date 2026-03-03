from __future__ import annotations
from utils.resume import auto_resume
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
from models.calib_pmp_net import CalibPMPNet
from train.criteria_total import TotalCriterion
from utils.metrics_dc import DCMetrics, fmt_metrics
from utils.ddp import setup_ddp, ddp_cleanup


def _final_pred(pmp_out_list):
    return pmp_out_list[-1] if isinstance(pmp_out_list, (list, tuple)) else pmp_out_list


def _unpack_batch(batch):
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Unexpected batch type: {type(batch)}")
    if len(batch) < 4:
        raise ValueError(f"Batch must contain at least 4 items, got {len(batch)}")
    rgb, dep_sp, Kcam, dep = batch[:4]
    dep_sparse = batch[4] if len(batch) >= 5 else None
    return rgb, dep_sp, Kcam, dep, dep_sparse


def _eval_target_and_mask(dep_dense, dep_sparse, cfg_loss: dict):
    use_sparse_eval = bool(cfg_loss.get("metrics_use_sparse_gt", True))
    use_sparse_mask = bool(cfg_loss.get("use_sparse_gt_mask", True))

    target = dep_sparse if (dep_sparse is not None and use_sparse_eval) else dep_dense
    valid_mask = None
    if dep_sparse is not None and use_sparse_mask:
        valid_mask = dep_sparse > float(cfg_loss.get("t_valid", 1e-3))
    return target, valid_mask


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch, criterion: TotalCriterion):
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
        disable=(rank != 0),
    )

    met = DCMetrics(t_valid=cfg["loss"].get("t_valid", 1e-3))

    for step, batch in enumerate(pbar):
        rgb, dep_sp, Kcam, dep, dep_sparse = _unpack_batch(batch)
        rgb = rgb.to(device, non_blocking=True)
        dep_sp = dep_sp.to(device, non_blocking=True)
        Kcam = Kcam.to(device, non_blocking=True)
        dep = dep.to(device, non_blocking=True)
        dep_sparse = dep_sparse.to(device, non_blocking=True) if dep_sparse is not None else None

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            pmp_out_list, s_pred_list, _sprime = model(rgb, dep_sp, Kcam)
            loss, parts = criterion(
                pmp_out_list=pmp_out_list,
                s_pred_list=s_pred_list,
                dep_sp=dep_sp,
                dep_gt=dep,
                dep_sparse_gt=dep_sparse,
                cfg_loss_calib=cfg["loss"]["calib"],
            )

        scaler.scale(loss).backward()

        if cfg["train"]["grad_clip_norm"] is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])

        scaler.step(optimizer)
        scaler.update()

        elapsed = time.time() - start_time
        meter.update(loss=float(loss.detach().cpu()), n=rgb.size(0), parts=parts)

        pred = _final_pred(pmp_out_list)
        eval_tgt, eval_mask = _eval_target_and_mask(dep, dep_sparse, cfg["loss"])
        met.update(
            pred,
            eval_tgt,
            valid_mask=eval_mask,
            depth_min=cfg["loss"].get("depth_min", None),
            depth_max=cfg["loss"].get("depth_max", None),
        )

        pbar.set_postfix({"time": f"{elapsed/60:.1f}m"})

        if rank == 0 and (step + 1) % log_every == 0:
            print(
                f"[train] step={step+1}/{len(loader)} "
                f"loss={meter.avg('loss'):.6f} "
                f"calib={meter.avg('loss_calib'):.6f} "
                f"pmp={meter.avg('loss_pmp'):.6f} "
                f"sparse={meter.avg('loss_sparse'):.6f} "
                f"consis={meter.avg('loss_consis'):.6f} "
                f"energy={meter.avg('loss_energy'):.6f} "
                f"outside={meter.avg('loss_outside'):.6f}"
            )

    met.all_reduce_()
    return meter, met.compute()


@torch.no_grad()
def validate(model, loader, device, cfg, criterion: TotalCriterion):
    model.eval()
    meter = AvgMeter()
    rank = dist.get_rank() if dist.is_initialized() else 0
    met = DCMetrics(t_valid=cfg["loss"].get("t_valid", 1e-3))

    for batch in loader:
        rgb, dep_sp, Kcam, dep, dep_sparse = _unpack_batch(batch)
        rgb = rgb.to(device, non_blocking=True)
        dep_sp = dep_sp.to(device, non_blocking=True)
        Kcam = Kcam.to(device, non_blocking=True)
        dep = dep.to(device, non_blocking=True)
        dep_sparse = dep_sparse.to(device, non_blocking=True) if dep_sparse is not None else None

        with autocast(enabled=cfg["exp"]["amp"]):
            pmp_out_list, s_pred_list, _sprime = model(rgb, dep_sp, Kcam)
            loss, parts = criterion(
                pmp_out_list=pmp_out_list,
                s_pred_list=s_pred_list,
                dep_sp=dep_sp,
                dep_gt=dep,
                dep_sparse_gt=dep_sparse,
                cfg_loss_calib=cfg["loss"]["calib"],
            )

        meter.update(loss=float(loss.detach().cpu()), n=rgb.size(0), parts=parts)

        pred = _final_pred(pmp_out_list)
        eval_tgt, eval_mask = _eval_target_and_mask(dep, dep_sparse, cfg["loss"])
        met.update(
            pred,
            eval_tgt,
            valid_mask=eval_mask,
            depth_min=cfg["loss"].get("depth_min", None),
            depth_max=cfg["loss"].get("depth_max", None),
        )

    met.all_reduce_()
    m = met.compute()

    if rank == 0:
        print(
            f"[val] loss={meter.avg('loss'):.6f} "
            f"calib={meter.avg('loss_calib'):.6f} "
            f"pmp={meter.avg('loss_pmp'):.6f}"
        )
        print(f"[val] metrics: {fmt_metrics(m)}")

    return meter, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to yaml config")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="auto resume from runs/<exp.name>/latest epoch ckpt",
    )
    ap.add_argument(
        "--resume_prefer",
        type=str,
        default="latest",
        choices=["latest", "best"],
        help="resume from latest epoch_*.pth or best.pth",
    )
    args = ap.parse_args()

    local_rank = setup_ddp()
    cfg = load_yaml(args.config)

    rank = dist.get_rank() if dist.is_initialized() else 0
    set_seed(cfg["exp"]["seed"] + rank)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    out_dir = Path(cfg["exp"]["out_dir"]) / cfg["exp"]["name"]
    if rank == 0:
        cwd = Path.cwd().resolve()
        cfg_path = Path(args.config).resolve()
        out_dir_abs = out_dir.resolve()
        print(f"[info] cwd={cwd}")
        print(f"[info] config={cfg_path}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] out_dir={out_dir} (abs={out_dir_abs})")

    train_kwargs = dict(cfg["data"]["dataset_kwargs"])
    train_kwargs["mode"] = "train"
    val_kwargs = dict(cfg["data"]["dataset_kwargs"])
    val_kwargs["mode"] = "val"
    train_ds = build_dataset(cfg["data"]["dataset_class"], train_kwargs)
    val_ds = build_dataset(cfg["data"]["dataset_class"], val_kwargs)

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

    model = CalibPMPNet(cfg["model"]).to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    criterion = TotalCriterion(cfg["loss"]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = GradScaler(enabled=cfg["exp"]["amp"])

    epochs = cfg["train"]["epochs"]
    start_epoch = 1
    best_rmse = 1e18

    if args.resume:
        se, loaded_best, ckpt_path = auto_resume(
            out_dir=out_dir,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            prefer=args.resume_prefer,
        )
        start_epoch = se
        if loaded_best is not None:
            best_rmse = loaded_best
        if rank == 0:
            if ckpt_path is None:
                print("[resume] no checkpoint found, start from scratch.")
            else:
                print(f"[resume] loaded: {ckpt_path}")
                print(f"[resume] start_epoch={start_epoch}, best_rmse={best_rmse}")

    for epoch in range(start_epoch, epochs + 1):
        train_sampler.set_epoch(epoch)

        base_lr = cfg["train"]["lr"]
        decay_steps = (epoch - 1) // 10
        new_lr = max(base_lr - decay_steps * 2e-5, 1e-7)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

        if rank == 0:
            print(f"[lr] epoch={epoch}  lr={new_lr:.8f}")
            print(f"\n===== epoch {epoch}/{epochs} =====")

        train_meter, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg, epoch, criterion
        )
        if rank == 0:
            print(f"[train] metrics: {fmt_metrics(train_metrics)}")

        if epoch % cfg["train"]["val_every"] == 0:
            val_meter, val_metrics = validate(model, val_loader, device, cfg, criterion)

            if rank == 0:
                rmse = float(val_metrics["RMSE"])
                is_best = rmse < best_rmse
                best_rmse = min(best_rmse, rmse)

                if epoch % cfg["train"]["save_every"] == 0 or is_best:
                    save_checkpoint(
                        out_dir=out_dir,
                        epoch=epoch,
                        model=model.module,
                        optimizer=optimizer,
                        scaler=scaler,
                        best=is_best,
                        extra={"best_rmse": best_rmse, "val_rmse": rmse},
                    )

    if rank == 0:
        print("[done]")

    ddp_cleanup()


if __name__ == "__main__":
    main()
