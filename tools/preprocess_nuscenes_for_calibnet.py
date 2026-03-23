from __future__ import annotations

import argparse
import os
import traceback
from typing import Dict, List

import numpy as np
from PIL import Image

from tools.depth_io import ensure_dir, save_depth_png_16bit, save_mask_png, save_meta_npz
from tools.nuscenes_adapter import (
    collect_split_records,
    get_dynamic_boxes,
    make_nuscenes,
    make_sample_id,
)
from tools.nuscenes_geometry import (
    interpolate_depth_map,
    merge_lidar_sweeps_to_depth_map,
    project_lidar_to_depth_map,
    project_radar_to_uvd,
)


def str2bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "on")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--camera_name", type=str, default="CAM_FRONT")
    parser.add_argument("--radar_name", type=str, default="RADAR_FRONT")
    parser.add_argument("--lidar_name", type=str, default="LIDAR_TOP")
    parser.add_argument("--n_backward", type=int, default=9)
    parser.add_argument("--n_forward", type=int, default=9)
    parser.add_argument("--min_depth", type=float, default=1.0)
    parser.add_argument("--max_depth", type=float, default=80.0)
    parser.add_argument("--filter_dynamic", type=str, default="true")
    parser.add_argument("--verbose", type=str, default="true")
    return parser.parse_args()


def prepare_dirs(output_root: str):
    base = os.path.join(output_root, "data")
    dirs = {
        "base": base,
        "image": os.path.join(base, "image"),
        "radar": os.path.join(base, "radar"),
        "gt": os.path.join(base, "gt"),
        "gt_interp": os.path.join(base, "gt_interp"),
        "meta": os.path.join(base, "meta"),
        "mask_raw": os.path.join(base, "valid_mask_raw"),
        "mask_interp": os.path.join(base, "valid_mask_interp"),
        "logs": os.path.join(output_root, "logs"),
    }
    for p in dirs.values():
        ensure_dir(p)
    return dirs


def save_split_file(path: str, names: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for n in names:
            f.write(f"{n}\n")


def process_one_record(
    nusc,
    record: Dict,
    dirs: Dict[str, str],
    cfg: Dict,
):
    sample_id = make_sample_id(record["scene_name"], record["sample_token"])

    img_out = os.path.join(dirs["image"], f"{sample_id}.png")
    radar_out = os.path.join(dirs["radar"], f"{sample_id}.npy")
    gt_out = os.path.join(dirs["gt"], f"{sample_id}.png")
    gt_interp_out = os.path.join(dirs["gt_interp"], f"{sample_id}.png")
    meta_out = os.path.join(dirs["meta"], f"{sample_id}.npz")
    mask_raw_out = os.path.join(dirs["mask_raw"], f"{sample_id}.png")
    mask_interp_out = os.path.join(dirs["mask_interp"], f"{sample_id}.png")

    # Save image
    img_path = os.path.join(cfg["nuscenes_root"], record["camera_filename"])
    rgb = Image.open(img_path).convert("RGB")
    W, H = rgb.size
    assert W == record["camera_width"] and H == record["camera_height"]
    rgb.save(img_out)

    # Radar projection
    radar_uvd = project_radar_to_uvd(
        nusc=nusc,
        radar_token=record["radar_token"],
        camera_token=record["camera_token"],
        min_depth=cfg["min_depth"],
        max_depth=cfg["max_depth"],
    )
    np.save(radar_out, radar_uvd.astype(np.float32))

    # Single-frame lidar GT
    gt_raw, valid_raw = project_lidar_to_depth_map(
        nusc=nusc,
        lidar_token=record["lidar_token"],
        camera_token=record["camera_token"],
        min_depth=cfg["min_depth"],
        max_depth=cfg["max_depth"],
    )

    # Multi-frame merged lidar GT
    def dynamic_boxes_provider(sample_token: str):
        sample = nusc.get("sample", sample_token)
        cam_token = sample["data"][cfg["camera_name"]]
        return get_dynamic_boxes(nusc, cam_token)

    gt_merged_raw, valid_merged_raw = merge_lidar_sweeps_to_depth_map(
        nusc=nusc,
        ref_sample_token=record["sample_token"],
        ref_camera_token=record["camera_token"],
        lidar_name=cfg["lidar_name"],
        n_backward=cfg["n_backward"],
        n_forward=cfg["n_forward"],
        min_depth=cfg["min_depth"],
        max_depth=cfg["max_depth"],
        filter_dynamic=cfg["filter_dynamic"],
        dynamic_boxes_provider=dynamic_boxes_provider,
    )

    gt_interp, valid_interp = interpolate_depth_map(
        gt_merged_raw,
        max_fill_distance=12.0,
    )

    # For safety, keep raw values where merged/interp are missing but single frame has support.
    fill_from_single = (gt_interp <= 0) & (gt_raw > 0)
    gt_interp[fill_from_single] = gt_raw[fill_from_single]
    valid_interp = (gt_interp > 0).astype(np.uint8)

    save_depth_png_16bit(gt_out, gt_raw)
    save_depth_png_16bit(gt_interp_out, gt_interp)
    save_mask_png(mask_raw_out, valid_raw)
    save_mask_png(mask_interp_out, valid_interp)

    meta = {
        "K": record["camera_intrinsics"].astype(np.float32),
        "width": np.int32(W),
        "height": np.int32(H),
        "sample_token": np.array(record["sample_token"]),
        "scene_token": np.array(record["scene_token"]),
        "scene_name": np.array(record["scene_name"]),
        "camera_name": np.array(record["camera_name"]),
        "radar_name": np.array(record["radar_name"]),
        "lidar_name": np.array(record["lidar_name"]),
        "valid_raw_count": np.int32(valid_raw.sum()),
        "valid_interp_count": np.int32(valid_interp.sum()),
        "valid_merged_raw_count": np.int32(valid_merged_raw.sum()),
    }
    save_meta_npz(meta_out, meta)

    stats = {
        "sample_id": sample_id,
        "radar_points": int(radar_uvd.shape[0]),
        "gt_valid": int(valid_raw.sum()),
        "gt_interp_valid": int(valid_interp.sum()),
    }
    return stats


def main():
    args = parse_args()
    cfg = {
        "nuscenes_root": args.nuscenes_root,
        "output_root": args.output_root,
        "version": args.version,
        "camera_name": args.camera_name,
        "radar_name": args.radar_name,
        "lidar_name": args.lidar_name,
        "n_backward": args.n_backward,
        "n_forward": args.n_forward,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "filter_dynamic": str2bool(args.filter_dynamic),
        "verbose": str2bool(args.verbose),
    }

    dirs = prepare_dirs(cfg["output_root"])
    nusc = make_nuscenes(cfg["version"], cfg["nuscenes_root"], verbose=cfg["verbose"])

    split_names = ["train", "val"]
    if "test" in cfg["version"]:
        split_names = ["test"]

    split_to_names = {}
    all_failures = []

    for split in split_names:
        print(f"[preprocess] collecting split={split}")
        records = collect_split_records(
            nusc=nusc,
            split=split,
            camera_name=cfg["camera_name"],
            radar_name=cfg["radar_name"],
            lidar_name=cfg["lidar_name"],
        )

        split_sample_ids = []
        radar_counts = []
        gt_counts = []
        gt_interp_counts = []

        print(f"[preprocess] split={split}, records={len(records)}")
        for i, rec in enumerate(records):
            try:
                stats = process_one_record(nusc, rec, dirs, cfg)
                split_sample_ids.append(stats["sample_id"])
                radar_counts.append(stats["radar_points"])
                gt_counts.append(stats["gt_valid"])
                gt_interp_counts.append(stats["gt_interp_valid"])

                if (i + 1) % 100 == 0:
                    print(
                        f"[{split}] {i+1}/{len(records)} "
                        f"radar_mean={np.mean(radar_counts):.1f} "
                        f"gt_mean={np.mean(gt_counts):.1f} "
                        f"gt_interp_mean={np.mean(gt_interp_counts):.1f}"
                    )
            except Exception as e:
                sample_id = make_sample_id(rec["scene_name"], rec["sample_token"])
                tb = traceback.format_exc()
                all_failures.append((split, sample_id, str(e), tb))
                print(f"[ERROR] split={split}, sample_id={sample_id}, err={e}")

        split_to_names[split] = split_sample_ids
        save_split_file(os.path.join(dirs["base"], f"{split}.txt"), split_sample_ids)

        if len(split_sample_ids) > 0:
            print(
                f"[done] split={split}, saved={len(split_sample_ids)}, "
                f"radar_mean={np.mean(radar_counts):.2f}, "
                f"gt_mean={np.mean(gt_counts):.2f}, "
                f"gt_interp_mean={np.mean(gt_interp_counts):.2f}"
            )
        else:
            print(f"[done] split={split}, saved=0")

    fail_log = os.path.join(dirs["logs"], "preprocess_failures.txt")
    with open(fail_log, "w", encoding="utf-8") as f:
        for split, sample_id, err, tb in all_failures:
            f.write(f"[{split}] {sample_id}\n")
            f.write(f"{err}\n")
            f.write(tb)
            f.write("\n\n")

    print(f"[final] failure_count={len(all_failures)}")
    print(f"[final] failure_log={fail_log}")


if __name__ == "__main__":
    main()