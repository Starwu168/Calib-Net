from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


DYNAMIC_PREFIXES = ("vehicle", "human", "cycle")


def make_nuscenes(version: str, dataroot: str, verbose: bool = True) -> NuScenes:
    return NuScenes(version=version, dataroot=dataroot, verbose=verbose)


def get_scene_names_for_split(version: str, split: str) -> List[str]:
    """
    nuScenes official split helper.
    """
    split_map = create_splits_scenes()

    if split == "train":
        return split_map["train"]
    if split == "val":
        return split_map["val"]
    if split == "test":
        if "test" in split_map:
            return split_map["test"]
        return []

    raise ValueError(f"unsupported split: {split}")


def scene_name_to_token_map(nusc: NuScenes) -> Dict[str, str]:
    return {scene["name"]: scene["token"] for scene in nusc.scene}


def collect_split_records(
    nusc: NuScenes,
    split: str,
    camera_name: str = "CAM_FRONT",
    radar_name: str = "RADAR_FRONT",
    lidar_name: str = "LIDAR_TOP",
) -> List[Dict]:
    """
    Returns sample-level records. One sample corresponds to one reference camera frame.
    """
    scene_names = set(get_scene_names_for_split(nusc.version, split))
    records: List[Dict] = []

    for scene in nusc.scene:
        if scene["name"] not in scene_names:
            continue

        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            data = sample["data"]

            if camera_name not in data or radar_name not in data or lidar_name not in data:
                sample_token = sample["next"]
                continue

            cam_token = data[camera_name]
            radar_token = data[radar_name]
            lidar_token = data[lidar_name]

            cam_sd = nusc.get("sample_data", cam_token)
            radar_sd = nusc.get("sample_data", radar_token)
            lidar_sd = nusc.get("sample_data", lidar_token)

            cs_cam = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])

            record = {
                "split": split,
                "scene_name": scene["name"],
                "scene_token": scene["token"],
                "sample_token": sample["token"],
                "timestamp": sample["timestamp"],
                "camera_name": camera_name,
                "radar_name": radar_name,
                "lidar_name": lidar_name,
                "camera_token": cam_token,
                "radar_token": radar_token,
                "lidar_token": lidar_token,
                "camera_filename": cam_sd["filename"],
                "camera_width": int(cam_sd["width"]),
                "camera_height": int(cam_sd["height"]),
                "camera_intrinsics": np.asarray(cs_cam["camera_intrinsic"], dtype=np.float32),
            }
            records.append(record)
            sample_token = sample["next"]

    return records


def make_sample_id(scene_name: str, sample_token: str) -> str:
    return f"scene-{scene_name}__sample-{sample_token}"


def get_sample_data_path(nusc: NuScenes, sample_data_token: str) -> str:
    return nusc.get_sample_data_path(sample_data_token)


def get_dynamic_boxes(
    nusc: NuScenes,
    sample_data_token: str,
    prefixes=DYNAMIC_PREFIXES,
):
    """
    Get nuScenes boxes for one sample_data, filtered by dynamic-ish categories.
    Returns boxes in global frame as provided by get_boxes.
    """
    boxes = nusc.get_boxes(sample_data_token)
    out = []
    for box in boxes:
        name = getattr(box, "name", "") or ""
        if name.startswith(prefixes):
            out.append(box)
    return out


def get_ref_camera_intrinsics(nusc: NuScenes, camera_token: str) -> np.ndarray:
    cam_sd = nusc.get("sample_data", camera_token)
    cs_cam = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    return np.asarray(cs_cam["camera_intrinsic"], dtype=np.float32)