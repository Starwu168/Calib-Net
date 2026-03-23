from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from pyquaternion import Quaternion
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt


def transform_points(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    points_xyz: (N, 3)
    T: 4x4
    """
    if points_xyz.size == 0:
        return points_xyz.reshape(0, 3)

    homo = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float32)], axis=1)
    out = (T @ homo.T).T
    return out[:, :3]


def make_transform(translation, rotation_quat) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Quaternion(rotation_quat).rotation_matrix.astype(np.float32)
    T[:3, 3] = np.asarray(translation, dtype=np.float32)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float32)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def sample_data_to_global_transform(nusc: NuScenes, sample_data_token: str) -> np.ndarray:
    sd = nusc.get("sample_data", sample_data_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    T_sensor_to_ego = make_transform(cs["translation"], cs["rotation"])
    T_ego_to_global = make_transform(ep["translation"], ep["rotation"])
    return T_ego_to_global @ T_sensor_to_ego


def global_to_camera_transform(nusc: NuScenes, camera_token: str) -> np.ndarray:
    cam_sd = nusc.get("sample_data", camera_token)
    cs_cam = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    ep_cam = nusc.get("ego_pose", cam_sd["ego_pose_token"])

    T_cam_to_ego = make_transform(cs_cam["translation"], cs_cam["rotation"])
    T_ego_to_global = make_transform(ep_cam["translation"], ep_cam["rotation"])

    T_global_to_ego = invert_transform(T_ego_to_global)
    T_ego_to_cam = invert_transform(T_cam_to_ego)

    return T_ego_to_cam @ T_global_to_ego


def project_camera_points_to_image(
    points_cam: np.ndarray,
    K: np.ndarray,
    image_h: int,
    image_w: int,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """
    points_cam: (N,3) in camera frame.
    return (M,3) = [u,v,d]
    """
    if points_cam.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = points_cam[:, 2]
    valid = (z > 0.0) & (z >= min_depth) & (z <= max_depth)
    pts = points_cam[valid]
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    inside = (u >= 0.0) & (u < image_w) & (v >= 0.0) & (v < image_h)
    u = u[inside]
    v = v[inside]
    z = z[inside]

    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    return np.stack([u, v, z], axis=1).astype(np.float32)


def rasterize_uvd_to_depth_map(
    uvd: np.ndarray,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    """
    Same pixel conflict -> nearest depth.
    """
    depth = np.zeros((image_h, image_w), dtype=np.float32)
    if uvd.size == 0:
        return depth

    uu = np.floor(uvd[:, 0]).astype(np.int32)
    vv = np.floor(uvd[:, 1]).astype(np.int32)
    dd = uvd[:, 2].astype(np.float32)

    inside = (uu >= 0) & (uu < image_w) & (vv >= 0) & (vv < image_h) & (dd > 0)
    uu = uu[inside]
    vv = vv[inside]
    dd = dd[inside]

    for x, y, z in zip(uu, vv, dd):
        cur = depth[y, x]
        if cur <= 0.0 or z < cur:
            depth[y, x] = z
    return depth


def load_lidar_points_sensor(nusc: NuScenes, lidar_token: str) -> np.ndarray:
    path = nusc.get_sample_data_path(lidar_token)
    pc = LidarPointCloud.from_file(path)
    return pc.points[:3, :].T.astype(np.float32)


def load_radar_points_sensor(nusc: NuScenes, radar_token: str) -> np.ndarray:
    path = nusc.get_sample_data_path(radar_token)
    pc = RadarPointCloud.from_file(path)
    return pc.points[:3, :].T.astype(np.float32)


def sensor_points_to_camera_uvd(
    nusc: NuScenes,
    points_sensor_xyz: np.ndarray,
    src_sample_data_token: str,
    ref_camera_token: str,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """
    Transform arbitrary sensor-frame points into reference camera and project to image.
    """
    ref_cam_sd = nusc.get("sample_data", ref_camera_token)
    H = int(ref_cam_sd["height"])
    W = int(ref_cam_sd["width"])
    cs_cam = nusc.get("calibrated_sensor", ref_cam_sd["calibrated_sensor_token"])
    K = np.asarray(cs_cam["camera_intrinsic"], dtype=np.float32)

    T_src_to_global = sample_data_to_global_transform(nusc, src_sample_data_token)
    T_global_to_cam = global_to_camera_transform(nusc, ref_camera_token)

    pts_global = transform_points(points_sensor_xyz, T_src_to_global)
    pts_cam = transform_points(pts_global, T_global_to_cam)
    return project_camera_points_to_image(pts_cam, K, H, W, min_depth, max_depth)


def project_radar_to_uvd(
    nusc: NuScenes,
    radar_token: str,
    camera_token: str,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    pts_sensor = load_radar_points_sensor(nusc, radar_token)
    return sensor_points_to_camera_uvd(
        nusc=nusc,
        points_sensor_xyz=pts_sensor,
        src_sample_data_token=radar_token,
        ref_camera_token=camera_token,
        min_depth=min_depth,
        max_depth=max_depth,
    )


def project_lidar_to_depth_map(
    nusc: NuScenes,
    lidar_token: str,
    camera_token: str,
    min_depth: float,
    max_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        depth_raw: HxW float32
        valid_raw: HxW uint8
    """
    ref_cam_sd = nusc.get("sample_data", camera_token)
    H = int(ref_cam_sd["height"])
    W = int(ref_cam_sd["width"])

    pts_sensor = load_lidar_points_sensor(nusc, lidar_token)
    uvd = sensor_points_to_camera_uvd(
        nusc=nusc,
        points_sensor_xyz=pts_sensor,
        src_sample_data_token=lidar_token,
        ref_camera_token=camera_token,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    depth = rasterize_uvd_to_depth_map(uvd, H, W)
    valid = (depth > 0).astype(np.uint8)
    return depth, valid


def points_in_box_global(points_global: np.ndarray, box) -> np.ndarray:
    """
    box is nuScenes Box in global frame.
    """
    if points_global.size == 0:
        return np.zeros((0,), dtype=bool)

    center = box.center.astype(np.float32)
    size = box.wlh.astype(np.float32)  # width, length, height
    R = box.orientation.rotation_matrix.astype(np.float32)

    local = (points_global - center[None, :]) @ R
    half = size[None, :] / 2.0
    inside = np.all(np.abs(local) <= half + 1e-6, axis=1)
    return inside


def filter_points_by_dynamic_boxes_global(
    points_global: np.ndarray,
    boxes_global: List,
) -> np.ndarray:
    if points_global.size == 0 or len(boxes_global) == 0:
        return np.ones((points_global.shape[0],), dtype=bool)

    keep = np.ones((points_global.shape[0],), dtype=bool)
    for box in boxes_global:
        inside = points_in_box_global(points_global, box)
        keep &= (~inside)
    return keep


def iter_neighbor_sample_tokens(
    nusc: NuScenes,
    ref_sample_token: str,
    n_backward: int,
    n_forward: int,
) -> List[str]:
    sample = nusc.get("sample", ref_sample_token)
    tokens = [ref_sample_token]

    prev_token = sample["prev"]
    for _ in range(n_backward):
        if not prev_token:
            break
        tokens.append(prev_token)
        prev_token = nusc.get("sample", prev_token)["prev"]

    next_token = sample["next"]
    for _ in range(n_forward):
        if not next_token:
            break
        tokens.append(next_token)
        next_token = nusc.get("sample", next_token)["next"]

    return tokens


def merge_lidar_sweeps_to_depth_map(
    nusc: NuScenes,
    ref_sample_token: str,
    ref_camera_token: str,
    lidar_name: str,
    n_backward: int,
    n_forward: int,
    min_depth: float,
    max_depth: float,
    filter_dynamic: bool,
    dynamic_boxes_provider=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge multi-frame LiDAR to reference camera.
    Returns:
        depth_merged_raw: HxW
        valid_merged_raw: HxW
    """
    ref_cam_sd = nusc.get("sample_data", ref_camera_token)
    H = int(ref_cam_sd["height"])
    W = int(ref_cam_sd["width"])
    cs_cam = nusc.get("calibrated_sensor", ref_cam_sd["calibrated_sensor_token"])
    K = np.asarray(cs_cam["camera_intrinsic"], dtype=np.float32)

    T_global_to_cam = global_to_camera_transform(nusc, ref_camera_token)

    all_uvd = []

    neighbor_tokens = iter_neighbor_sample_tokens(nusc, ref_sample_token, n_backward, n_forward)
    for stoken in neighbor_tokens:
        sample = nusc.get("sample", stoken)
        if lidar_name not in sample["data"]:
            continue
        lidar_token = sample["data"][lidar_name]
        pts_sensor = load_lidar_points_sensor(nusc, lidar_token)
        T_lidar_to_global = sample_data_to_global_transform(nusc, lidar_token)
        pts_global = transform_points(pts_sensor, T_lidar_to_global)

        if filter_dynamic and stoken != ref_sample_token and dynamic_boxes_provider is not None:
            boxes = dynamic_boxes_provider(stoken)
            keep = filter_points_by_dynamic_boxes_global(pts_global, boxes)
            pts_global = pts_global[keep]

        pts_cam = transform_points(pts_global, T_global_to_cam)
        uvd = project_camera_points_to_image(
            pts_cam,
            K,
            H,
            W,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if uvd.shape[0] > 0:
            all_uvd.append(uvd)

    if len(all_uvd) == 0:
        depth = np.zeros((H, W), dtype=np.float32)
        valid = np.zeros((H, W), dtype=np.uint8)
        return depth, valid

    all_uvd = np.concatenate(all_uvd, axis=0)
    depth = rasterize_uvd_to_depth_map(all_uvd, H, W)
    valid = (depth > 0).astype(np.uint8)
    return depth, valid


def interpolate_depth_map(
    depth_raw: np.ndarray,
    max_fill_distance: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Conservative interpolation:
    - linear interpolation on valid pixels
    - only keep filled pixels within max_fill_distance from raw support
    Returns:
        depth_interp, valid_interp
    """
    H, W = depth_raw.shape
    ys, xs = np.nonzero(depth_raw > 0)
    vals = depth_raw[ys, xs]

    if vals.size < 4:
        return depth_raw.copy(), (depth_raw > 0).astype(np.uint8)

    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

    depth_linear = griddata(
        points=points,
        values=vals.astype(np.float32),
        xi=(grid_x, grid_y),
        method="linear",
        fill_value=0.0,
    ).astype(np.float32)

    raw_mask = (depth_raw > 0).astype(np.uint8)
    dist = distance_transform_edt(1 - raw_mask).astype(np.float32)

    keep = (depth_linear > 0) & (dist <= max_fill_distance)
    depth_interp = np.where(keep, depth_linear, 0.0).astype(np.float32)

    # preserve raw points exactly
    depth_interp[raw_mask > 0] = depth_raw[raw_mask > 0]

    valid_interp = (depth_interp > 0).astype(np.uint8)
    return depth_interp, valid_interp


def colorize_depth(depth: np.ndarray, max_depth: float = 80.0) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32).copy()
    d[d <= 0] = np.nan
    d = np.clip(d / max_depth, 0.0, 1.0)
    d = (d * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(d, cv2.COLORMAP_JET)
    vis[np.isnan(np.asarray(depth, dtype=np.float32))] = 0
    return vis