from __future__ import annotations

import torch
import torch.nn.functional as F


def _depth_valid_mask(
    depth: torch.Tensor,
    t_valid: float,
    depth_min: float | None,
    depth_max: float | None,
) -> torch.Tensor:
    m = depth > t_valid
    if depth_min is not None:
        m = m & (depth > float(depth_min))
    if depth_max is not None:
        m = m & (depth < float(depth_max))
    return m


def _sample_dense_depth(depth: torch.Tensor, u_full: torch.Tensor, v_full: torch.Tensor):
    B, _, H, W = depth.shape
    x_norm = 2.0 * u_full / max(W - 1, 1) - 1.0
    y_norm = 2.0 * v_full / max(H - 1, 1) - 1.0
    if x_norm.dim() == 4 and x_norm.shape[1] == 1:
        x_norm = x_norm.squeeze(1)
    if y_norm.dim() == 4 and y_norm.shape[1] == 1:
        y_norm = y_norm.squeeze(1)
    grid = torch.stack([x_norm, y_norm], dim=-1)
    sampled = F.grid_sample(depth, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled


def compute_losses(calib_aux_list, dep_gt, cfg_loss: dict):
    ms_w = cfg_loss["ms_weights"]
    assert len(ms_w) == len(calib_aux_list)

    _, _, H, W = dep_gt.shape
    t_valid = float(cfg_loss.get("t_valid", 1e-3))
    depth_min = cfg_loss.get("depth_min", None)
    depth_max = cfg_loss.get("depth_max", None)
    delta_weights = torch.tensor(
        cfg_loss.get("delta_reg_weights", (1.0, 1.0, 1.0)),
        dtype=dep_gt.dtype,
        device=dep_gt.device,
    ).view(1, 3, 1, 1)
    range_weights = torch.tensor(
        cfg_loss.get("range_reg_weights", (1.0, 1.0, 1.0)),
        dtype=dep_gt.dtype,
        device=dep_gt.device,
    ).view(1, 3, 1, 1)

    valid_gt = _depth_valid_mask(dep_gt, t_valid=t_valid, depth_min=depth_min, depth_max=depth_max).float()

    loss_point = dep_gt.new_zeros(())
    loss_delta_reg = dep_gt.new_zeros(())
    loss_range_reg = dep_gt.new_zeros(())
    valid_points_total = dep_gt.new_zeros(())

    mean_abs_delta = dep_gt.new_zeros((3,))
    mean_range = dep_gt.new_zeros((3,))

    for w, calib_aux in zip(ms_w, calib_aux_list):
        proj_u = calib_aux["proj_u"]
        proj_v = calib_aux["proj_v"]
        proj_z = calib_aux["proj_z"]
        valid = calib_aux["valid"] > 0
        delta_xyz = calib_aux["delta_xyz"]
        range_xyz = calib_aux["range_xyz"]
        Hl, Wl = calib_aux["scale_hw"]

        scale_x = W / float(Wl)
        scale_y = H / float(Hl)
        u_full = proj_u * scale_x
        v_full = proj_v * scale_y

        inside = (u_full >= 0) & (u_full <= (W - 1)) & (v_full >= 0) & (v_full <= (H - 1))
        gt_sample = _sample_dense_depth(dep_gt, u_full, v_full)
        gt_valid_sample = _sample_dense_depth(valid_gt, u_full, v_full)

        point_mask = valid & inside & (proj_z > t_valid) & (gt_valid_sample > 0.5)
        point_mask_f = point_mask.float()
        point_denom = point_mask_f.sum()

        if point_denom.detach().item() > 0.0:
            point_loss = F.smooth_l1_loss(proj_z * point_mask_f, gt_sample * point_mask_f, reduction="sum")
            point_loss = point_loss / (point_denom + 1e-6)
            loss_point = loss_point + float(w) * point_loss
            valid_points_total = valid_points_total + point_denom

        reg_mask = valid.float()
        reg_denom = reg_mask.sum() + 1e-6

        delta_abs = delta_xyz.abs()
        range_pos = range_xyz

        delta_reg = ((delta_abs * delta_weights) * reg_mask).sum() / reg_denom
        range_reg = ((range_pos * range_weights) * reg_mask).sum() / reg_denom

        loss_delta_reg = loss_delta_reg + float(w) * delta_reg
        loss_range_reg = loss_range_reg + float(w) * range_reg

        mean_abs_delta = mean_abs_delta + float(w) * (delta_abs * reg_mask).sum(dim=(0, 2, 3)) / reg_denom
        mean_range = mean_range + float(w) * (range_pos * reg_mask).sum(dim=(0, 2, 3)) / reg_denom

    w_point = float(cfg_loss.get("w_point", 1.0))
    w_delta_reg = float(cfg_loss.get("w_delta_reg", 0.0))
    w_range_reg = float(cfg_loss.get("w_range_reg", 0.0))

    total = w_point * loss_point + w_delta_reg * loss_delta_reg + w_range_reg * loss_range_reg
    return total, {
        "loss_point": float(loss_point.detach().cpu()),
        "loss_delta_reg": float(loss_delta_reg.detach().cpu()),
        "loss_range_reg": float(loss_range_reg.detach().cpu()),
        "mean_abs_delta_x": float(mean_abs_delta[0].detach().cpu()),
        "mean_abs_delta_y": float(mean_abs_delta[1].detach().cpu()),
        "mean_abs_delta_z": float(mean_abs_delta[2].detach().cpu()),
        "mean_range_x": float(mean_range[0].detach().cpu()),
        "mean_range_y": float(mean_range[1].detach().cpu()),
        "mean_range_z": float(mean_range[2].detach().cpu()),
        "num_calib_points": float(valid_points_total.detach().cpu()),
    }
