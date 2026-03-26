from __future__ import annotations

import torch
import torch.nn.functional as F


def _depth_valid_mask(
    depth: torch.Tensor,
    t_valid: float,
    depth_min: float | None,
    depth_max: float | None,
    ignore_top_rows: int = 0,
) -> torch.Tensor:
    m = depth > t_valid
    if depth_min is not None:
        m = m & (depth > float(depth_min))
    if depth_max is not None:
        m = m & (depth < float(depth_max))
    if ignore_top_rows > 0:
        m[..., :ignore_top_rows, :] = False
    return m


def _remove_outlier_mask(depth: torch.Tensor, valid: torch.Tensor, kernel_size: int, threshold: float):
    if kernel_size <= 1 or threshold <= 0:
        return valid

    max_value = 10.0 * torch.clamp(depth.max(), min=1.0)
    depth_filled = torch.where(valid, depth, torch.full_like(depth, max_value))
    padding = kernel_size // 2
    depth_pad = F.pad(depth_filled, (padding, padding, padding, padding), mode="constant", value=max_value)
    min_values = -F.max_pool2d(-depth_pad, kernel_size=kernel_size, stride=1, padding=0)

    not_outlier = ~(min_values < (depth - threshold))
    return valid & not_outlier


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


def compute_losses(calib_aux_list, dep_gt, cfg_loss: dict, dep_sparse_gt=None):
    ms_w = cfg_loss["ms_weights"]
    assert len(ms_w) == len(calib_aux_list)

    supervision_target = str(cfg_loss.get("supervision_target", "dense")).strip().lower()
    dense_sup_weight = float(cfg_loss.get("dense_sup_weight", 1.0))
    sparse_sup_weight = float(cfg_loss.get("sparse_sup_weight", 1.0))
    use_sparse_target = supervision_target == "sparse" and dep_sparse_gt is not None
    use_hybrid_target = supervision_target in {"dense+sparse", "hybrid"} and dep_sparse_gt is not None
    dep_sup = dep_sparse_gt if use_sparse_target else dep_gt

    _, _, H, W = dep_sup.shape
    t_valid = float(cfg_loss.get("t_valid", 1e-3))
    depth_min = cfg_loss.get("depth_min", None)
    depth_max = cfg_loss.get("depth_max", None)
    ignore_top_rows = int(cfg_loss.get("ignore_top_rows", 0))
    gt_outlier_kernel_size = int(cfg_loss.get("gt_outlier_kernel_size", -1))
    gt_outlier_threshold = float(cfg_loss.get("gt_outlier_threshold", -1.0))
    conf_tau = float(cfg_loss.get("conf_tau", 1.0))
    delta_weights = torch.tensor(
        cfg_loss.get("delta_reg_weights", (1.0, 1.0, 1.0)),
        dtype=dep_sup.dtype,
        device=dep_sup.device,
    ).view(1, 3, 1, 1)
    range_weights = torch.tensor(
        cfg_loss.get("range_reg_weights", (1.0, 1.0, 1.0)),
        dtype=dep_sup.dtype,
        device=dep_sup.device,
    ).view(1, 3, 1, 1)

    valid_gt = _depth_valid_mask(
        dep_sup,
        t_valid=t_valid,
        depth_min=depth_min,
        depth_max=depth_max,
        ignore_top_rows=ignore_top_rows,
    )
    valid_gt = _remove_outlier_mask(
        dep_sup,
        valid_gt,
        gt_outlier_kernel_size,
        gt_outlier_threshold,
    ).float()

    loss_point = dep_sup.new_zeros(())
    loss_conf = dep_sup.new_zeros(())
    loss_delta_reg = dep_sup.new_zeros(())
    loss_range_reg = dep_sup.new_zeros(())
    valid_points_total = dep_sup.new_zeros(())

    mean_abs_delta = dep_sup.new_zeros((3,))
    mean_range = dep_sup.new_zeros((3,))

    for w, calib_aux in zip(ms_w, calib_aux_list):
        proj_u = calib_aux["proj_u"]
        proj_v = calib_aux["proj_v"]
        proj_z = calib_aux["proj_z"]
        proj_conf = calib_aux.get("proj_conf")
        valid = calib_aux["valid"] > 0
        delta_xyz = calib_aux["delta_xyz"]
        range_xyz = calib_aux["range_xyz"]
        Hl, Wl = calib_aux["scale_hw"]

        scale_x = W / float(Wl)
        scale_y = H / float(Hl)
        u_full = proj_u * scale_x
        v_full = proj_v * scale_y

        inside = (u_full >= 0) & (u_full <= (W - 1)) & (v_full >= 0) & (v_full <= (H - 1))
        point_loss = dep_sup.new_zeros(())
        point_denom = dep_sup.new_zeros(())
        conf_loss = dep_sup.new_zeros(())

        if use_hybrid_target:
            dense_valid_gt = _depth_valid_mask(
                dep_gt,
                t_valid=t_valid,
                depth_min=depth_min,
                depth_max=depth_max,
                ignore_top_rows=ignore_top_rows,
            )
            dense_valid_gt = _remove_outlier_mask(
                dep_gt,
                dense_valid_gt,
                gt_outlier_kernel_size,
                gt_outlier_threshold,
            ).float()
            sparse_valid_gt = _depth_valid_mask(
                dep_sparse_gt,
                t_valid=t_valid,
                depth_min=depth_min,
                depth_max=depth_max,
                ignore_top_rows=ignore_top_rows,
            )
            sparse_valid_gt = _remove_outlier_mask(
                dep_sparse_gt,
                sparse_valid_gt,
                gt_outlier_kernel_size,
                gt_outlier_threshold,
            ).float()

            dense_sample = _sample_dense_depth(dep_gt, u_full, v_full)
            dense_valid_sample = _sample_dense_depth(dense_valid_gt, u_full, v_full)
            sparse_sample = _sample_dense_depth(dep_sparse_gt, u_full, v_full)
            sparse_valid_sample = _sample_dense_depth(sparse_valid_gt, u_full, v_full)

            dense_point_mask = valid & inside & (proj_z > t_valid) & (dense_valid_sample > 0.5)
            sparse_point_mask = valid & inside & (proj_z > t_valid) & (sparse_valid_sample > 0.5)

            dense_point_mask_f = dense_point_mask.float()
            sparse_point_mask_f = sparse_point_mask.float()
            dense_point_denom = dense_point_mask_f.sum()
            sparse_point_denom = sparse_point_mask_f.sum()

            if dense_point_denom.detach().item() > 0.0 and dense_sup_weight > 0.0:
                dense_point_loss = F.smooth_l1_loss(
                    proj_z * dense_point_mask_f,
                    dense_sample * dense_point_mask_f,
                    reduction="sum",
                )
                point_loss = point_loss + dense_sup_weight * (dense_point_loss / (dense_point_denom + 1e-6))
                point_denom = point_denom + dense_point_denom
                if proj_conf is not None:
                    dense_target_conf = torch.exp(-(proj_z - dense_sample).abs() / max(conf_tau, 1e-6))
                    dense_conf_loss = F.smooth_l1_loss(
                        proj_conf * dense_point_mask_f,
                        dense_target_conf * dense_point_mask_f,
                        reduction="sum",
                    )
                    conf_loss = conf_loss + dense_sup_weight * (dense_conf_loss / (dense_point_denom + 1e-6))

            if sparse_point_denom.detach().item() > 0.0 and sparse_sup_weight > 0.0:
                sparse_point_loss = F.smooth_l1_loss(
                    proj_z * sparse_point_mask_f,
                    sparse_sample * sparse_point_mask_f,
                    reduction="sum",
                )
                point_loss = point_loss + sparse_sup_weight * (sparse_point_loss / (sparse_point_denom + 1e-6))
                point_denom = point_denom + sparse_point_denom
                if proj_conf is not None:
                    sparse_target_conf = torch.exp(-(proj_z - sparse_sample).abs() / max(conf_tau, 1e-6))
                    sparse_conf_loss = F.smooth_l1_loss(
                        proj_conf * sparse_point_mask_f,
                        sparse_target_conf * sparse_point_mask_f,
                        reduction="sum",
                    )
                    conf_loss = conf_loss + sparse_sup_weight * (sparse_conf_loss / (sparse_point_denom + 1e-6))
        else:
            gt_sample = _sample_dense_depth(dep_sup, u_full, v_full)
            gt_valid_sample = _sample_dense_depth(valid_gt, u_full, v_full)

            point_mask = valid & inside & (proj_z > t_valid) & (gt_valid_sample > 0.5)
            point_mask_f = point_mask.float()
            point_denom = point_mask_f.sum()

            if point_denom.detach().item() > 0.0:
                point_loss = F.smooth_l1_loss(proj_z * point_mask_f, gt_sample * point_mask_f, reduction="sum")
                point_loss = point_loss / (point_denom + 1e-6)
                if proj_conf is not None:
                    target_conf = torch.exp(-(proj_z - gt_sample).abs() / max(conf_tau, 1e-6))
                    conf_loss = F.smooth_l1_loss(
                        proj_conf * point_mask_f,
                        target_conf * point_mask_f,
                        reduction="sum",
                    )
                    conf_loss = conf_loss / (point_denom + 1e-6)

        if point_denom.detach().item() > 0.0:
            loss_point = loss_point + float(w) * point_loss
            loss_conf = loss_conf + float(w) * conf_loss
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
    w_conf = float(cfg_loss.get("w_conf", 0.0))
    w_delta_reg = float(cfg_loss.get("w_delta_reg", 0.0))
    w_range_reg = float(cfg_loss.get("w_range_reg", 0.0))

    total = w_point * loss_point + w_conf * loss_conf + w_delta_reg * loss_delta_reg + w_range_reg * loss_range_reg
    return total, {
        "loss_point": float(loss_point.detach().cpu()),
        "loss_conf": float(loss_conf.detach().cpu()),
        "loss_delta_reg": float(loss_delta_reg.detach().cpu()),
        "loss_range_reg": float(loss_range_reg.detach().cpu()),
        "calib_supervision_sparse": float(1.0 if use_sparse_target else 0.0),
        "calib_supervision_hybrid": float(1.0 if use_hybrid_target else 0.0),
        "mean_abs_delta_x": float(mean_abs_delta[0].detach().cpu()),
        "mean_abs_delta_y": float(mean_abs_delta[1].detach().cpu()),
        "mean_abs_delta_z": float(mean_abs_delta[2].detach().cpu()),
        "mean_range_x": float(mean_range[0].detach().cpu()),
        "mean_range_y": float(mean_range[1].detach().cpu()),
        "mean_range_z": float(mean_range[2].detach().cpu()),
        "num_calib_points": float(valid_points_total.detach().cpu()),
    }
