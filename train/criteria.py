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


def compute_losses(s_pred_list, dep_sp, dep_gt, cfg_loss: dict, dep_sparse_gt=None):
    """
    s_pred_list: list of S_l' (B,1,Hl,Wl)
    dep_sp:      (B,1,H,W)
    dep_gt:      (B,1,H,W)
    dep_sparse_gt(optional): (B,1,H,W), sparse lidar GT
    """
    ms_w = cfg_loss["ms_weights"]
    assert len(ms_w) == len(s_pred_list)

    _, _, H, W = dep_sp.shape
    t_valid = float(cfg_loss.get("t_valid", 1e-3))
    depth_min = cfg_loss.get("depth_min", None)
    depth_max = cfg_loss.get("depth_max", None)
    target_source = str(cfg_loss.get("target_source", "")).strip().lower()
    mask_source = str(cfg_loss.get("valid_mask_source", "")).strip().lower()
    use_sparse_gt_mask = bool(cfg_loss.get("use_sparse_gt_mask", True))
    outlier_kernel = int(cfg_loss.get("gt_outlier_kernel_size", -1))
    outlier_thr = float(cfg_loss.get("gt_outlier_threshold", -1.0))

    mask_sparse = (dep_sp > 0).float()
    dense_valid = _depth_valid_mask(dep_gt, t_valid=t_valid, depth_min=depth_min, depth_max=depth_max)
    dense_valid = _remove_outlier_mask(dep_gt, dense_valid, outlier_kernel, outlier_thr)

    sparse_valid = None
    if dep_sparse_gt is not None:
        sparse_valid = _depth_valid_mask(
            dep_sparse_gt, t_valid=t_valid, depth_min=depth_min, depth_max=depth_max
        )
        sparse_valid = _remove_outlier_mask(dep_sparse_gt, sparse_valid, outlier_kernel, outlier_thr)

    if target_source == "sparse" and dep_sparse_gt is not None:
        target = torch.where(sparse_valid, dep_sparse_gt, dep_gt)
    elif target_source == "dense":
        target = dep_gt
    elif dep_sparse_gt is not None and use_sparse_gt_mask:
        target = torch.where(sparse_valid, dep_sparse_gt, dep_gt)
    else:
        target = dep_gt

    if mask_source == "sparse" and sparse_valid is not None:
        gt_valid = sparse_valid
    elif mask_source == "dense":
        gt_valid = dense_valid
    elif mask_source == "target":
        gt_valid = _depth_valid_mask(target, t_valid=t_valid, depth_min=depth_min, depth_max=depth_max)
        gt_valid = _remove_outlier_mask(target, gt_valid, outlier_kernel, outlier_thr)
    elif dep_sparse_gt is not None and use_sparse_gt_mask and sparse_valid is not None:
        gt_valid = sparse_valid
    else:
        gt_valid = dense_valid

    mask = mask_sparse * gt_valid.float()
    mask_inv = 1.0 - mask_sparse

    loss_sparse = 0.0
    loss_consis = 0.0
    loss_energy = 0.0
    loss_outside = 0.0

    for l, s_l in enumerate(s_pred_list):
        s_up = F.interpolate(s_l, size=(H, W), mode="nearest")

        ls = F.smooth_l1_loss(s_up * mask, target * mask, reduction="sum") / (mask.sum() + 1e-6)
        lc = F.smooth_l1_loss(s_up * mask, dep_sp * mask, reduction="sum") / (mask.sum() + 1e-6)
        delta_mag = torch.abs((s_up - dep_sp) * mask).sum() / (mask.sum() + 1e-6)
        l_outside = torch.abs(s_up * mask_inv).sum() / (mask_inv.sum() + 1e-6)

        loss_sparse += ms_w[l] * ls
        loss_consis += ms_w[l] * lc
        loss_energy += ms_w[l] * delta_mag
        loss_outside += ms_w[l] * l_outside

    w_sparse = cfg_loss["w_sparse"]
    w_consis = cfg_loss["w_consistency"]
    w_energy = cfg_loss["w_energy"]
    w_outside = cfg_loss["w_outside"]

    total = (
        w_sparse * loss_sparse
        + w_consis * loss_consis
        + w_energy * loss_energy
        + w_outside * loss_outside
    )
    return total, {
        "loss_sparse": float(loss_sparse.detach().cpu()),
        "loss_consis": float(loss_consis.detach().cpu()),
        "loss_energy": float(loss_energy.detach().cpu()),
        "loss_outside": float(loss_outside.detach().cpu()),
    }
