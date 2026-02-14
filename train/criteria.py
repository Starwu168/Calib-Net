from __future__ import annotations
import torch
import torch.nn.functional as F


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

        # 3) 残差幅度约束（防止 ΔS 过大）
        delta_mag = torch.abs((s_up - dep_sp) * mask).mean()

        loss_sparse += ms_w[l] * ls
        loss_consis += ms_w[l] * lc
        loss_energy += ms_w[l] * delta_mag

    w_sparse = cfg_loss["w_sparse"]
    w_consis = cfg_loss["w_consistency"]
    w_energy = cfg_loss["w_energy"]

    total = w_sparse * loss_sparse + w_consis * loss_consis + w_energy * loss_energy
    return total, {
        "loss_sparse": float(loss_sparse.detach().cpu()),
        "loss_consis": float(loss_consis.detach().cpu()),
        "loss_energy": float(loss_energy.detach().cpu()),
    }
