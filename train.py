# Training loop with alternating value / policy updates
# (actor-critic style) for the sovereign default BSDE model.
#
# - 奇数 epoch：VALUE step  —— 优化 BSDE + PDE + corner + mono_k
# - 偶数 epoch：POLICY step —— 优化 policy loss (FOC + cw 预算)
#
# 不再使用自适应权重，所有权重来自 config.train.loss_weights：
#   bsde, pde, policy, corner, mono_k
# ---------------------------------------------------------

from __future__ import annotations
import os
import argparse

import torch
import torch.optim as optim

from utils.config import load_config, ensure_dirs, dump_config
from utils.misc import seed_everything, get_device, count_params, clip_grad_norm_
from utils.ema import EMA
from monitor.monitor import MetricsLogger
from data.sampler import sample_states

from models.sovereign_net import SovereignNet
from losses import compute_losses
from export_grid_excel import export_state_grid_excel


# ============================================================
#  Builders
# ============================================================

def build_model(cfg) -> SovereignNet:
    hidden  = cfg["model"]["hidden_sizes"]
    act     = cfg["model"]["activation"]
    dropout = cfg["model"]["dropout"]
    net = SovereignNet(input_dim=4, hidden_sizes=hidden, act=act, dropout=dropout)
    return net


def build_optimizer(net: torch.nn.Module, cfg):
    opt_name = cfg["train"]["optimizer"].lower()
    lr       = cfg["train"]["lr"]
    wd       = cfg["train"]["weight_decay"]
    betas    = tuple(cfg["train"]["betas"])

    if opt_name == "adam":
        return optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=wd)
    elif opt_name == "adamw":
        return optim.AdamW(net.parameters(), lr=lr, betas=betas, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# ============================================================
#  Main training loop
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()

    # 1) load config / setup paths & device
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    dump_config(cfg, cfg["paths"]["cfg_dump"])

    device = get_device(cfg["experiment"]["device"])
    seed_everything(cfg["experiment"]["seed"])

    # 2) build model / optimizer / EMA
    net = build_model(cfg).to(device)
    opt = build_optimizer(net, cfg)

    ema_cfg = cfg["train"]["ema"]
    ema = EMA(net, ema_cfg["decay"]) if ema_cfg["use"] else None
    if ema is not None:
        try:
            ema.attach(net)
        except Exception:
            # 某些实现可能在 __init__ 里已经 attach 过了
            pass

    # 3) monitor/logger
    logger = MetricsLogger(cfg)

    epochs    = int(cfg["train"]["epochs"])
    bsz       = int(cfg["train"]["batch_size"])
    grad_clip = float(cfg["train"]["grad_clip"])

    lw_cfg   = cfg["train"]["loss_weights"]
    lw_bsde   = float(lw_cfg["bsde"])
    lw_pde    = float(lw_cfg["pde"])
    lw_policy = float(lw_cfg["policy"])
    lw_corner = float(lw_cfg.get("corner", 0.0))
    lw_mono   = float(lw_cfg.get("mono_k", 0.0))

    Delta = float(cfg["numerics"]["Delta"])

    # For reproducible Brownian draws inside losses
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(int(cfg["experiment"]["seed"]) + 1234)

    logger.print(f"Device: {device}  |  Params: {count_params(net):,}")
    logger.print(f"Δ (Delta) = {Delta}  |  Batch = {bsz}  |  Epochs = {epochs}")
    logger.print("Training scheme: alternating VALUE / POLICY updates (actor-critic style).")
    logger.print("Adaptive loss weights: DISABLED (using fixed weights from config).")
    logger.print(
        "loss_weights:"
        f" bsde={lw_bsde}, pde={lw_pde}, policy={lw_policy},"
        f" corner={lw_corner}, mono_k={lw_mono}"
    )

    # 4) main training loop: alternate value-step and policy-step
    for ep in range(1, epochs + 1):
        net.train()

        # 4.1 sample initial states uniformly from box
        states = sample_states(cfg, bsz, device=device, dtype=torch.float32)

        # 4.2 compute all loss components（原始 loss，不带权重）
        loss_dict = compute_losses(net, states, cfg, generator=torch_gen)

        # 4.3 value / policy 交替
        # 奇数 epoch：value step；偶数 epoch：policy step
        is_policy_step = (ep % 2 == 0)

        # 原始各项 loss
        loss_bsde   = loss_dict["loss_bsde"]
        loss_pde    = loss_dict["loss_pde"]
        loss_policy = loss_dict["loss_policy"]
        loss_corner = loss_dict.get("loss_corner", torch.zeros((), device=device))
        # 注意：单调性约束对应 key 为 "loss_mono_k"
        loss_mono_k  = loss_dict.get("loss_mono_k",  0.0)
        loss_mono_vb = loss_dict.get("loss_mono_vb", 0.0)
        loss_mono    = loss_mono_k + 0.0001*loss_mono_vb

        
        # ----- 4.3.1 VALUE objective: BSDE + PDE + corner + mono_k -----
        value_obj = (
            lw_bsde   * loss_bsde
            + lw_pde  * loss_pde
            + lw_corner * loss_corner
            + lw_mono   * loss_mono
        )

        # ----- 4.3.2 POLICY objective: FOC + 自给预算残差 -----
        policy_obj = lw_policy * loss_policy

        if is_policy_step:
            loss_obj = policy_obj
        else:
            loss_obj = value_obj

        # 4.4 optimization
        opt.zero_grad(set_to_none=True)
        loss_obj.backward()
        if grad_clip and grad_clip > 0:
            clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()
        if ema is not None:
            ema.update()

        # 4.5 logging 组装
        loss_dict = dict(loss_dict)  # 浅拷贝以免污染原 dict

        # 使用“固定权重”的 total（只用于监控对比）
        full_loss_total = (
            lw_bsde   * loss_bsde
            + lw_pde  * loss_pde
            + lw_policy * loss_policy
            + lw_corner * loss_corner
            + lw_mono   * loss_mono
        )


        loss_dict["loss_total_full"]  = full_loss_total.detach()
        loss_dict["loss_value_step"]  = value_obj.detach()
        loss_dict["loss_policy_step"] = policy_obj.detach()
        loss_dict["metrics/is_policy_step"] = torch.tensor(
            1.0 if is_policy_step else 0.0,
            device=device,
            dtype=torch.float32,
        )

        # NOTE: monitor 里会用传入的 loss_obj 作为 "loss/total"
        logger.log_step(ep, loss_obj, loss_dict)

        # 4.6 checkpoint
        if ep % cfg["monitor"]["save_every"] == 0 or ep == epochs:
            ckpt_path = os.path.join(cfg["paths"]["ckpt_dir"], f"ckpt_ep{ep}.pt")
            payload = {
                "epoch": ep,
                "model": (ema.shadow if ema else net).state_dict(),
                "optimizer": opt.state_dict(),
                "cfg": cfg,
            }
            torch.save(payload, ckpt_path)
            logger.print(f"[ckpt] saved: {ckpt_path}")

    try:
        model_for_export = ema.shadow if ema is not None else net
        excel_path = os.path.join(cfg["paths"]["work_dir"], "state_grid.xlsx")
        export_state_grid_excel(model_for_export, cfg, excel_path)
        logger.print(f"[excel] state grid saved to {excel_path}")
    except Exception as e:
        logger.print(f"[excel] failed to export state grid: {e}")

    logger.close()
    logger.print("Training done.")
if __name__ == "__main__":
    main()





# #=====================================================自适应权重=================================
# # train.py
# # ---------------------------------------------------------
# # Training loop with alternating value / policy updates
# # (actor-critic style) for the sovereign default BSDE model.
# # Now with adaptive weights for:
# #   - each PDE component: loss_v_pde, loss_w_pde, loss_q_pde
# #   - BSDE aggregate: loss_bsde
# # ---------------------------------------------------------
# import os
# import time
# import json
# import math
# import argparse
# from typing import Dict

# import torch
# import torch.optim as optim

# from utils.config import load_config, ensure_dirs, dump_config
# from utils.misc import seed_everything, get_device, count_params, clip_grad_norm_
# from utils.ema import EMA
# from monitor.monitor import MetricsLogger
# from data.sampler import sample_states

# from models.sovereign_net import SovereignNet
# from losses import compute_losses


# class AdaptivePDEWeights:
#     """
#     EMA-based adaptive weighting for the three PDE loss components:
#       - loss_v_pde
#       - loss_w_pde
#       - loss_q_pde

#     记 m_k 为 |loss_k| 的指数滑动均值，则
#         avg = (m_v + m_w + m_q) / 3
#         scale_k = avg / (m_k + eps)
#     这样三个方向在目标里的“有效尺度”更接近。
#     """

#     def __init__(self, beta: float = 0.99, eps: float = 1e-8):
#         self.beta = beta
#         self.eps = eps
#         self.names = ["loss_v_pde", "loss_w_pde", "loss_q_pde"]
#         # EMA 存成 python float
#         self.ema: Dict[str, float | None] = {name: None for name in self.names}

#     def update(self, loss_dict: Dict[str, torch.Tensor]):
#         for name in self.names:
#             if name not in loss_dict:
#                 continue
#             v = loss_dict[name]
#             if torch.is_tensor(v):
#                 v = float(v.detach().abs().mean().item())
#             else:
#                 v = abs(float(v))
#             old = self.ema[name]
#             if old is None:
#                 self.ema[name] = v
#             else:
#                 self.ema[name] = self.beta * old + (1.0 - self.beta) * v

#     def get_scales(self) -> Dict[str, float]:
#         # 还没 warmup 好时，统一用 1.0
#         if any(self.ema[name] is None for name in self.names):
#             return {name: 1.0 for name in self.names}
#         avg = sum(self.ema[name] for name in self.names) / len(self.names)
#         return {
#             name: float(avg / (self.ema[name] + self.eps))
#             for name in self.names
#         }


# class AdaptiveScalarWeight:
#     """
#     对单个 loss 做 EMA 自适应权重，例如整体的 loss_bsde：

#         m = EMA(|loss_bsde|)
#         scale = 1 / (m + eps)

#     最终在目标里用：
#         lw_bsde * scale * loss_bsde
#     """

#     def __init__(self, beta: float = 0.99, eps: float = 1e-8):
#         self.beta = beta
#         self.eps = eps
#         self.ema: float | None = None

#     def update_from_dict(self, loss_dict: Dict[str, torch.Tensor], key: str):
#         if key not in loss_dict:
#             return
#         v = loss_dict[key]
#         if torch.is_tensor(v):
#             v = float(v.detach().abs().mean().item())
#         else:
#             v = abs(float(v))
#         if self.ema is None:
#             self.ema = v
#         else:
#             self.ema = self.beta * self.ema + (1.0 - self.beta) * v

#     def get_scale(self) -> float:
#         if self.ema is None:
#             return 1.0
#         return float(1.0 / (self.ema + self.eps))


# def build_model(cfg):
#     hidden = cfg["model"]["hidden_sizes"]
#     act = cfg["model"]["activation"]
#     dropout = cfg["model"]["dropout"]
#     net = SovereignNet(input_dim=4, hidden_sizes=hidden, act=act, dropout=dropout)
#     return net


# def build_optimizer(net, cfg):
#     opt_name = cfg["train"]["optimizer"].lower()
#     lr = cfg["train"]["lr"]
#     wd = cfg["train"]["weight_decay"]
#     betas = tuple(cfg["train"]["betas"])

#     if opt_name == "adam":
#         return optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=wd)
#     elif opt_name == "adamw":
#         return optim.AdamW(net.parameters(), lr=lr, betas=betas, weight_decay=wd)
#     else:
#         raise ValueError(f"Unknown optimizer: {opt_name}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="config/default.yaml")
#     args = parser.parse_args()

#     # 1) load config / setup paths & device
#     cfg = load_config(args.config)
#     ensure_dirs(cfg)
#     dump_config(cfg, cfg["paths"]["cfg_dump"])
#     device = get_device(cfg["experiment"]["device"])
#     seed_everything(cfg["experiment"]["seed"])

#     # 2) build model / optimizer / EMA
#     net = build_model(cfg).to(device)
#     opt = build_optimizer(net, cfg)
#     ema = EMA(net, cfg["train"]["ema"]["decay"]) if cfg["train"]["ema"]["use"] else None
#     if ema is not None:
#         try:
#             ema.attach(net)
#         except Exception:
#             # 某些实现可能在 __init__ 里已经 attach 过了
#             pass

#     # 3) monitor/logger
#     logger = MetricsLogger(cfg)

#     epochs    = int(cfg["train"]["epochs"])
#     bsz       = int(cfg["train"]["batch_size"])
#     grad_clip = float(cfg["train"]["grad_clip"])
#     lw_policy = float(cfg["train"]["loss_weights"]["policy"])
#     lw_bsde   = float(cfg["train"]["loss_weights"]["bsde"])
#     lw_pde    = float(cfg["train"]["loss_weights"]["pde"])
#     lw_corner = float(cfg["train"]["loss_weights"]["corner"])
#     delta     = float(cfg["numerics"]["Delta"])

#     # PDE 自适应权重配置
#     adaptive_pde_cfg = cfg["train"].get("adaptive_pde", {})
#     use_adaptive_pde = bool(adaptive_pde_cfg.get("use", True))
#     pde_beta = float(adaptive_pde_cfg.get("beta", 0.99))
#     pde_eps  = float(adaptive_pde_cfg.get("eps", 1e-8))
#     adaptive_pde = AdaptivePDEWeights(beta=pde_beta, eps=pde_eps) if use_adaptive_pde else None

#     # BSDE 自适应权重配置（整体一个权重）
#     adaptive_bsde_cfg = cfg["train"].get("adaptive_bsde", {})
#     use_adaptive_bsde = bool(adaptive_bsde_cfg.get("use", True))
#     bsde_beta = float(adaptive_bsde_cfg.get("beta", 0.99))
#     bsde_eps  = float(adaptive_bsde_cfg.get("eps", 1e-8))
#     adaptive_bsde = AdaptiveScalarWeight(beta=bsde_beta, eps=bsde_eps) if use_adaptive_bsde else None

#     # For reproducible Brownian draws inside losses
#     torch_gen = torch.Generator(device=device)
#     torch_gen.manual_seed(int(cfg["experiment"]["seed"]) + 1234)

#     logger.print(f"Device: {device}  |  Params: {count_params(net):,}")
#     logger.print(f"Δ (Delta) = {delta}  |  Batch = {bsz}  |  Epochs = {epochs}")
#     logger.print("Training scheme: alternating VALUE / POLICY updates (actor-critic style).")
#     if adaptive_pde is not None:
#         logger.print(f"Adaptive PDE weights: ENABLED  (beta={pde_beta}, eps={pde_eps})")
#     else:
#         logger.print("Adaptive PDE weights: DISABLED")
#     if adaptive_bsde is not None:
#         logger.print(f"Adaptive BSDE weight: ENABLED  (beta={bsde_beta}, eps={bsde_eps})")
#     else:
#         logger.print("Adaptive BSDE weight: DISABLED")

#     # 4) main training loop: alternate value-step and policy-step
#     for ep in range(1, epochs + 1):
#         net.train()

#         # 4.1 sample initial states uniformly from box
#         states = sample_states(cfg, bsz, device=device, dtype=torch.float32)

#         # 4.2 compute all loss components (原始 loss，不带权重)
#         loss_dict = compute_losses(net, states, cfg, generator=torch_gen)

#         # 先更新自适应权重相关的 EMA
#         # ---- BSDE ----
#         if adaptive_bsde is not None:
#             adaptive_bsde.update_from_dict(loss_dict, "loss_bsde")
#             bsde_scale = adaptive_bsde.get_scale()
#         else:
#             bsde_scale = 1.0

#         # ---- PDE 三个分量 ----
#         if adaptive_pde is not None:
#             adaptive_pde.update(loss_dict)
#             pde_scales = adaptive_pde.get_scales()
#             s_v = pde_scales["loss_v_pde"]
#             s_w = pde_scales["loss_w_pde"]
#             s_q = pde_scales["loss_q_pde"]

#             # 自适应 PDE：三个方向各自乘以 scale，再一起乘 lw_pde
#             loss_pde_eff = lw_pde * (
#                 s_v * loss_dict["loss_v_pde"]
#                 + s_w * loss_dict["loss_w_pde"]
#                 + 100*s_q * loss_dict["loss_q_pde"]
#                 + lw_corner * loss_dict["loss_corner"]
#             )

#             # 记录方便对比
#             loss_dict["loss_pde_adaptive"] = loss_pde_eff.detach()
#         else:
#             loss_pde_eff = lw_pde * loss_dict["loss_pde"]
#             s_v = s_w = s_q = 1.0

#         # “完整总 loss”（老版固定权重，仅用于监控对比，不参与反向）
#         full_loss_total = (
#             lw_bsde   * loss_dict["loss_bsde"]
#             + lw_pde  * loss_dict["loss_pde"]
#             + lw_policy * loss_dict["loss_policy"]
#             + lw_corner * loss_dict["loss_corner"]
#         )

#         # 4.3 value / policy 交替
#         # 奇数 epoch：value step；偶数 epoch：policy step
#         is_policy_step = (ep % 2 == 0)

#         # ----- 4.3.1 VALUE objective: BSDE + (adaptive) PDE + corner penalty -----
#         # BSDE 整体自适应权重
#         loss_bsde_eff = lw_bsde * bsde_scale * loss_dict["loss_bsde"]
#         loss_dict["loss_bsde_adaptive"] = loss_bsde_eff.detach()

#         value_obj = (
#             loss_bsde_eff
#             + loss_pde_eff
#             + lw_corner * loss_dict["loss_corner"]
#         )

#         # ----- 4.3.2 POLICY objective: FOC + 自给预算残差 -----
#         policy_obj = lw_policy * loss_dict["loss_policy"]

#         if is_policy_step:
#             loss_obj = policy_obj
#         else:
#             loss_obj = value_obj

#         # 4.4 optimization
#         opt.zero_grad(set_to_none=True)
#         loss_obj.backward()
#         if grad_clip and grad_clip > 0:
#             clip_grad_norm_(net.parameters(), grad_clip)
#         opt.step()
#         if ema is not None:
#             ema.update()

#         # 4.5 logging
#         loss_dict = dict(loss_dict)  # 浅拷贝以免污染原 dict

#         # 记录 effective loss 和权重 / scale
#         loss_dict["loss_total_full"]  = full_loss_total.detach()
#         loss_dict["loss_value_step"]  = value_obj.detach()
#         loss_dict["loss_policy_step"] = policy_obj.detach()
#         loss_dict["metrics/is_policy_step"] = torch.tensor(
#             1.0 if is_policy_step else 0.0,
#             device=device,
#             dtype=torch.float32,
#         )

#         # 监控 BSDE 的自适应权重
#         loss_dict["metrics/bsde_scale"] = torch.tensor(
#             bsde_scale, device=device, dtype=torch.float32
#         )
#         if adaptive_bsde is not None and adaptive_bsde.ema is not None:
#             loss_dict["metrics/bsde_ema"] = torch.tensor(
#                 adaptive_bsde.ema, device=device, dtype=torch.float32
#             )

#         # 监控 PDE 三个分量的自适应权重
#         loss_dict["metrics/pde_scale_v"] = torch.tensor(
#             s_v, device=device, dtype=torch.float32
#         )
#         loss_dict["metrics/pde_scale_w"] = torch.tensor(
#             s_w, device=device, dtype=torch.float32
#         )
#         loss_dict["metrics/pde_scale_q"] = torch.tensor(
#             s_q, device=device, dtype=torch.float32
#         )

#         logger.log_step(ep, loss_obj, loss_dict)

#         # 4.6 checkpoint
#         if ep % cfg["monitor"]["save_every"] == 0 or ep == epochs:
#             ckpt_path = os.path.join(cfg["paths"]["ckpt_dir"], f"ckpt_ep{ep}.pt")
#             payload = {
#                 "epoch": ep,
#                 "model": (ema.shadow if ema else net).state_dict(),
#                 "optimizer": opt.state_dict(),
#                 "cfg": cfg,
#             }
#             torch.save(payload, ckpt_path)
#             logger.print(f"[ckpt] saved: {ckpt_path}")

#     logger.close()
#     logger.print("Training done.")


# if __name__ == "__main__":
#     main()
