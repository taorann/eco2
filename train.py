# train.py
# ---------------------------------------------------------
# Training loop with alternating value / policy updates
# (actor-critic style) for the sovereign default BSDE model.
# ---------------------------------------------------------
import os
import time
import json
import math
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


def build_model(cfg):
    hidden = cfg["model"]["hidden_sizes"]
    act = cfg["model"]["activation"]
    dropout = cfg["model"]["dropout"]
    net = SovereignNet(input_dim=4, hidden_sizes=hidden, act=act, dropout=dropout)
    return net


def build_optimizer(net, cfg):
    opt_name = cfg["train"]["optimizer"].lower()
    lr = cfg["train"]["lr"]
    wd = cfg["train"]["weight_decay"]
    betas = tuple(cfg["train"]["betas"])

    if opt_name == "adam":
        return optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=wd)
    elif opt_name == "adamw":
        return optim.AdamW(net.parameters(), lr=lr, betas=betas, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


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
    ema = EMA(net, cfg["train"]["ema"]["decay"]) if cfg["train"]["ema"]["use"] else None
    if ema is not None:
        # 确保 EMA 知道当前跟踪的 live model
        try:
            ema.attach(net)
        except Exception:
            # 老版本 EMA 可能已经在 __init__ 里 attach 了，这里容错一下
            pass

    # 3) monitor/logger
    logger = MetricsLogger(cfg)

    epochs    = int(cfg["train"]["epochs"])
    bsz       = int(cfg["train"]["batch_size"])
    grad_clip = float(cfg["train"]["grad_clip"])
    lw_policy = float(cfg["train"]["loss_weights"]["policy"])
    lw_bsde   = float(cfg["train"]["loss_weights"]["bsde"])
    lw_pde    = float(cfg["train"]["loss_weights"]["pde"])
    lw_corner = float(cfg["train"]["loss_weights"]["corner"])
    delta     = float(cfg["numerics"]["Delta"])
    
    # For reproducible Brownian draws inside losses
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(int(cfg["experiment"]["seed"]) + 1234)

    logger.print(f"Device: {device}  |  Params: {count_params(net):,}")
    logger.print(f"Δ (Delta) = {delta}  |  Batch = {bsz}  |  Epochs = {epochs}")
    logger.print("Training scheme: alternating VALUE / POLICY updates (actor-critic style).")

    # 4) main training loop: alternate value-step and policy-step
    for ep in range(1, epochs + 1):
        net.train()

        # 4.1 sample initial states uniformly from box
        states = sample_states(cfg, bsz, device=device, dtype=torch.float32)

        # 4.2 compute all loss components (no weighting here)
        loss_dict = compute_losses(net, states, cfg, generator=torch_gen)

        # “完整总 loss”（旧版本那种三项加权和），只用于监控，不直接反向
        full_loss_total = (
            lw_bsde   * loss_dict["loss_bsde"]
            + lw_pde  * loss_dict["loss_pde"]
            + lw_policy * loss_dict["loss_policy"]
        )

        # 4.3 choose which objective to backprop: value or policy
        # odd epochs -> VALUE step; even epochs -> POLICY step
        is_policy_step = (ep % 2 == 0)

        # value 部分：BSDE + PDE
        value_obj = lw_bsde * loss_dict["loss_bsde"] + lw_pde * loss_dict["loss_pde"] + lw_corner * loss_dict["loss_corner"]
        # policy 部分：FOC + 自给预算残差
        policy_obj = lw_policy * loss_dict["loss_policy"]

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

        # 4.5 logging
        # - logger.log_step 的第二个参数 loss_total 就是真正用于本次反向的目标
        # - 为了兼容旧脚本，把 full_loss_total / value_obj / policy_obj 也丢进 loss_dict
        loss_dict = dict(loss_dict)  # 浅拷贝以免污染原 dict
        loss_dict["loss_total_full"] = full_loss_total.detach()
        loss_dict["loss_value_step"] = value_obj.detach()
        loss_dict["loss_policy_step"] = policy_obj.detach()
        # 0 = value update, 1 = policy update，方便在 TensorBoard 里看交替模式
        loss_dict["metrics/is_policy_step"] = torch.tensor(
            1.0 if is_policy_step else 0.0,
            device=device,
            dtype=torch.float32,
        )

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

    logger.close()
    logger.print("Training done.")


if __name__ == "__main__":
    main()
