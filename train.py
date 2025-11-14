# train.py
# ---------------------------------------------------------
# Minimal training loop for the sovereign default BSDE model
# ---------------------------------------------------------
import os, time, json, math
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

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    dump_config(cfg, cfg["paths"]["cfg_dump"])
    device = get_device(cfg["experiment"]["device"])
    seed_everything(cfg["experiment"]["seed"])

    # Build model / opt / ema
    net = build_model(cfg).to(device)
    opt = build_optimizer(net, cfg)
    ema = EMA(net, cfg["train"]["ema"]["decay"]) if cfg["train"]["ema"]["use"] else None

    # Monitor
    logger = MetricsLogger(cfg)

    epochs = cfg["train"]["epochs"]
    bsz = cfg["train"]["batch_size"]
    grad_clip = float(cfg["train"]["grad_clip"])
    lw_policy = float(cfg["train"]["loss_weights"]["policy"])
    lw_bsde   = float(cfg["train"]["loss_weights"]["bsde"])
    lw_pde    = float(cfg["train"]["loss_weights"]["pde"])
    delta = cfg["numerics"]["Delta"]

    # For reproducible Brownian draws inside losses
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(cfg["experiment"]["seed"] + 1234)

    logger.print(f"Device: {device}  |  Params: {count_params(net):,}")
    logger.print(f"Δ (Delta) = {delta}  |  Batch = {bsz}  |  Epochs = {epochs}")

    for ep in range(1, epochs + 1):
        net.train()
        # 1) sample initial states uniformly from box
        states = sample_states(cfg, bsz, device=device, dtype=torch.float32)

        # 2) compute losses
        loss_dict = compute_losses(net, states, cfg, generator=torch_gen)
        loss_total = (
            lw_bsde   * loss_dict["loss_bsde"]
            + lw_pde  * loss_dict["loss_pde"]
            + lw_policy * loss_dict["loss_policy"]
        )

        # 3) optimize
        opt.zero_grad(set_to_none=True)
        loss_total.backward()
        if grad_clip and grad_clip > 0:
            clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()
        if ema is not None:
            ema.update()

        # 4) log & save
        logger.log_step(ep, loss_total, loss_dict)

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
