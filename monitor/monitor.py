# monitor/monitor.py
# ---------------------------------------------------------
# Minimal metrics logger with optional TensorBoard
# ---------------------------------------------------------
from __future__ import annotations
import os, time
from typing import Dict, Any
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_OK = True
except Exception:
    SummaryWriter = None
    _TB_OK = False


class MetricsLogger:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.log_every = int(cfg["monitor"]["log_every"])
        self.save_every = int(cfg["monitor"]["save_every"])
        self.tracked = list(cfg["monitor"]["tracked_metrics"])
        self.tb = None
        if cfg["monitor"]["tensorboard"] and _TB_OK:
            self.tb = SummaryWriter(log_dir=cfg["paths"]["log_dir"])
        self._last_print = time.time()

    def print(self, msg: str):
        print(msg, flush=True)

    def log_step(self, step: int, loss_total: torch.Tensor, loss_dict: Dict[str, torch.Tensor]):
        if step % self.log_every != 1:
            return
        # scalar pack
        scalars = {
            "loss/total": loss_total.detach().item(),
            "loss/bsde": loss_dict["loss_bsde"].detach().item(),
            "loss/policy": loss_dict["loss_policy"].detach().item(),
            "loss/v": loss_dict["loss_v"].detach().item(),
            "loss/w": loss_dict["loss_w"].detach().item(),
            "loss/q": loss_dict["loss_q"].detach().item(),
            "loss/sigma": loss_dict["loss_sigma"].detach().item(),
        }
        # tracked metrics from losses dict (already detached)
        for k in self.tracked:
            if k in loss_dict:
                val = loss_dict[k]
            else:
                # losses.py uses keys like "metrics/xxx"
                mk = f"metrics/{k}" if not k.startswith("metrics/") else k
                val = loss_dict.get(mk, None)
            if val is not None:
                scalars[mk if 'mk' in locals() else k] = float(val.item()) if torch.is_tensor(val) else float(val)

        # print line
        line = f"[ep {step:05d}] " + " | ".join([f"{k}:{v:.6f}" for k, v in scalars.items()])
        self.print(line)

        # tensorboard
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(k, v, global_step=step)

    def close(self):
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
