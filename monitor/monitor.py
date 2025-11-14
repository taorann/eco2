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

        # ========= 1. 标量部分：loss + 关键均值（给终端 & TensorBoard） =========
        scalars: Dict[str, float] = {
            "loss/total": float(loss_total.detach().item()),
            "loss/bsde":  float(loss_dict["loss_bsde"].detach().item()),
            "loss/policy":float(loss_dict["loss_policy"].detach().item()),
            "loss/v":     float(loss_dict["loss_v"].detach().item()),
            "loss/w":     float(loss_dict["loss_w"].detach().item()),
            "loss/q":     float(loss_dict["loss_q"].detach().item()),
            "loss/sigma": float(loss_dict["loss_sigma"].detach().item()),
        }

        # 如果 compute_losses 里拆出了 policy_cw / policy_foc，就单独监控
        if "loss_policy_cw" in loss_dict:
            scalars["loss/policy_cw"] = float(loss_dict["loss_policy_cw"].detach().item())
        if "loss_policy_foc" in loss_dict:
            scalars["loss/policy_foc"] = float(loss_dict["loss_policy_foc"].detach().item())

        # tracked_metrics 里配置的那些（比如 budget_gap_mean, euler_v_residual_mean 等）
        for k in self.tracked:
            mk = None
            if k in loss_dict:
                val = loss_dict[k]
            else:
                # losses.py uses keys like "metrics/xxx"
                mk = f"metrics/{k}" if not k.startswith("metrics/") else k
                val = loss_dict.get(mk, None)
            if val is not None:
                key = mk if mk is not None else k
                if torch.is_tensor(val):
                    val = float(val.detach().item())
                scalars[key] = val

        # 不再在标量里打印 q_min / q_mean（只看右边分位数可视化即可）
        for drop_key in ("metrics/q_mean", "metrics/q_min"):
            if drop_key in scalars:
                scalars.pop(drop_key)

        # ========= 2. 终端输出：分块展示 =========
        epoch_line = f"================ Epoch {step:05d} ================"

        # 2.1 Losses（以及其他标量）
        loss_scalars = {k: v for k, v in scalars.items() if k.startswith("loss/")}
        other_scalars = {k: v for k, v in scalars.items() if not k.startswith("loss/")}

        loss_parts = [f"{k}:{v:.6f}" for k, v in loss_scalars.items()]
        loss_line = "Losses: " + (" | ".join(loss_parts) if loss_parts else "<none>")

        if other_scalars:
            other_parts = [f"{k}:{v:.6f}" for k, v in other_scalars.items()]
            loss_line += " || Scalars: " + " | ".join(other_parts)

        def _collect_quantile_blocks(groups):
            blocks = []
            for base_name, label in groups.items():
                prefix = f"metrics/{base_name}"
                keys = {
                    "min": f"{prefix}_min",
                    "p25": f"{prefix}_p25",
                    "p50": f"{prefix}_p50",
                    "p75": f"{prefix}_p75",
                    "max": f"{prefix}_max",
                }
                if not all(k in loss_dict for k in keys.values()):
                    continue

                mn = float(loss_dict[keys["min"]].detach().item())
                q25 = float(loss_dict[keys["p25"]].detach().item())
                md = float(loss_dict[keys["p50"]].detach().item())
                q75 = float(loss_dict[keys["p75"]].detach().item())
                mx = float(loss_dict[keys["max"]].detach().item())
                blocks.append(
                    f"{label}[min,p25,med,p75,max]=[{mn:.3e},{q25:.3e},{md:.3e},{q75:.3e},{mx:.3e}]"
                )
            return blocks

        # 2.2 网络及偏导数的分位数
        network_quantile_groups = {
            "value_v":            "v",
            "value_w":            "w",
            "bond_price_q":       "q",
            "bond_vol_sigma_g":   "sigma_g",
            "c_good":             "c_good",
            "c_autarky":          "c_autarky",
            "labor_good":         "labor_good",
            "labor_autarky":      "labor_autarky",
            "investment_good":    "investment_good",
            "investment_autarky": "investment_autarky",
            "issuance_good":      "issuance_good",
            "output_good":        "output_good",
            "output_autarky":     "output_autarky",
            "generator_Dv":       "Dv",
            "generator_Dw":       "Dw",
            "generator_Dq":       "Dq",
            "budget_gap_good_abs":    "|budget_gap_good|",
            "budget_gap_autarky_abs": "|budget_gap_autarky|",
            "foc_issuance_abs":       "|policy_foc|",
            "euler_v_resid_abs":      "|analytic_v|",
            "euler_w_resid_abs":      "|analytic_w|",
            "bsde_q_resid_abs":       "|bsde_q|",
        }
        network_blocks = _collect_quantile_blocks(network_quantile_groups)
        network_line = "Network/Deriv quantiles: "
        network_line += " | ".join(network_blocks) if network_blocks else "<none>"

        # 2.3 Loss target 分位数
        target_quantile_groups = {
            "bsde_target_v":        "bsde_v",
            "bsde_target_w":        "bsde_w",
            "bsde_target_q":        "bsde_q",
            "bsde_target_sigma_q":  "bsde_sigma_q",
            "pde_target_v":         "pde_v",
            "pde_target_w":         "pde_w",
            "pde_target_q":         "pde_q",
        }
        target_blocks = _collect_quantile_blocks(target_quantile_groups)
        target_line = "Loss target quantiles: "
        target_line += " | ".join(target_blocks) if target_blocks else "<none>"

        self.print(epoch_line)
        self.print(loss_line)
        self.print(network_line)
        self.print(target_line)

        # ========= 4. TensorBoard 仍然只吃 scalars（纯数字） =========
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(k, v, global_step=step)

    def close(self):
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
