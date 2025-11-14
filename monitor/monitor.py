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

        # ========= 2. 从 loss_dict 抽分位数，拼成更可视的 summary =========
        # compute_losses 里生成的是 metrics/<name>_min/_p25/_p50/_p75/_max
        quantile_groups = {
            "bond_price_q":          "q",                     # 债券价格
            "budget_gap_good_abs":   "|budget_gap_good|",     # 好状态预算残差绝对值
            "budget_gap_autarky_abs":"|budget_gap_autarky|",  # 自给状态预算残差绝对值
            "foc_issuance_abs":      "|policy_foc|",          # 发行 FOC 残差
            "euler_v_resid_abs":     "|analytic_v|",          # v 方程 analytic 残差
            "euler_w_resid_abs":     "|analytic_w|",          # w 方程 analytic 残差
            "bsde_q_resid_abs":      "|bsde_q|",              # q 的 BSDE 残差
        }

        pretty_blocks = []
        for base_name, label in quantile_groups.items():
            prefix = f"metrics/{base_name}"
            keys = {
                "min": f"{prefix}_min",
                "p25": f"{prefix}_p25",
                "p50": f"{prefix}_p50",
                "p75": f"{prefix}_p75",
                "max": f"{prefix}_max",
            }
            # 如果这一批没有算出这些分位数，就跳过
            if not all(k in loss_dict for k in keys.values()):
                continue

            mn  = float(loss_dict[keys["min"]].detach().item())
            q25 = float(loss_dict[keys["p25"]].detach().item())
            md  = float(loss_dict[keys["p50"]].detach().item())
            q75 = float(loss_dict[keys["p75"]].detach().item())
            mx  = float(loss_dict[keys["max"]].detach().item())

            pretty_blocks.append(
                f"{label}[min,p25,med,p75,max]=[{mn:.3e},{q25:.3e},{md:.3e},{q75:.3e},{mx:.3e}]"
            )

        # ========= 3. 终端输出：前半 loss/scalars，后半 分位数 summary =========
        scalar_str = " | ".join([f"{k}:{v:.6f}" for k, v in scalars.items()])
        if pretty_blocks:
            quantile_str = " | ".join(pretty_blocks)
            line = f"[ep {step:05d}] {scalar_str} || {quantile_str}"
        else:
            line = f"[ep {step:05d}] {scalar_str}"

        self.print(line)

        # ========= 4. TensorBoard 仍然只吃 scalars（纯数字） =========
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(k, v, global_step=step)

    def close(self):
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
