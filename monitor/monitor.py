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
        if step % self.log_every != 0:
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

        if "loss_pde" in loss_dict:
            scalars["loss/pde"] = float(loss_dict["loss_pde"].detach().item())
        if "loss_v_pde" in loss_dict:
            scalars["loss/v_pde"] = float(loss_dict["loss_v_pde"].detach().item())
        if "loss_w_pde" in loss_dict:
            scalars["loss/w_pde"] = float(loss_dict["loss_w_pde"].detach().item())
        if "loss_q_pde" in loss_dict:
            scalars["loss/q_pde"] = float(loss_dict["loss_q_pde"].detach().item())

        # 如果 compute_losses 里拆出了 policy_cw / policy_foc，就单独监控
        if "loss_policy_cw" in loss_dict:
            scalars["loss/policy_cw"] = float(loss_dict["loss_policy_cw"].detach().item())
        if "loss_policy_foc" in loss_dict:
            scalars["loss/policy_foc"] = float(loss_dict["loss_policy_foc"].detach().item())

        # NEW: corner loss & monotonicity loss
        if "loss_corner" in loss_dict:
            scalars["loss/corner"] = float(loss_dict["loss_corner"].detach().item())
        if "loss_mono" in loss_dict:
            scalars["loss/mono"] = float(loss_dict["loss_mono"].detach().item())

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

        loss_lines = ["Losses:"]
        if loss_scalars:
            loss_lines.extend(
                f"- {k}:{v:.6f}" for k, v in loss_scalars.items()
            )
        else:
            loss_lines.append("- <none>")

        if other_scalars:
            loss_lines.append("- Scalars:")
            loss_lines.extend(
                f"  - {k}:{v:.6f}" for k, v in other_scalars.items()
            )

        def _format_quantile_block(base_name: str, label: str) -> str | None:
            prefix = f"metrics/{base_name}"
            keys = {
                "min": f"{prefix}_min",
                "p25": f"{prefix}_p25",
                "p50": f"{prefix}_p50",
                "p75": f"{prefix}_p75",
                "max": f"{prefix}_max",
            }
            if not all(k in loss_dict for k in keys.values()):
                return None

            mn = float(loss_dict[keys["min"]].detach().item())
            q25 = float(loss_dict[keys["p25"]].detach().item())
            md = float(loss_dict[keys["p50"]].detach().item())
            q75 = float(loss_dict[keys["p75"]].detach().item())
            mx = float(loss_dict[keys["max"]].detach().item())
            return (
                f"{label}[min,p25,med,p75,max]=[{mn:.3e},{q25:.3e},{md:.3e},{q75:.3e},{mx:.3e}]"
            )

        def _collect_quantile_blocks(groups):
            blocks = []
            for base_name, label in groups.items():
                block = _format_quantile_block(base_name, label)
                if block is not None:
                    blocks.append(block)
            return blocks

        # 2.2 网络及偏导数的分位数
        network_specs = [
            (
                "value_v",
                "v",
                (
                    ("value_grad_V_b", "V_b"),
                    ("value_grad_V_k", "V_k"),
                    ("generator_Dv", "Dv"),
                ),
            ),
            (
                "value_w",
                "w",
                (
                    ("value_grad_W_b", "W_b"),
                    ("value_grad_W_k", "W_k"),
                    ("generator_Dw", "Dw"),
                ),
            ),
            ("bond_price_q", "q", (("generator_Dq", "Dq"),)),
            ("bond_vol_sigma_g", "sigma_g", None),
            ("c_good", "c_good", None),
            ("c_autarky", "c_autarky", None),
            ("labor_good", "labor_good", None),
            ("labor_autarky", "labor_autarky", None),
            ("investment_good", "investment_good", None),
            ("investment_autarky", "investment_autarky", None),
            ("issuance_good", "issuance_good", None),
            ("output_good", "output_good", None),
            ("output_autarky", "output_autarky", None),
            # 如果你在 losses.py 里对 mono_violation / corner_penalty 也做了分位数：
            ("mono_violation_V_k", "mono_V_k", None),
            ("mono_violation_W_k", "mono_W_k", None),
            ("corner_penalty", "corner_penalty", None),
        ]

        network_blocks = []
        for base_name, label, derivs in network_specs:
            main_block = _format_quantile_block(base_name, label)
            if main_block is None:
                continue
            extra_blocks = []
            if derivs is not None:
                if isinstance(derivs, tuple):
                    iter_derivs = derivs
                else:
                    iter_derivs = tuple(derivs)
                for deriv_base, deriv_label in iter_derivs:
                    deriv_block = _format_quantile_block(deriv_base, deriv_label)
                    if deriv_block is not None:
                        extra_blocks.append(deriv_block)
            if extra_blocks:
                main_block = " | ".join([main_block, *extra_blocks])
            network_blocks.append(main_block)
        network_lines = ["Network/Deriv quantiles:"]
        if network_blocks:
            network_lines.extend(f"- {block}" for block in network_blocks)
        else:
            network_lines.append("- <none>")

        # 2.3 Loss target 分位数
        target_sections = [
            (
                "BSDE target quantiles:",
                {
                    "bsde_target_v":       "bsde_target_v",
                    "bsde_target_w":       "bsde_target_w",
                    "bsde_target_q":       "bsde_target_q",
                    "bsde_target_sigma_q": "bsde_target_sigma_q",
                },
            ),
            (
                "PDE target quantiles:",
                {
                    "pde_target_v": "pde_target_v",
                    "pde_target_w": "pde_target_w",
                    "pde_target_q": "pde_target_q",
                },
            ),
        ]

        target_lines = ["Loss target quantiles:"]
        for header, group in target_sections:
            blocks = _collect_quantile_blocks(group)
            target_lines.append(f"- {header}")
            if blocks:
                target_lines.extend(f"  - {block}" for block in blocks)
            else:
                target_lines.append("  - <none>")

        def _print_block(lines):
            for line in lines:
                self.print(line)

        self.print(epoch_line)
        self.print("-")
        _print_block(loss_lines)
        self.print("-")
        _print_block(network_lines)
        self.print("-")
        _print_block(target_lines)

        # ========= 4. TensorBoard 仍然只吃 scalars（纯数字） =========
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(k, v, global_step=step)

    def close(self):
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
