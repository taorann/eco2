# models/sovereign_net.py
# ---------------------------------------------------------
# A compact multi-head MLP for the quantitative sovereign
# default model. Inputs are state variables x = [b, k, s, z].
# Heads (all scalar):
#   - c      : consumption in good standing  (positive)
#   - cw     : consumption in autarky        (positive)
#   - q      : long-term bond price          (positive)
#   - v      : value in good standing        (bounded via tanh)
#   - w      : value in autarky              (bounded via tanh)
#   - sigma_g: diffusion loading of q on W0  (positive)
# ---------------------------------------------------------

from __future__ import annotations
import math
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Small utils
# -------------------------
def _act(name: str) -> nn.Module:
    name = name.lower()
    if name in ("relu",):
        return nn.ReLU(inplace=True)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name in ("gelu",):
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class PositiveHead(nn.Module):
    """Head that ensures strictly positive outputs via Softplus."""
    def __init__(self, in_dim: int, out_dim: int = 1, min_val: float = 1e-6):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.beta = 20.0           # steeper softplus => closer to ReLU but smooth
        self.min_val = float(min_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.lin(x), beta=self.beta) + self.min_val


class BoundedHead(nn.Module):
    """
    Bounded head: scale * tanh(Wx+b).
    用在 v,w 上，避免数值发散，同时导数连续。
    """
    def __init__(self, in_dim: int, out_dim: int = 1, scale: float = 50.0):
        super().__init__()
        assert out_dim == 1, "v/w are scalar heads"
        self.lin = nn.Linear(in_dim, out_dim)
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.lin(x))


def _init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# -------------------------
# Main model
# -------------------------
class SovereignNet(nn.Module):
    """
    Shared trunk + light scalar heads.
    Forward returns a dict of tensors with shape [B, 1] for each head.

    Example:
        net = SovereignNet(input_dim=4, hidden_sizes=[256,256,256], act="silu")
        out = net(torch.randn(32, 4))
        c, cw, q, v, w, sig = out["c"], out["cw"], out["q"], out["v"], out["w"], out["sigma_g"]
    """
    def __init__(
        self,
        input_dim: int = 4,                 # [b, k, s, z]
        hidden_sizes: Iterable[int] = (256, 256, 256),
        act: str = "silu",
        dropout: float = 0.0,
        min_positive: float = 1e-6,
        scale_v: float = 50.0,
        scale_w: float = 50.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        layers: list[nn.Module] = []
        last = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), _act(act)]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        self.trunk = nn.Sequential(*layers)

        # Heads
        self.head_c      = PositiveHead(last, 1, min_val=min_positive)
        self.head_cw     = PositiveHead(last, 1, min_val=min_positive)
        self.head_q      = PositiveHead(last, 1, min_val=min_positive)
        self.head_sigma  = PositiveHead(last, 1, min_val=min_positive)
        # bounded value heads
        self.head_v      = BoundedHead(last, 1, scale=scale_v)
        self.head_w      = BoundedHead(last, 1, scale=scale_w)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 4] tensor with columns (b, k, s, z)

        Returns:
            dict with keys: 'c', 'cw', 'q', 'v', 'w', 'sigma_g'
            each value is [B, 1]
        """
        h = self.trunk(x)
        out = {
            "c":        self.head_c(h),
            "cw":       self.head_cw(h),
            "q":        self.head_q(h),
            "v":        self.head_v(h),
            "w":        self.head_w(h),
            "sigma_g":  self.head_sigma(h),
        }
        return out

    # -------- convenience: value gradients wrt b,k (used in closed-form ι and FOCs) -------
    def value_and_grads(
        self,
        states: torch.Tensor,
        create_graph: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns values (v,w) and their partial derivatives w.r.t. b and k.

        Args:
            states: [B,4] (b,k,s,z)
            create_graph: keep graph for higher-order gradients if needed

        Returns:
            dict with keys: v, w, V_b, V_k, W_b, W_k  (each [B,1])
        """
        states = states.requires_grad_(True)
        fk = 1  # index of k
        fb = 0  # index of b

        out = self.forward(states)
        v, w = out["v"], out["w"]

        V_b = torch.autograd.grad(v, states, grad_outputs=torch.ones_like(v),
                                  retain_graph=True, create_graph=create_graph)[0][:, fb:fb+1]
        V_k = torch.autograd.grad(v, states, grad_outputs=torch.ones_like(v),
                                  retain_graph=True, create_graph=create_graph)[0][:, fk:fk+1]
        W_b = torch.autograd.grad(w, states, grad_outputs=torch.ones_like(w),
                                  retain_graph=True, create_graph=create_graph)[0][:, fb:fb+1]
        W_k = torch.autograd.grad(w, states, grad_outputs=torch.ones_like(w),
                                  retain_graph=True, create_graph=create_graph)[0][:, fk:fk+1]

        return {
            "v": v, "w": w,
            "V_b": V_b, "V_k": V_k,
            "W_b": W_b, "W_k": W_k,
        }

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    net = SovereignNet()
    x = torch.randn(5, 4)
    out = net(x)
    vg = net.value_and_grads(x)
    for k, v in out.items():
        print(k, v.shape, v.min().item(), v.max().item())
    for k, v in vg.items():
        print("grad:", k, v.shape)
    print("params:", net.n_parameters())
