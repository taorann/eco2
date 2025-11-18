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
    def __init__(self, in_dim: int, out_dim: int = 1, min_val: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.beta = 20.0           # steeper softplus => closer to ReLU but smooth
        self.min_val = float(min_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.lin(x), beta=self.beta) + self.min_val

class SigmoidBoundedHead(nn.Module):
    """
    q-head: 0 < q < q_max via sigmoid.
    典型用法：选 q_max=2 => 初始 q ≈ 1（因为 sigmoid(0)=0.5）
    """
    def __init__(self, in_dim: int, out_dim: int = 1, max_val: float = 2.0):
        super().__init__()
        assert out_dim == 1
        self.lin = nn.Linear(in_dim, out_dim)
        self.max_val = float(max_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 0 < sigmoid(.) < 1, 所以 0 < q < max_val
        return self.max_val * torch.sigmoid(self.lin(x))


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
        return self.scale * torch.tanh(self.lin(x)/10.0)


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
    Multi-head network for the quantitative sovereign default model.

    Inputs are state variables x = [b, k, s, z].

    结构设计：
    - 首先对 (k,s,z) 做共享的 state 表征 h_state；
    - autarky 分支 (cw, w) 只依赖 h_state，不看 b；
    - good-standing 分支 (c, q, v, sigma_g) 在 h_state 上再接入 b 的 embedding。
    """

    def __init__(
        self,
        input_dim: int = 4,                 # [b, k, s, z]
        hidden_sizes: Iterable[int] = (256),
        act: str = "silu",
        dropout: float = 0.0,
        min_positive: float = 1e-6,
        scale_v: float = 50.0,
        scale_w: float = 50.0,
        max_q: float = 2.0,
    ):
        super().__init__()
        assert input_dim == 4, "SovereignNet expects x=[b,k,s,z] with dim=4"
        self.input_dim = input_dim

        hidden_sizes = tuple(hidden_sizes)
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must be non-empty")

        self.act_name = act
        self.dropout = float(dropout)

        # ===== 1. 每个状态变量的 scalar encoder =====
        # 这里用同一个 embed_dim，直接取第一个 hidden size，方便与 trunk 对接
        embed_dim = hidden_sizes[0]
        self.embed_dim = embed_dim

        def make_scalar_encoder() -> nn.Sequential:
            layers: list[nn.Module] = [nn.Linear(1, embed_dim), _act(act)]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            return nn.Sequential(*layers)

        # b, k, s, z 各自的 encoder
        self.enc_b = make_scalar_encoder()
        self.enc_k = make_scalar_encoder()
        self.enc_s = make_scalar_encoder()
        self.enc_z = make_scalar_encoder()

        # ===== 2. shared state trunk：只吃 (k,s,z) 的 embedding =====
        def make_trunk(input_dim_trunk: int) -> Tuple[nn.Sequential, int]:
            layers: list[nn.Module] = []
            last = input_dim_trunk
            for h in hidden_sizes:
                layers.append(nn.Linear(last, h))
                layers.append(_act(act))
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
                last = h
            return nn.Sequential(*layers), last

        # (k,s,z) 的共享表示
        self.trunk_state, state_dim = make_trunk(input_dim_trunk=3 * embed_dim)

        # ===== 3. autarky 分支：只依赖 h_state（不接 b） =====
        # cw, w 都走 trunk_aut(h_state)
        self.trunk_aut, aut_dim = make_trunk(input_dim_trunk=state_dim)

        # ===== 4. good-standing 分支：在 h_state 上拼接 b 的 embedding =====
        # c, q, v, sigma_g 走 trunk_good([h_state, h_b])
        self.trunk_good, good_dim = make_trunk(input_dim_trunk=state_dim + embed_dim)

        # ===== 5. Heads =====
        # good-standing heads: 依赖 (b,k,s,z) 通过 h_state + h_b
        self.head_c      = SigmoidBoundedHead(good_dim, 1, max_val=max_q)
        self.head_q      = SigmoidBoundedHead(good_dim, 1, max_val=max_q)
        self.head_sigma  = PositiveHead(good_dim, 1, min_val=min_positive)
        self.head_v      = BoundedHead(good_dim, 1, scale=scale_v)

        # autarky heads: 只依赖 (k,s,z)，通过 h_state → trunk_aut
        self.head_cw     = SigmoidBoundedHead(aut_dim, 1, max_val=max_q)
        self.head_w      = BoundedHead(aut_dim, 1, scale=scale_w)

        # 初始化权重（沿用你原来的工具函数）
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 4] tensor with columns (b, k, s, z)

        Returns:
            dict with keys: 'c', 'cw', 'q', 'v', 'w', 'sigma_g'
            each value has shape [B, 1]
        """
        assert x.shape[-1] == 4, f"Expected x shape [B,4], got {x.shape}"

        # 拆出各个状态变量
        b = x[:, 0:1]  # [B,1]
        k = x[:, 1:2]  # [B,1]
        s = x[:, 2:3]  # [B,1]
        z = x[:, 3:4]  # [B,1]

        # ---- scalar encoders ----
        h_b = self.enc_b(b)   # [B, embed_dim]
        h_k = self.enc_k(k)
        h_s = self.enc_s(s)
        h_z = self.enc_z(z)

        # ---- shared state representation from (k,s,z) ----
        h_state_in = torch.cat([h_k, h_s, h_z], dim=-1)   # [B, 3*embed_dim]
        h_state    = self.trunk_state(h_state_in)         # [B, state_dim]

        # ---- autarky branch: only (k,s,z) ----
        h_aut = self.trunk_aut(h_state)                   # [B, aut_dim]

        # ---- good-standing branch: (k,s,z) + b ----
        h_good_in = torch.cat([h_state, h_b], dim=-1)     # [B, state_dim + embed_dim]
        h_good    = self.trunk_good(h_good_in)            # [B, good_dim]

        out = {
            # good-standing objects: depend on full state via h_state + h_b
            "c":        0.5*(self.head_c(h_good)-1)+1,
            "q":        self.head_q(h_good),
            "v":        self.head_v(h_good/50),
            "sigma_g":  self.head_sigma(h_good),

            # autarky objects: depend only on (k,s,z) via h_state
            "cw":       0.5*(self.head_cw(h_aut)-1)+1,
            "w":        self.head_w(h_aut/50),
        }
        return out

    # -------- value gradients wrt b,k (used in FOCs and closed-form ι) --------
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
        fb = 0  # index of b
        fk = 1  # index of k

        out = self.forward(states)
        v, w = out["v"], out["w"]

        # grads for v
        grads_v = torch.autograd.grad(
            v,
            states,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        V_b = grads_v[:, fb:fb+1]
        V_k = grads_v[:, fk:fk+1]

        # grads for w（结构上 w 不依赖 b，这里 W_b 应该接近 0，用来 sanity check）
        grads_w = torch.autograd.grad(
            w,
            states,
            grad_outputs=torch.ones_like(w),
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        W_b = grads_w[:, fb:fb+1]
        W_k = grads_w[:, fk:fk+1]

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
