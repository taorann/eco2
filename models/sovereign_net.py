# models/sovereign_net.py
# ---------------------------------------------------------
# Multi–head MLP for the quantitative sovereign default model.
# Inputs are state variables x = [b, k, s, z].
# Heads (all scalar):
#   - c      : consumption in good standing
#   - cw     : consumption in autarky
#   - q      : long-term bond price
#   - v      : value in good standing
#   - w      : value in autarky
#   - sigma_g: diffusion loading of q on common shock W0
# ---------------------------------------------------------

from __future__ import annotations
from typing import Dict, Iterable
import math

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
    """
    Strictly positive head via Softplus.

    y = min_val + softplus(Wx+b, beta)

    - min_val 确保数值下界 > 0；
    - beta 取 1，避免 softplus 过陡导致负区梯度直接消失；
    - init_val 用来把初始输出推到一个合理的 σ_0，而不是一上来就粘在 min_val。
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        min_val: float = 0.0,
        beta: float = 1.0,
        init_val: float | None = None,
    ):
        super().__init__()
        assert out_dim == 1
        self.lin = nn.Linear(in_dim, out_dim)
        self.beta = float(beta)
        self.min_val = float(min_val)
        self._init_val = init_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_val + F.softplus(self.lin(x), beta=self.beta)

    def init_parameters(self):
        """
        把权重初始化成「输出接近 init_val」：
          softplus(bias) + min_val ≈ init_val
          ⇒  bias ≈ softplus^{-1}(init_val - min_val) = log(exp(y)-1)
        """
        if self._init_val is None:
            return
        target = max(self._init_val - self.min_val, 1e-10)
        with torch.no_grad():
            # weight 归零，让输出主要由 bias 控制
            self.lin.weight.zero_()
            # softplus^{-1}(y) = log(exp(y) - 1)
            self.lin.bias.fill_(math.log(math.expm1(target)))


class SigmoidBoundedHead(nn.Module):
    """
    Bounded head: 0 < y < max_val via sigmoid。

    y = max_val * sigmoid( Wx + b ) / temperature

    - max_val 控制上界；
    - temperature > 1 可以「拉平」logit，减弱 sigmoid 饱和，避免梯度过快消失；
    - init_fraction 控制初始输出占上界的比例，用来对齐 PDE 期望的标度。
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        max_val: float = 2.0,
        temperature: float = 2.0,
        init_fraction: float | None = 0.5,
    ):
        super().__init__()
        assert out_dim == 1
        self.lin = nn.Linear(in_dim, out_dim)
        self.max_val = float(max_val)
        self.temperature = float(temperature)
        self._init_fraction = init_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.lin(x) / self.temperature
        return self.max_val * torch.sigmoid(logits)

    def init_parameters(self):
        """
        让初始输出接近 max_val * init_fraction：
          sigmoid(bias) ≈ p  ⇒  bias ≈ log( p / (1-p) )
        """
        if self._init_fraction is None:
            return
        p = float(self._init_fraction)
        # 避免 p 太接近 0/1 导致 logit 爆炸
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            self.lin.weight.zero_()
            bias = math.log(p / (1.0 - p))
            self.lin.bias.fill_(bias)


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
        # /10.0 只是缓和一下输入，防止 tanh 太快饱和
        return self.scale * torch.tanh(self.lin(x) / 10.0)


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

    结构设计（关键点：policy 与 value 完全不共享参数）：
    - value 分支：
        * 对 (b,k,s,z) 做 scalar encoder + shared state trunk h_val_state；
        * autarky value 分支 w 只依赖 h_val_state（不看 b）；
        * good-standing 分支 v,q,sigma_g 在 h_val_state 上再接入 h_val_b。
    - policy 分支：
        * 单独的一套 scalar encoder + state trunk h_pol_state；
        * cw 只依赖 h_pol_state（不看 b）；
        * c 在 h_pol_state 上再接入 h_pol_b。
    """

    def __init__(
        self,
        input_dim: int = 4,                 # [b, k, s, z]
        hidden_sizes: Iterable[int] = (256,),
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
        self.max_q = float(max_q)

        # ===== helper builders =====
        def make_scalar_encoder() -> nn.Sequential:
            embed_dim = hidden_sizes[0]
            layers: list[nn.Module] = [nn.Linear(1, embed_dim), _act(act)]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            return nn.Sequential(*layers)

        def make_trunk(input_dim_trunk: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            last = input_dim_trunk
            for h in hidden_sizes:
                layers.append(nn.Linear(last, h))
                layers.append(_act(act))
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
                last = h
            return nn.Sequential(*layers)

        embed_dim = hidden_sizes[0]
        self.embed_dim = embed_dim

        # ============================
        # 1. Value branch (v, w, q, sigma_g)
        # ============================
        # scalar encoders
        self.val_enc_b = make_scalar_encoder()
        self.val_enc_k = make_scalar_encoder()
        self.val_enc_s = make_scalar_encoder()
        self.val_enc_z = make_scalar_encoder()

        # shared state trunk: only (k,s,z)
        self.val_trunk_state = make_trunk(input_dim_trunk=3 * embed_dim)

        # autarky trunk for w
        self.val_trunk_aut = make_trunk(input_dim_trunk=hidden_sizes[-1])

        # good-standing trunk for v,q,sigma_g : [h_state, h_b]
        self.val_trunk_good = make_trunk(input_dim_trunk=hidden_sizes[-1] + embed_dim)

        # value heads
        val_good_dim = hidden_sizes[-1]
        val_aut_dim = hidden_sizes[-1]

        # q: 0 < q < max_q，初始大约在 0.8 * max_q（贴近 PDE 期望的区间），
        # temperature>1 减缓 sigmoid 饱和，避免 q 提不上来。
        self.head_q = SigmoidBoundedHead(
            val_good_dim,
            1,
            max_val=max_q,
            temperature=2.0,
            init_fraction=0.8,
        )
        # sigma_g: > min_positive，初始设置在一个中等值，比如 0.02
        self.head_sigma = PositiveHead(
            val_good_dim,
            1,
            min_val=min_positive,
            beta=1.0,
            init_val=0.02,
        )
        # v, w: 有界值函数
        self.head_v = BoundedHead(val_good_dim, 1, scale=scale_v)
        self.head_w = BoundedHead(val_aut_dim, 1, scale=scale_w)

        # ============================
        # 2. Policy branch (c, cw) – independent from value branch
        # ============================
        self.pol_enc_b = make_scalar_encoder()
        self.pol_enc_k = make_scalar_encoder()
        self.pol_enc_s = make_scalar_encoder()
        self.pol_enc_z = make_scalar_encoder()

        self.pol_trunk_state = make_trunk(input_dim_trunk=3 * embed_dim)
        self.pol_trunk_aut = make_trunk(input_dim_trunk=hidden_sizes[-1])
        self.pol_trunk_good = make_trunk(input_dim_trunk=hidden_sizes[-1] + embed_dim)

        pol_good_dim = hidden_sizes[-1]
        pol_aut_dim = hidden_sizes[-1]
        # c, cw 也用有上界的 sigmoid 头，
        #   c_raw in (0, max_q)，cw_raw in (0, max_q)
        self.head_c = SigmoidBoundedHead(
            pol_good_dim,
            1,
            max_val=max_q,
            temperature=2.0,
            init_fraction=0.5,
        )
        self.head_cw = SigmoidBoundedHead(
            pol_aut_dim,
            1,
            max_val=max_q,
            temperature=2.0,
            init_fraction=0.5,
        )

        # ---------- weight init ----------
        self.apply(_init_weights)
        # 覆盖通用 init，让几个 head 从「有意义的」输出开始
        self.head_q.init_parameters()
        self.head_sigma.init_parameters()
        self.head_c.init_parameters()
        self.head_cw.init_parameters()

    # ----------------------------
    # forward
    # ----------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 4] tensor with columns (b, k, s, z)

        Returns:
            dict with keys: 'c', 'cw', 'q', 'v', 'w', 'sigma_g'
            each value has shape [B, 1]
        """
        assert x.shape[-1] == 4, f"Expected x shape [B,4], got {x.shape}"

        b = x[:, 0:1]  # [B,1]
        k = x[:, 1:2]
        s = x[:, 2:3]
        z = x[:, 3:4]

        # ===== value branch =====
        vb = self.val_enc_b(b)
        vk = self.val_enc_k(k)
        vs = self.val_enc_s(s)
        vz = self.val_enc_z(z)

        h_val_state_in = torch.cat([vk, vs, vz], dim=-1)  # [B,3*embed_dim]
        h_val_state = self.val_trunk_state(h_val_state_in)  # [B, H]

        h_val_aut = self.val_trunk_aut(h_val_state)         # [B, H]
        h_val_good_in = torch.cat([h_val_state, vb], dim=-1)
        h_val_good = self.val_trunk_good(h_val_good_in)     # [B, H]

        q = self.head_q(h_val_good)
        sigma_g = self.head_sigma(h_val_good)
        # 为了稳一点，再把输入缩小一些，避免 tanh 太快饱和
        v = self.head_v(h_val_good / 50.0)
        w = self.head_w(h_val_aut / 50.0)

        # ===== policy branch =====
        pb = self.pol_enc_b(b)
        pk = self.pol_enc_k(k)
        ps = self.pol_enc_s(s)
        pz = self.pol_enc_z(z)

        h_pol_state_in = torch.cat([pk, ps, pz], dim=-1)
        h_pol_state = self.pol_trunk_state(h_pol_state_in)

        h_pol_aut = self.pol_trunk_aut(h_pol_state)
        h_pol_good_in = torch.cat([h_pol_state, pb], dim=-1)
        h_pol_good = self.pol_trunk_good(h_pol_good_in)

        # c_raw, cw_raw in (0, max_q)
        c_raw = self.head_c(h_pol_good)
        cw_raw = self.head_cw(h_pol_aut)
        # cw: 映射到 (0.5, 0.5 + 0.5 * max_q)，c 保持在 (0, max_q)
        c = c_raw
        cw = 0.5 * cw_raw + 0.5

        return {
            "c": c,
            "cw": cw,
            "q": q,
            "v": v,
            "w": w,
            "sigma_g": sigma_g,
        }

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

        # grads for w
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
