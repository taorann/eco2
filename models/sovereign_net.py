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
#
# 结构要求：
#   - v, w      共享一套 encoder（对状态的编码），在上层 trunk 不共享；
#   - c, cw     共享另一套 encoder，在上层 trunk 不共享；
#   - q, sigma_g 各自有独立的 encoder + trunk，不与任何其他 head 共享。
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



class TanhBoundedHead(nn.Module):
    """
    Bounded head on (0, max_val) via shifted tanh:
    y = 0.5 * max_val * (tanh(Wx+b) + 1)
    """
    def __init__(self, in_dim: int, out_dim: int = 1,
                 max_val: float = 2.0, scale_in: float = 5.0):
        super().__init__()
        assert out_dim == 1
        self.lin = nn.Linear(in_dim, out_dim)
        self.max_val = float(max_val)
        self.scale_in = float(scale_in)

    def forward(self, x):
        # /scale_in 防止 tanh 太快饱和
        y = torch.tanh(self.lin(x) / self.scale_in)
        return 0.5 * self.max_val * (y + 1.0)


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

    结构设计（按你的要求）：
    - v / w：
        * 共享一套 scalar encoder + state encoder（只用 k,s,z）；
        * 在上层各自有独立的 trunk，再接各自的有界 head_v / head_w。
    - c / cw：
        * 共享另一套 scalar encoder + state encoder（也只用 k,s,z）；
        * 在上层各自有独立的 trunk，再接各自的 head_c / head_cw。
    - q：
        * 独立的 encoder（b,k,s,z）+ trunk + head_q，不与 v/w/c/cw/sigma_g 共享。
    - sigma_g：
        * 独立的 encoder（b,k,s,z）+ trunk + head_sigma，不与其他任何 head 共享。
    """

    def __init__(
        self,
        input_dim: int = 4,                 # [b, k, s, z]
        hidden_sizes: Iterable[int] = (512, 512, 512),
        act: str = "silu",
        dropout: float = 0.0,
        min_positive: float = 1e-6,
        scale_v: float = 50.0,
        scale_w: float = 50.0,
        max_q: float = 1.6,
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
        trunk_out_dim = hidden_sizes[-1]

        # =====================================================
        # 1. Value branch (v, w) — 共享 encoder，独立上层 trunk
        # =====================================================
        # scalar encoders（value 用）
        self.val_enc_b = make_scalar_encoder()
        self.val_enc_k = make_scalar_encoder()
        self.val_enc_s = make_scalar_encoder()
        self.val_enc_z = make_scalar_encoder()

        # 共享的 state encoder：只基于 (k,s,z) 做特征
        self.val_encoder_state = make_trunk(input_dim_trunk=3 * embed_dim)

        # 上层 trunk：v / w 各自一套
        #   v: 依赖 state + b
        #   w: 仅依赖 state（autarky 不直接看 b）
        self.val_trunk_v = make_trunk(input_dim_trunk=trunk_out_dim + embed_dim)
        self.val_trunk_w = make_trunk(input_dim_trunk=trunk_out_dim)

        # 有界 value heads
        self.head_v = BoundedHead(trunk_out_dim, 1, scale=scale_v)
        self.head_w = BoundedHead(trunk_out_dim, 1, scale=scale_w)

        # =====================================================
        # 2. Policy branch (c, cw) — 共享 encoder，独立上层 trunk
        # =====================================================
        self.pol_enc_b = make_scalar_encoder()
        self.pol_enc_k = make_scalar_encoder()
        self.pol_enc_s = make_scalar_encoder()
        self.pol_enc_z = make_scalar_encoder()

        # 共享的 policy state encoder：只用 (k,s,z)
        self.pol_encoder_state = make_trunk(input_dim_trunk=3 * embed_dim)

        # 上层 trunk：c / cw 各自一套
        #   c: 依赖 state + b
        #   cw: 仅依赖 state
        self.pol_trunk_c = make_trunk(input_dim_trunk=trunk_out_dim + embed_dim)
        self.pol_trunk_cw = make_trunk(input_dim_trunk=trunk_out_dim)

        pol_good_dim = trunk_out_dim
        pol_aut_dim = trunk_out_dim
        self.head_c = SigmoidBoundedHead(
            pol_good_dim,
            1,
            max_val=2.0,
            temperature=1.0,
            init_fraction=0.5,
        )
        self.head_cw = SigmoidBoundedHead(
            pol_aut_dim,
            1,
            max_val=max_q,
            temperature=2.0,
            init_fraction=0.5,
        )

        # =====================================================
        # 3. q branch — 完全独立的 encoder + trunk
        # =====================================================
        self.q_enc_b = make_scalar_encoder()
        self.q_enc_k = make_scalar_encoder()
        self.q_enc_s = make_scalar_encoder()
        self.q_enc_z = make_scalar_encoder()

        # q 直接基于 (b,k,s,z) 的 encoder
        self.q_encoder = make_trunk(input_dim_trunk=4 * embed_dim)
        self.head_q = SigmoidBoundedHead(
            trunk_out_dim,
            1,
            max_val=max_q,
            temperature=0.2,
            init_fraction=0.8,
        )

        # =====================================================
        # 4. sigma_g branch — 完全独立的 encoder + trunk
        # =====================================================
        self.sig_enc_b = make_scalar_encoder()
        self.sig_enc_k = make_scalar_encoder()
        self.sig_enc_s = make_scalar_encoder()
        self.sig_enc_z = make_scalar_encoder()

        self.sig_encoder = make_trunk(input_dim_trunk=4 * embed_dim)
        self.head_sigma = PositiveHead(
            trunk_out_dim,
            1,
            min_val=min_positive,
            beta=1.0,
            init_val=0.2,
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

        # ============================
        # 1. Value branch (v, w)
        # ============================
        vb = self.val_enc_b(b)
        vk = self.val_enc_k(k)
        vs = self.val_enc_s(s)
        vz = self.val_enc_z(z)

        h_val_state_in = torch.cat([vk, vs, vz], dim=-1)        # [B,3*embed_dim]
        h_val_state = self.val_encoder_state(h_val_state_in)    # [B,H]

        # v: 依赖 state + b
        h_val_v_in = torch.cat([h_val_state, vb], dim=-1)
        h_val_v = self.val_trunk_v(h_val_v_in)                  # [B,H]

        # w: 仅依赖 state
        h_val_w = self.val_trunk_w(h_val_state)                 # [B,H]

        # 为了稳一点，保持之前的缩放
        v = self.head_v(h_val_v / 50.0)
        w = self.head_w(h_val_w / 50.0)

        # ============================
        # 2. Policy branch (c, cw)
        # ============================
        pb = self.pol_enc_b(b)
        pk = self.pol_enc_k(k)
        ps = self.pol_enc_s(s)
        pz = self.pol_enc_z(z)

        h_pol_state_in = torch.cat([pk, ps, pz], dim=-1)
        h_pol_state = self.pol_encoder_state(h_pol_state_in)

        h_pol_c_in = torch.cat([h_pol_state, pb], dim=-1)
        h_pol_c = self.pol_trunk_c(h_pol_c_in)
        h_pol_cw = self.pol_trunk_cw(h_pol_state)

        c_raw = self.head_c(h_pol_c)
        cw_raw = self.head_cw(h_pol_cw)
        # cw: 映射到 (0.5, 0.5 + 0.5 * max_q)，c 保持在 (0, max_val_c)
        c = 1.5 * (c_raw - 1.0) + 1.0
        cw = 0.5 * cw_raw + 0.5

        # ============================
        # 3. q branch（独立）
        # ============================
        qb = self.q_enc_b(b)
        qk = self.q_enc_k(k)
        qs = self.q_enc_s(s)
        qz = self.q_enc_z(z)

        h_q_in = torch.cat([qb, qk, qs, qz], dim=-1)
        h_q = self.q_encoder(h_q_in)
        q = self.head_q(h_q)

        # ============================
        # 4. sigma_g branch（独立）
        # ============================
        sb = self.sig_enc_b(b)
        sk = self.sig_enc_k(k)
        ss_ = self.sig_enc_s(s)
        sz = self.sig_enc_z(z)

        h_sig_in = torch.cat([sb, sk, ss_, sz], dim=-1)
        h_sig = self.sig_encoder(h_sig_in)
        sigma_g = self.head_sigma(h_sig)

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
        fb = 0  # index of b
        fk = 1  # index of k

        # 不直接在原 states 上改 requires_grad，防止污染外部图
        states_req = states.detach().clone()
        states_req.requires_grad_(True)

        out = self.forward(states_req)
        v, w = out["v"], out["w"]          # [B,1]

        # ---- grads for v wrt (b,k,s,z) ----
        grads_v = torch.autograd.grad(
            v.sum(),                        # 标量，等价于 grad_outputs=ones_like(v)
            states_req,
            retain_graph=True,
            create_graph=create_graph,
        )[0]                                # [B,4]
        V_b = grads_v[:, fb:fb+1]           # [B,1]
        V_k = grads_v[:, fk:fk+1]

        # ---- grads for w wrt (b,k,s,z) ----
        grads_w = torch.autograd.grad(
            w.sum(),
            states_req,
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        W_b = grads_w[:, fb:fb+1]
        W_k = grads_w[:, fk:fk+1]

        return {
            "v":   v,
            "w":   w,
            "V_b": V_b,
            "V_k": V_k,
            "W_b": W_b,
            "W_k": W_k,
        }

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
