# losses.py
# -----------------------------------------------------------
# Loss construction for the sovereign default BSDE solver.
# - Step 2 uses CLOSED-FORM controls (labor l, investment iota).
# - Targets in the losses DO NOT carry gradients
#   (detach / computed under torch.no_grad()).
# - Only network outputs (c, cw, q, v, w, sigma_g) receive gradients.
# -----------------------------------------------------------

from __future__ import annotations
from typing import Dict, Tuple
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from models.sovereign_net import SovereignNet


# ============== basic economics helpers ==============

def phi_default(z: Tensor, cfg) -> Tensor:
    """Output loss during default: phi(z) = clip(kappa0 + kappa1 * exp(z), 0, 1)."""
    k0 = cfg["params"]["kappa0_phi"]
    k1 = cfg["params"]["kappa1_phi"]
    lo = cfg["params"]["phi_clip_min"]
    hi = cfg["params"]["phi_clip_max"]
    return torch.clamp(k0 + k1 * torch.exp(z), lo, hi)


def prod_y(k: Tensor, l: Tensor, z: Tensor, alpha: float) -> Tensor:
    """Final good production y = exp(z) * k^alpha * l^(1-alpha)."""
    return torch.exp(z) * (k ** alpha) * (l ** (1.0 - alpha))


# -------- utility & marginal utility (exactly as in the paper) --------

def utility_level(c: Tensor, l: Tensor, cfg) -> Tensor:
    """
    GHH 偏好下的效用水平：
        u(c,l) = (1/(1-sigma)) * ( c - eta * l^omega / omega )^{1-sigma}.
    """
    sigma = cfg["params"]["sigma_crra"]
    eta   = cfg["params"]["eta"]
    omega = cfg["params"]["omega"]

    inside_min = cfg["clamp"]["inside_min"]
    inside = c - eta * (l ** omega) / omega
    inside = torch.clamp(inside, min=inside_min)

    if abs(1.0 - sigma) > 1e-12:
        return (inside ** (1.0 - sigma) - 1.0) / (1.0 - sigma)
    else:
        return torch.log(inside)


def uc(c: Tensor, l: Tensor, cfg) -> Tensor:
    """
    对 c 的边际效用：
        u_c(c,l) = ( c - eta * l^omega / omega )^{-sigma}.
    用于 FOC 和 iota 的闭式解。
    """
    sigma = cfg["params"]["sigma_crra"]
    eta   = cfg["params"]["eta"]
    omega = cfg["params"]["omega"]

    inside_min = cfg["clamp"]["inside_min"]
    inside = c - eta * (l ** omega) / omega
    inside = torch.clamp(inside, min=inside_min)

    return inside.pow(-sigma)

# -------- closed-form labor (good / autarky) --------

def labor_good_closed_form(k: Tensor, z: Tensor, cfg) -> Tensor:
    """
    好状态下的劳动闭式解（来自 GHH FOC）：
        (1-alpha) Z_eff k^alpha = eta * l^{omega - 1 + alpha}
        => l* = [ (1-alpha)/eta * Z_eff * k^alpha ]^{ 1 / (omega - 1 + alpha) }.
    """
    alpha = cfg["params"]["alpha"]
    eta   = cfg["params"]["eta"]
    omega = cfg["params"]["omega"]

    Z_eff = torch.exp(z)  # 或者直接用 z，看你产出函数的设定

    expo = 1.0 / (omega - 1.0 + alpha)
    base = ((1.0 - alpha) / eta) * Z_eff * k.pow(alpha)
    l = base.pow(expo)
    return torch.clamp(l, min=cfg["clamp"]["ell_min"], max=1.0)


def labor_autarky_closed_form(k: Tensor, z: Tensor, cfg) -> Tensor:
    """
    Autarky 下的劳动闭式解：
        (1-phi(z))*(1-alpha) Z_eff k^alpha = eta * l^{omega - 1 + alpha}
        => l* = [ (1-phi(z))*(1-alpha)/eta * Z_eff * k^alpha ]^{ 1 / (omega - 1 + alpha) }.
    """
    alpha = cfg["params"]["alpha"]
    eta   = cfg["params"]["eta"]
    omega = cfg["params"]["omega"]

    phi_z = phi_default(z, cfg)
    Z_eff = torch.exp(z)

    expo = 1.0 / (omega - 1.0 + alpha)
    base = (1.0 - phi_z) * (1.0 - alpha) * Z_eff * k.pow(alpha) / eta
    l = base.pow(expo)
    return torch.clamp(l, min=cfg["clamp"]["ell_min"], max=1.0)

def adjust_cost(iota: Tensor, k: Tensor, cfg) -> Tensor:
    """Phi(iota,k) = 0.5 * Theta * (iota - delta * k)^2."""
    theta = cfg["params"]["Theta"]
    delta = cfg["params"]["delta"]
    return 0.5 * theta * (iota - delta * k) ** 2


def iota_closed_form(Vk_or_Wk: Tensor, k: Tensor, c: Tensor, l: Tensor, cfg) -> Tensor:
    """iota* = delta*k + (V_k * k) / (Theta * u_c)."""
    theta = cfg["params"]["Theta"]
    delta = cfg["params"]["delta"]
    uc_val = uc(c, l, cfg)
    iota = delta * k + (Vk_or_Wk * k) / (theta * uc_val + 1e-12)
    return torch.clamp(iota, min=cfg["clamp"]["iota_min"])


def issuance_from_budget(c: Tensor, y: Tensor, phi_adj: Tensor, b: Tensor, q: Tensor, cfg) -> Tensor:
    """
    From budget:
      c + Phi + (lambda+zeta) * b = y + q * i
      =>  i = [c + Phi + (lambda+zeta) * b - y] / q
    """
    lam = cfg["params"]["lambda"]
    zeta = cfg["params"]["zeta"]
    num = c + phi_adj + (lam + zeta) * b - y
    q_safe = torch.clamp(q, min=cfg["clamp"]["q_min"])
    return num / q_safe


# ============== exogenous dynamics (one step) ==============

def mu_z(z: Tensor, cfg) -> Tensor:
    """mu(z) = -phi_z * (z - mu_z)."""
    phi_z = cfg["params"]["phi_z"]
    mu_val = cfg["params"]["mu_z"]
    return -phi_z * (z - mu_val)


def sig_z(z: Tensor, cfg) -> Tuple[Tensor, Tensor]:
    """Return (sigma_z, sigma0_z) as constants by default (broadcastable)."""
    return (torch.as_tensor(cfg["params"]["sigma_z"], device=z.device, dtype=z.dtype),
            torch.as_tensor(cfg["params"]["sigma0_z"], device=z.device, dtype=z.dtype))


def step_s_z(
    s: Tensor, z: Tensor, W: Tensor, W0: Tensor, cfg
) -> Tuple[Tensor, Tensor]:
    """
    One Euler step for s_t (CIR full truncation) and z_t.
    W, W0 ~ N(0, Delta). Shapes: [N, h, 1]
    Returns S_next, Z_next with shape [N, h, 1].
    """
    Delta = cfg["numerics"]["Delta"]
    kappa = cfg["params"]["kappa"]
    s_bar = cfg["params"]["s_bar"]
    sigma_r = cfg["params"]["sigma_r"]

    # Broadcast base states to [N, h, 1]
    N, h = W.shape[0], W.shape[1]
    s0 = s.view(N, 1, 1).expand(N, h, 1)
    z0 = z.view(N, 1, 1).expand(N, h, 1)

    # s: CIR (full truncation Euler)
    sqrt_s = torch.sqrt(torch.clamp(s0, min=0.0))
    S_next = s0 + kappa * (s_bar - s0) * Delta + sigma_r * sqrt_s * W0
    S_next = torch.clamp(S_next, min=cfg["clamp"]["s_min"])

    # z: OU-like with two shocks
    sigma_z, sigma0_z = sig_z(z0, cfg)
    mu_val = mu_z(z0, cfg)
    Z_next = z0 + mu_val * Delta + sigma_z * W + sigma0_z * W0
    Z_next = torch.clamp(Z_next, min=cfg["clamp"]["z_min"], max=cfg["clamp"]["z_max"])
    return S_next, Z_next


# ============== regression helpers ==============

def ridge_batch_solve(X: Tensor, Y: Tensor, ridge: float) -> Tensor:
    """
    Batched ridge regression:
        beta = (X^T X + lambda I)^{-1} X^T Y
    X: [N, h, p], Y: [N, h, 1]
    return beta: [N, p, 1]
    """
    N, h, p = X.shape
    XT = X.transpose(1, 2)                     # [N, p, h]
    XTX = XT @ X                               # [N, p, p]
    eye = torch.eye(p, device=X.device, dtype=X.dtype).unsqueeze(0).expand(N, p, p)
    XTX = XTX + ridge * eye
    XTY = XT @ Y                               # [N, p, 1]
    beta = torch.linalg.solve(XTX, XTY)
    return beta


# ============== main loss builder ==============

def compute_losses(
    net: SovereignNet,
    states: Tensor,          # [N,4] columns: (b,k,s,z)
    cfg,
    generator: torch.Generator | None = None,
) -> Dict[str, Tensor]:
    """
    Build all losses for one batch of starting states.
    Returns a dict with scalars and useful diagnostics.
    """

    device = states.device
    dtype = states.dtype
    N = states.shape[0]

    # ---- unpack constants (ASCII names)
    alpha = cfg["params"]["alpha"]
    delta = cfg["params"]["delta"]
    theta = cfg["params"]["Theta"]
    rho = cfg["params"]["rho"]
    lam = cfg["params"]["lambda"]
    zeta = cfg["params"]["zeta"]
    nu = cfg["params"]["nu"]
    Delta = cfg["numerics"]["Delta"]
    h = cfg["numerics"]["h_paths"]
    ridge = cfg["numerics"]["ridge_reg"]

    # ---- split states
    b, k, s, z = [states[:, i:i+1] for i in range(4)]

    # ---- network outputs at current states
    out_now = net(states)
    c, cw, q, v, w, sigma_g = (
        out_now["c"], out_now["cw"], out_now["q"],
        out_now["v"], out_now["w"], out_now["sigma_g"]
    )

    # value gradients wrt b,k (for controls/FOC)
    grads = net.value_and_grads(states, create_graph=True)
    V_b, V_k, W_k = grads["V_b"], grads["V_k"], grads["W_k"]

    # ---- Step-2: CLOSED-FORM controls
    l   = labor_good_closed_form(k, z, cfg)          # good-standing labor
    l_w = labor_autarky_closed_form(k, z, cfg)       # autarky labor

    iota   = iota_closed_form(V_k, k, c,  l,   cfg)  # good
    iota_w = iota_closed_form(W_k, k, cw, l_w, cfg)  # autarky

    phi_adj   = adjust_cost(iota,   k, cfg)
    phi_adj_w = adjust_cost(iota_w, k, cfg)

    y = prod_y(k, l, z, alpha)
    phi_z_val = phi_default(z, cfg)
    y_w = (1.0 - phi_z_val) * prod_y(k, l_w, z, alpha)

    i   = issuance_from_budget(c,  y,  phi_adj,   b, q, cfg)  # explicit good
    i_w = torch.zeros_like(i)                                 # autarky issues none

    # ---- Budget residuals (diagnostics)
    c_budget  = y  + q * i  - phi_adj   - (lam + zeta) * b
    c_gap     = c  - c_budget
    cw_budget = y_w          - phi_adj_w
    cw_gap    = cw - cw_budget

    # ---- Default indicator (for logs / optional losses)
    default_mask = (v < w).float()

    # ---- Brownian shocks
    std = math.sqrt(Delta)
    W  = torch.randn((N, h, 1), device=device, dtype=dtype, generator=generator) * std
    W0 = torch.randn((N, h, 1), device=device, dtype=dtype, generator=generator) * std

    # ---- Update exogenous states one step
    S_next, Z_next = step_s_z(s, z, W, W0, cfg)

    # ---- Update endogenous states one step (broadcast to h paths)
    B_next = (b + (i - lam * b)   * Delta).view(N, 1, 1).expand(N, h, 1)
    K_next = (k + (iota - delta * k) * Delta).view(N, 1, 1).expand(N, h, 1)

    # ---- Evaluate network on next states (NO GRADIENT -> targets)
    with torch.no_grad():
        X_next = torch.cat([B_next, K_next, S_next, Z_next], dim=2)  # [N,h,4]
        X_next_flat = X_next.reshape(N * h, 4)
        out_next = net(X_next_flat)
        V_next = out_next["v"].reshape(N, h, 1)
        W_next = out_next["w"].reshape(N, h, 1)
        Q_next = out_next["q"].reshape(N, h, 1)

    # ---- Build design matrices for BSDE local regression
    ones_rho    = torch.ones((N, h, 1), device=device, dtype=dtype) * (1.0 + rho * Delta)
    X_rho       = torch.cat([ones_rho, W, W0], dim=2)      # [N,h,3]
    ones_lambda = torch.ones((N, h, 1), device=device, dtype=dtype) * (1.0 + lam * Delta)
    X_lambda    = torch.cat([ones_lambda, W0], dim=2)      # [N,h,2]

    # ---- Instantaneous utility levels (for BSDE), exactly as in the paper
    u_level_g = utility_level(c,  l,   cfg)   # [N,1]
    u_level_w = utility_level(cw, l_w, cfg)   # [N,1]

    ones_h = torch.ones((N, h, 1), device=device, dtype=dtype)
    Yv = V_next + u_level_g.view(N, 1, 1) * Delta * ones_h
    Yw = W_next + u_level_w.view(N, 1, 1) * Delta * ones_h

    # ---- bond price BSDE drift
    # drift_q = (lambda + zeta) - q * (r(s) + nu * sigma_g),
    # where r(s) = s_t here (CIR short rate).
    drift_q = (lam + zeta) - q * (s + sigma_g * nu)
    Yq = Q_next + drift_q.view(N, 1, 1) * Delta * ones_h

    # ---- Local regression to get (v_hat, w_hat, q_hat, sigma_g_hat) -- TARGETS (detached)
    beta_v = ridge_batch_solve(X_rho,    Yv, ridge)   # [N,3,1]
    beta_w = ridge_batch_solve(X_rho,    Yw, ridge)   # [N,3,1]
    beta_q = ridge_batch_solve(X_lambda, Yq, ridge)   # [N,2,1]

    v_hat     = beta_v[:, 0:1, 0].detach()           # [N,1]
    w_hat     = beta_w[:, 0:1, 0].detach()
    q_hat     = beta_q[:, 0:1, 0].detach()
    sigma_hat = beta_q[:, 1:2, 0].detach()

    # ---- BSDE losses (only network outputs carry grad)
    loss_v   = F.mse_loss(v,        v_hat)
    loss_w   = F.mse_loss(w,        w_hat)
    loss_q   = F.mse_loss(q,        q_hat)
    loss_sig = F.mse_loss(sigma_g,  sigma_hat)
    loss_bsde = loss_v + loss_w + loss_q + loss_sig

    # ---- Policy & consistency residuals (diagnostics + small weight)
    # FOC: issuance
    foc_issuance = uc(c, l, cfg) * q + V_b
    # FOC: investment (good/bad)
    phi_iota   = theta * (iota   - delta * k)
    phi_iota_w = theta * (iota_w - delta * k)
    foc_i   = uc(c,  l,   cfg) * phi_iota   - V_k * k
    foc_i_w = uc(cw, l_w, cfg) * phi_iota_w - W_k * k

    loss_policy = (
        (foc_issuance ** 2).mean()
        + (foc_i ** 2).mean()
        + (foc_i_w ** 2).mean()
        + (c_gap ** 2).mean()
        + (cw_gap ** 2).mean()
    )

    # ---- Package results
    # compute useful quantiles and basic stats for network outputs
    def _stats_quantiles(x: Tensor, name: str) -> Dict[str, Tensor]:
        x1 = x.detach().view(-1)
        if x1.numel() == 0:
            return {}
        qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=x1.device, dtype=x1.dtype)
        try:
            qvals = torch.quantile(x1, qs)
        except Exception:
            # fallback: approximate with k-th order (cpu numpy)
            qvals = torch.as_tensor([x1.kthvalue(max(1, int(q * x1.numel()))).values for q in qs], device=x1.device)
        out = {
            f"metrics/{name}_p05": qvals[0].detach(),
            f"metrics/{name}_p25": qvals[1].detach(),
            f"metrics/{name}_p50": qvals[2].detach(),
            f"metrics/{name}_p75": qvals[3].detach(),
            f"metrics/{name}_p95": qvals[4].detach(),
            f"metrics/{name}_mean": x1.mean().detach(),
            f"metrics/{name}_std":  x1.std().detach(),
            f"metrics/{name}_min":  x1.min().detach(),
            f"metrics/{name}_max":  x1.max().detach(),
        }
        return out

    metrics: Dict[str, Tensor] = {}
    metrics.update(_stats_quantiles(c, "c"))
    metrics.update(_stats_quantiles(cw, "cw"))
    metrics.update(_stats_quantiles(q, "q"))
    metrics.update(_stats_quantiles(v, "v"))
    metrics.update(_stats_quantiles(w, "w"))
    metrics.update(_stats_quantiles(sigma_g, "sigma_g"))

    # other diagnostics
    metrics.update({
        "metrics/budget_gap_mean":       c_gap.abs().mean().detach(),
        "metrics/foc_issue_residual_mean": foc_issuance.abs().mean().detach(),
        "metrics/euler_v_residual_mean": (v - v_hat).abs().mean().detach(),
        "metrics/euler_w_residual_mean": (w - w_hat).abs().mean().detach(),
        "metrics/bsde_q_residual_mean":  (q - q_hat).abs().mean().detach(),
        # proportion of states where v < w (default region)
        "metrics/prop_v_lt_w":           default_mask.mean().detach(),
        # quick check for c lower bound
        "metrics/c_min_check":           c.min().detach(),
    })

    losses = {
        "loss_bsde":   loss_bsde,
        "loss_v":      loss_v,
        "loss_w":      loss_w,
        "loss_q":      loss_q,
        "loss_sigma":  loss_sig,
        "loss_policy": loss_policy,
        # overall; caller can combine with weights
        "loss_total": loss_bsde + cfg["train"]["loss_weights"]["policy"] * loss_policy,
    }

    # merge metrics (flat)
    losses.update(metrics)
    return losses
