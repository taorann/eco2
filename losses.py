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
        return (inside ** (1.0 - sigma) ) / (1.0 - sigma)
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

    对齐文档数值部分：
      - Step 2: closed-form policies (labor, investment, issuance)
      - Step 4: BSDE local regression for (v, w, q, sigma_q)
      - Step 5: Analytic HJB/PDE losses for v, w, q
      - Step 6: Policy approximation loss.

    同时对核心变量/残差/target 记录分位数统计。
    """
    device = states.device
    dtype = states.dtype
    N = states.shape[0]

    # ---- unpack constants
    alpha = cfg["params"]["alpha"]
    delta = cfg["params"]["delta"]
    theta = cfg["params"]["Theta"]
    rho   = cfg["params"]["rho"]
    lam   = cfg["params"]["lambda"]
    zeta  = cfg["params"]["zeta"]
    nu    = cfg["params"]["nu"]
    lambda_b = cfg["params"]["lambda_b"]
    gamma    = cfg["params"]["gamma"]

    Delta = cfg["numerics"]["Delta"]
    h     = cfg["numerics"]["h_paths"]
    ridge = cfg["numerics"]["ridge_reg"]

    # ---- split states
    b, k, s, z = [states[:, i:i+1] for i in range(4)]

    # ---- network outputs at current states (with grad)
    out_now = net(states)
    c, cw, q, v, w, sigma_g = (
        out_now["c"], out_now["cw"], out_now["q"],
        out_now["v"], out_now["w"], out_now["sigma_g"]
    )

    # value gradients wrt b,k
    grads = net.value_and_grads(states, create_graph=True)
    V_b, V_k, W_k = grads["V_b"], grads["V_k"], grads["W_k"]

    # ============================================================
    # Step 2: CLOSED-FORM controls
    # ============================================================
    l   = labor_good_closed_form(k, z, cfg)          # good-standing labor
    l_w = labor_autarky_closed_form(k, z, cfg)       # autarky labor

    iota   = iota_closed_form(V_k, k, c,  l,   cfg)  # good-standing investment
    iota_w = iota_closed_form(W_k, k, cw, l_w, cfg)  # autarky investment

    phi_adj   = adjust_cost(iota,   k, cfg)
    phi_adj_w = adjust_cost(iota_w, k, cfg)

    # output and autarky output
    y         = prod_y(k, l,   z, alpha)
    phi_z_val = phi_default(z, cfg)
    y_w       = (1.0 - phi_z_val) * prod_y(k, l_w, z, alpha)

    # issuance i (explicit from budget); autarky does not issue
    i   = issuance_from_budget(c,  y,  phi_adj,   b, q, cfg)
    i_w = torch.zeros_like(i)

    # ---- Budget residuals (good / autarky)
    c_budget  = y  + q * i  - phi_adj   - (lam + zeta) * b
    c_gap     = c  - c_budget                     # 好状态预算残差

    cw_budget = y_w          - phi_adj_w
    cw_gap    = cw - cw_budget                    # 自给状态预算残差

    # ---- Default indicator for HJB (1{v < w})
    default_mask = (v < w).float()                # 违约区域指示函数

    # ============================================================
    # Brownian shocks & exogenous processes
    # ============================================================
    std = math.sqrt(Delta)
    W  = torch.randn((N, h, 1), device=device, dtype=dtype, generator=generator) * std
    W0 = torch.randn((N, h, 1), device=device, dtype=dtype, generator=generator) * std

    S_next, Z_next = step_s_z(s, z, W, W0, cfg)

    # ---- One-step update of endogenous states (broadcast to h paths)
    B_next = (b + (i - lam * b)      * Delta).view(N, 1, 1).expand(N, h, 1)
    K_next = (k + (iota - delta * k) * Delta).view(N, 1, 1).expand(N, h, 1)

    # ---- Evaluate network on next states (NO GRADIENT -> BSDE / PDE targets)
    with torch.no_grad():
        X_next = torch.cat([B_next, K_next, S_next, Z_next], dim=2)  # [N,h,4]
        out_next = net(X_next.reshape(N * h, 4))
        V_next = out_next["v"].reshape(N, h, 1)
        W_next = out_next["w"].reshape(N, h, 1)
        Q_next = out_next["q"].reshape(N, h, 1)

    # ============================================================
    # Step 4: BSDE local regressions
    # ============================================================
    ones_rho    = torch.ones((N, h, 1), device=device, dtype=dtype) * (1.0 + rho * Delta)
    X_rho       = torch.cat([ones_rho, W, W0], dim=2)       # [N,h,3]
    ones_lambda = torch.ones((N, h, 1), device=device, dtype=dtype) * (1.0 + lam * Delta)
    X_lambda    = torch.cat([ones_lambda, W0], dim=2)       # [N,h,2]

    # Instantaneous utility (good / autarky) for BSDE part
    u_level_g = utility_level(c,  l,   cfg)   # [N,1]
    u_level_w = utility_level(cw, l_w, cfg)   # [N,1]

    ones_h = torch.ones((N, h, 1), device=device, dtype=dtype)
    Yv = V_next + u_level_g.view(N, 1, 1) * Delta * ones_h
    Yw = W_next + u_level_w.view(N, 1, 1) * Delta * ones_h

    # bond price BSDE drift: (lambda+zeta) - q (r + sigma^q nu), with r(s)=s
    drift_q = (lam + zeta) - q * (s + sigma_g * nu)
    Yq = Q_next + drift_q.view(N, 1, 1) * Delta * ones_h

    # Solve ridge regressions, detach coefficients as BSDE targets
    beta_v = ridge_batch_solve(X_rho,    Yv, ridge)   # [N,3,1]
    beta_w = ridge_batch_solve(X_rho,    Yw, ridge)
    beta_q = ridge_batch_solve(X_lambda, Yq, ridge)   # [N,2,1]

    v_hat     = beta_v[:, 0:1, 0].detach()
    w_hat     = beta_w[:, 0:1, 0].detach()
    q_hat     = beta_q[:, 0:1, 0].detach()
    sigma_hat = beta_q[:, 1:2, 0].detach()

    # BSDE losses
    loss_v_bsde   = F.mse_loss(v,       v_hat)
    loss_w_bsde   = F.mse_loss(w,       w_hat)
    loss_q_bsde   = F.mse_loss(q,       q_hat)
    loss_sig_bsde = F.mse_loss(sigma_g, sigma_hat)
    loss_bsde = loss_v_bsde + loss_w_bsde + loss_q_bsde + loss_sig_bsde

    # q 的 BSDE 残差（直接留作监控）
    bsde_q_resid = q - q_hat

    # ============================================================
    # Step 5: Analytic PDE / HJB losses
    # ============================================================
    # D-terms are discrete-time generators:
    #   D(x) = ( A(next_x) - x ) / Delta, where A is average over h paths.
    A_V = V_next.mean(dim=1, keepdim=True)   # [N,1,1]
    A_W = W_next.mean(dim=1, keepdim=True)
    A_Q = Q_next.mean(dim=1, keepdim=True)

    Dv = (A_V - v.view(N, 1, 1)) / Delta
    Dw = (A_W - w.view(N, 1, 1)) / Delta
    Dq = (A_Q - q.view(N, 1, 1)) / Delta

    Dv = Dv.view(N, 1)
    Dw = Dw.view(N, 1)
    Dq = Dq.view(N, 1)

    # value at b=0 for v(0,k,s,z) in Loss_w
    states_b0 = states.clone()
    states_b0[:, 0:1] = 0.0
    out_b0 = net(states_b0)
    v_b0 = out_b0["v"]  # [N,1]

    # Loss w = [ w_n - (u^w_n + D_w + lambda_b * v(0,k,s,z)) / (rho + lambda_b) ]^2
    denom_w   = rho + lambda_b
    target_w  = (u_level_w + Dw + lambda_b * v_b0) / denom_w
    loss_w_pde = ((w - target_w) ** 2).mean()

    # Loss v = [ v_n - (u_n + D_v + gamma*1{v<w}*w_n) / (rho + gamma*1{v<w}) ]^2
    pi = default_mask  # [N,1]
    denom_v  = rho + gamma * pi
    target_v = (u_level_g + Dv + gamma * pi * w) / denom_v
    loss_v_pde = ((v - target_v) ** 2).mean()

    # Loss q = [ q_n - (lambda+zeta + D_q) / (rho + gamma*1{v<w}) ]^2
    denom_q  = rho + gamma * pi
    target_q = (lam + zeta + Dq) / denom_q
    loss_q_pde = ((q - target_q) ** 2).mean()

    loss_pde = loss_w_pde + loss_v_pde + loss_q_pde

    # PDE 残差（真正 HJB 形式，方便看收敛）
    euler_v_res = (rho + gamma * pi) * v - (u_level_g + Dv + gamma * pi * w)
    euler_w_res = (rho + lambda_b) * w - (u_level_w + Dw + lambda_b * v_b0)

    # ============================================================
    # Step 6: Policy approximation loss
    # ============================================================
    # FOC: issuance (good-standing)
    foc_issuance = uc(c, l, cfg) * q + V_b

    # 只保留：
    #   - autarky 下 cw 的预算一致性
    #   - good-standing 下发行 FOC
    loss_policy_cw   = (cw_gap ** 2).mean()           # 自给状态预算残差
    loss_policy_foc  = (foc_issuance ** 2).mean()     # 发行 FOC 残差
    loss_policy      = loss_policy_cw + loss_policy_foc

    # ============================================================
    # Metrics / diagnostics: 分位数 + 直观命名
    # ============================================================
    def _stats_quantiles(x: Tensor, name: str) -> Dict[str, Tensor]:
        """
        返回 x 的分位数统计：p05/p25/p50/p75/p95 + mean/std/min/max
        存储为 metrics/{name}_p05 等。
        """
        x1 = x.detach().view(-1)
        if x1.numel() == 0:
            return {}
        qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=x1.device, dtype=x1.dtype)
        try:
            qvals = torch.quantile(x1, qs)
        except Exception:
            # fallback: approximate；极端情况下用 kthvalue
            qvals = torch.as_tensor(
                [x1.kthvalue(max(1, int(q.item() * x1.numel()))).values for q in qs],
                device=x1.device,
            )
        out = {
            f"metrics/{name}_p05": qvals[0],
            f"metrics/{name}_p25": qvals[1],
            f"metrics/{name}_p50": qvals[2],
            f"metrics/{name}_p75": qvals[3],
            f"metrics/{name}_p95": qvals[4],
            f"metrics/{name}_mean": x1.mean(),
            f"metrics/{name}_std":  x1.std(unbiased=False),
            f"metrics/{name}_min":  x1.min(),
            f"metrics/{name}_max":  x1.max(),
        }
        return {k: v.detach() for k, v in out.items()}

    metrics: Dict[str, Tensor] = {}

    # 1) 核心状态/控制/价值函数
    metrics.update(_stats_quantiles(c,        "c_good"))
    metrics.update(_stats_quantiles(cw,       "c_autarky"))
    metrics.update(_stats_quantiles(q,        "bond_price_q"))
    metrics.update(_stats_quantiles(v,        "value_v"))
    metrics.update(_stats_quantiles(w,        "value_w"))
    metrics.update(_stats_quantiles(sigma_g,  "bond_vol_sigma_g"))
    metrics.update(_stats_quantiles(l,        "labor_good"))
    metrics.update(_stats_quantiles(l_w,      "labor_autarky"))
    metrics.update(_stats_quantiles(iota,     "investment_good"))
    metrics.update(_stats_quantiles(iota_w,   "investment_autarky"))
    metrics.update(_stats_quantiles(i,        "issuance_good"))
    metrics.update(_stats_quantiles(y,        "output_good"))
    metrics.update(_stats_quantiles(y_w,      "output_autarky"))

    # 2) BSDE 目标
    metrics.update(_stats_quantiles(v_hat,     "bsde_target_v"))
    metrics.update(_stats_quantiles(w_hat,     "bsde_target_w"))
    metrics.update(_stats_quantiles(q_hat,     "bsde_target_q"))
    metrics.update(_stats_quantiles(sigma_hat, "bsde_target_sigma_q"))

    # 3) PDE 目标
    metrics.update(_stats_quantiles(target_v,  "pde_target_v"))
    metrics.update(_stats_quantiles(target_w,  "pde_target_w"))
    metrics.update(_stats_quantiles(target_q,  "pde_target_q"))

    # 4) D-terms
    metrics.update(_stats_quantiles(Dv,        "generator_Dv"))
    metrics.update(_stats_quantiles(Dw,        "generator_Dw"))
    metrics.update(_stats_quantiles(Dq,        "generator_Dq"))

    # 5) 各类残差（使用绝对值更直观）
    metrics.update(_stats_quantiles(c_gap.abs(),         "budget_gap_good_abs"))
    metrics.update(_stats_quantiles(cw_gap.abs(),        "budget_gap_autarky_abs"))
    metrics.update(_stats_quantiles(foc_issuance.abs(),  "foc_issuance_abs"))
    metrics.update(_stats_quantiles(euler_v_res.abs(),   "euler_v_resid_abs"))
    metrics.update(_stats_quantiles(euler_w_res.abs(),   "euler_w_resid_abs"))
    metrics.update(_stats_quantiles(bsde_q_resid.abs(),  "bsde_q_resid_abs"))

    # 6) 一些直观的汇总指标
    metrics.update({
        # v<w 区域占比：违约区域 share
        "metrics/share_default_region": default_mask.mean().detach(),
        # 消费最小值（快速 sanity check）
        "metrics/min_c_good":          c.min().detach(),
        "metrics/min_c_autarky":       cw.min().detach(),
    })

    # ============================================================
    # Combine losses with configurable weights
    # ============================================================
    w_bsde   = cfg["train"]["loss_weights"]["bsde"]
    w_pde    = cfg["train"]["loss_weights"]["pde"]
    w_policy = cfg["train"]["loss_weights"]["policy"]

    loss_total = (
        w_bsde   * loss_bsde
        + w_pde  * loss_pde
        + w_policy * loss_policy
    )

    losses: Dict[str, Tensor] = {
        "loss_bsde":        loss_bsde,
        "loss_v_bsde":      loss_v_bsde,
        "loss_w_bsde":      loss_w_bsde,
        "loss_q_bsde":      loss_q_bsde,
        "loss_sigma_bsde":  loss_sig_bsde,
        "loss_pde":         loss_pde,
        "loss_w_pde":       loss_w_pde,
        "loss_v_pde":       loss_v_pde,
        "loss_q_pde":       loss_q_pde,
        "loss_policy":      loss_policy,
        "loss_policy_cw":   loss_policy_cw,
        "loss_policy_foc":  loss_policy_foc,
        "loss_total":       loss_total,
    }



    # ---- 为旧版 monitor 提供兼容的 loss 名称（log_step 在用）----
    losses["loss_v"]     = loss_v_bsde
    losses["loss_w"]     = loss_w_bsde
    losses["loss_q"]     = loss_q_bsde
    losses["loss_sigma"] = loss_sig_bsde

    # 先合并新的 metrics
    losses.update(metrics)

    # ---- 为 config.monitor.tracked_metrics 中的旧 key 提供别名 ----
    # 现在 default.yaml 里 tracked_metrics 是：
    #   q_mean, q_min, budget_gap_mean, foc_issue_residual_mean,
    #   euler_v_residual_mean, euler_w_residual_mean,
    #   bsde_q_residual_mean, share_default_region, c_min_check, k_neg_guard_hits
    # 其中最后一个原来就没实现，这里先不管。
    def _alias(src_key: str, dst_key: str):
        if src_key in losses and dst_key not in losses:
            losses[dst_key] = losses[src_key]

    _alias("metrics/bond_price_q_mean",        "metrics/q_mean")
    _alias("metrics/bond_price_q_min",         "metrics/q_min")
    _alias("metrics/budget_gap_good_abs_mean", "metrics/budget_gap_mean")
    _alias("metrics/foc_issuance_abs_mean",    "metrics/foc_issue_residual_mean")
    _alias("metrics/euler_v_resid_abs_mean",   "metrics/euler_v_residual_mean")
    _alias("metrics/euler_w_resid_abs_mean",   "metrics/euler_w_residual_mean")
    _alias("metrics/bsde_q_resid_abs_mean",    "metrics/bsde_q_residual_mean")
    _alias("metrics/min_c_good",               "metrics/c_min_check")
    # metrics/share_default_region 名字本身就沿用原来的

    return losses
