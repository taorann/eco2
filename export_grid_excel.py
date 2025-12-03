# export_grid_excel.py
# ---------------------------------------------------------
# 在固定的 (s, z) 格点上，把网络的稳态输出与闭式控制
# 按 (b, k) 网格整理成 Excel，多 sheet 导出。
#
# 每个 (s, z) 组合对应一个 sheet，一共 5x5=25 个 sheet：
#   - s 以 s_bar 为中心，步长 0.01，共 5 个点；
#   - z 在 [mu_z - 2 sigma_z, mu_z + 2 sigma_z] 上，以 1 个 sigma_z 为步长，共 5 个点。
#
# 在每个 sheet 中：
#   - 顶部先写 s、z 与 k_grid；
#   - w_autarky 只写一行（与 b 无关）；
#   - v(b,k) 为完整的 b×k 矩阵；
#   - cw_autarky 一行（与 b 无关），c(b,k) 矩阵；
#   - q(b,k)、sigma_q(b,k) 矩阵；
#   - iota_good(b,k) 矩阵；
#   - iota_autarky 一行（与 b 无关）；
#   - issuance_good(b,k) 矩阵；
#   其中：
#     - 行方向是 b，列方向是 k；
#     - “坐标变量和坐标值”都在表里写清楚。
# ---------------------------------------------------------

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from losses import (
    labor_good_closed_form,
    labor_autarky_closed_form,
    adjust_cost,
    iota_closed_form,
    issuance_from_budget,
    prod_y,
)


# ---------------------------------------------------------
# 网格构造
# ---------------------------------------------------------
def _build_bk_grid(
    cfg: dict,
    b_step: float = 0.05,
    k_step: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 cfg['state_box'] 里读取 b,k 的范围，构造等距网格。
    """
    box = cfg["state_box"]
    b_min, b_max = box["b"]
    k_min, k_max = box["k"]

    b_vals = np.arange(b_min, b_max + 1e-8, b_step, dtype=float)
    k_vals = np.arange(k_min, k_max + 1e-8, k_step, dtype=float)
    return b_vals, k_vals


def _build_sz_grid(cfg: dict, s_step: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造 s,z 的 5×5 格点：
      - s: 以 s_bar 为中心，步长 s_step，共 5 个点；
      - z: [mu_z - 2 sigma_z, ..., mu_z + 2 sigma_z]，步长为 sigma_z，共 5 个点。
    """
    params = cfg["params"]
    s_bar = float(params["s_bar"])
    mu_z = float(params["mu_z"])
    sigma_z = float(params["sigma_z"])

    s_vals = np.array([s_bar + i * s_step for i in range(-2, 3)], dtype=float)
    z_vals = np.array([mu_z + i * sigma_z for i in range(-2, 3)], dtype=float)
    return s_vals, z_vals


# ---------------------------------------------------------
# 单个 (s,z) 下的 (b,k) 网格计算
# ---------------------------------------------------------
def _compute_grid_for_one(
    net: torch.nn.Module,
    cfg: dict,
    b_vals: np.ndarray,
    k_vals: np.ndarray,
    s_val: float,
    z_val: float,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, np.ndarray]:
    """
    在给定的 (s,z) 下：
      - 构造 (b,k) 网格；
      - 计算网络输出 (c,cw,q,v,w,sigma_g)；
      - 利用当前网络的 V_k, W_k 与闭式公式，算出 iota_good, iota_aut, issuance_good。
    返回的每个数组都是形状 [B,K] 的 numpy 数组。
    """
    if device is None:
        device = next(net.parameters()).device
    else:
        device = torch.device(device)

    net = net.to(device)
    net.eval()

    B = len(b_vals)
    K = len(k_vals)

    # [B], [K]
    b_t = torch.tensor(b_vals, device=device, dtype=dtype)
    k_t = torch.tensor(k_vals, device=device, dtype=dtype)

    # [B,K] 网格
    bb, kk = torch.meshgrid(b_t, k_t, indexing="ij")
    ss = torch.full_like(bb, float(s_val))
    zz = torch.full_like(bb, float(z_val))

    # [B*K,4]
    states = torch.stack([bb, kk, ss, zz], dim=-1).reshape(-1, 4)

    # --- 1) 网络输出 ---
    out = net(states)
    c = out["c"]        # 好状态消费
    cw = out["cw"]      # 自给状态消费
    q = out["q"]
    v = out["v"]
    w = out["w"]
    sigma_g = out["sigma_g"]

    # --- 2) 值函数对 k 的梯度（用于 iota 闭式解） ---
    grads = net.value_and_grads(states, create_graph=False)
    V_k = grads["V_k"]   # [BK,1]
    W_k = grads["W_k"]

    # --- 3) 闭式 labor, iota, issuance ---
    b = states[:, 0:1]   # [BK,1]
    k = states[:, 1:1+1]
    z = states[:, 3:3+1]

    l_good = labor_good_closed_form(k, z, cfg)        # 好状态劳动
    l_aut  = labor_autarky_closed_form(k, z, cfg)     # 自给状态劳动

    iota_good = iota_closed_form(V_k, k, c,  l_good, cfg)   # 好状态投资
    iota_aut  = iota_closed_form(W_k, k, cw, l_aut,  cfg)   # 自给状态投资

    phi_adj = adjust_cost(iota_good, k, cfg)

    alpha = float(cfg["params"]["alpha"])
    y_good = prod_y(k, l_good, z, alpha)

    # 这里我们导出的是 “good 状态” 下的发行，对应 π=0
    pi_zero = torch.zeros_like(b)
    issuance_good = issuance_from_budget(c, y_good, phi_adj, b, q, pi_zero, cfg)

    def _reshape(x: Tensor) -> np.ndarray:
        return x.view(B, K).detach().cpu().numpy()

    arrs = {
        "v": _reshape(v),
        "w": _reshape(w),
        "c": _reshape(c),
        "cw": _reshape(cw),
        "q": _reshape(q),
        "sigma_g": _reshape(sigma_g),
        "iota_good": _reshape(iota_good),
        "iota_aut": _reshape(iota_aut),
        "issuance_good": _reshape(issuance_good),
    }
    return arrs


# ---------------------------------------------------------
# 把一个 (s,z) 的所有量组织成一个 DataFrame
# ---------------------------------------------------------
def _make_sheet_dataframe(
    b_vals: np.ndarray,
    k_vals: np.ndarray,
    s_val: float,
    z_val: float,
    arrs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    按你要求的格式组织成一个表格：
      - 顶部写 s, z, k_grid；
      - w_autarky 一行，后接 v(b,k)；
      - cw_autarky 一行，后接 c(b,k)；
      - q(b,k)、sigma_q(b,k)、iota_good(b,k)、iota_autarky 一行、issuance_good(b,k)。
    第一列统一叫 coord_or_b，既可以放坐标标签，也可以放 b。
    """
    B = len(b_vals)
    K = len(k_vals)

    cols = ["coord_or_b"] + [f"{k:.3f}" for k in k_vals]
    rows: list[list[object]] = []

    # ---- 坐标信息 ----
    rows.append(["s", float(s_val)] + [""] * (K - 1))
    rows.append(["z", float(z_val)] + [""] * (K - 1))
    rows.append(["", *[""] * K])

    # 显式写出 k 网格
    rows.append(["k_grid"] + [float(k) for k in k_vals])

    # ---------- 1. w & v ----------
    w_mat = arrs["w"]       # [B,K]，理论上不随 b 变化
    v_mat = arrs["v"]

    w_row = w_mat[0, :]     # 取第一行作为 w(s,z,k)
    rows.append(["w_autarky"] + [float(x) for x in w_row])
    rows.append(["", *[""] * K])

    rows.append(["v(b,k)", *[""] * K])
    rows.append(["b\\k"] + [float(k) for k in k_vals])
    for i, b in enumerate(b_vals):
        rows.append([float(b)] + [float(x) for x in v_mat[i, :]])

    # ---------- 2. cw & c ----------
    rows.append(["", *[""] * K])
    cw_mat = arrs["cw"]
    c_mat = arrs["c"]

    cw_row = cw_mat[0, :]   # cw 与 b 无关，只写一行
    rows.append(["cw_autarky"] + [float(x) for x in cw_row])
    rows.append(["", *[""] * K])

    rows.append(["c(b,k)", *[""] * K])
    rows.append(["b\\k"] + [float(k) for k in k_vals])
    for i, b in enumerate(b_vals):
        rows.append([float(b)] + [float(x) for x in c_mat[i, :]])

    # ---------- 3. q ----------
    rows.append(["", *[""] * K])
    q_mat = arrs["q"]
    rows.append(["q(b,k)", *[""] * K])
    rows.append(["b\\k"] + [float(k) for k in k_vals])
    for i, b in enumerate(b_vals):
        rows.append([float(b)] + [float(x) for x in q_mat[i, :]])

    # ---------- 4. sigma_q (= sigma_g) ----------
    rows.append(["", *[""] * K])
    sigma_mat = arrs["sigma_g"]
    rows.append(["sigma_q(b,k)", *[""] * K])
    rows.append(["b\\k"] + [float(k) for k in k_vals])
    for i, b in enumerate(b_vals):
        rows.append([float(b)] + [float(x) for x in sigma_mat[i, :]])

    # ---------- 5. iota_good ----------
    rows.append(["", *[""] * K])
    iota_good = arrs["iota_good"]
    rows.append(["iota_good(b,k)", *[""] * K])
    rows.append(["b\\k"] + [float(k) for k in k_vals])
    for i, b in enumerate(b_vals):
        rows.append([float(b)] + [float(x) for x in iota_good[i, :]])

    # ---------- 6. iota_autarky（只一行，与 b 无关） ----------
    rows.append(["", *[""] * K])
    iota_aut = arrs["iota_aut"]
    iota_aut_row = iota_aut[0, :]
    rows.append(["iota_autarky"] + [float(x) for x in iota_aut_row])

    # ---------- 7. issuance_good ----------
    rows.append(["", *[""] * K])
    issuance = arrs["issuance_good"]
    rows.append(["issuance_good(b,k)", *[""] * K])
    rows.append(["b\\k"] + [float(k) for k in k_vals])
    for i, b in enumerate(b_vals):
        rows.append([float(b)] + [float(x) for x in issuance[i, :]])

    df = pd.DataFrame(rows, columns=cols)
    return df


# ---------------------------------------------------------
# 对外主函数：导出整个 (s,z)×(b,k) 网格到 Excel
# ---------------------------------------------------------
def export_state_grid_excel(
    net: torch.nn.Module,
    cfg: dict,
    out_path: str,
    b_step: float = 0.05,
    k_step: float = 0.05,
    s_step: float = 0.01,
) -> None:
    """
    主入口：在训练完成后调用一次即可。

    Args
    ----
    net      : 训练好的 SovereignNet（或者 DataParallel 包裹的都可以）。
    cfg      : 你的 YAML 配置（load_config 之后的 dict）。
    out_path : Excel 文件输出路径，比如 "./runs/exp1/state_grid.xlsx"。
    """
    # 统一转 device
    device = next(net.parameters()).device

    b_vals, k_vals = _build_bk_grid(cfg, b_step=b_step, k_step=k_step)
    s_vals, z_vals = _build_sz_grid(cfg, s_step=s_step)

    # 25 个 sheet，对应 5×5 个 (s,z) 组合
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for s_val in s_vals:
            for z_val in z_vals:
                arrs = _compute_grid_for_one(
                    net=net,
                    cfg=cfg,
                    b_vals=b_vals,
                    k_vals=k_vals,
                    s_val=float(s_val),
                    z_val=float(z_val),
                    device=device,
                )

                df = _make_sheet_dataframe(
                    b_vals=b_vals,
                    k_vals=k_vals,
                    s_val=float(s_val),
                    z_val=float(z_val),
                    arrs=arrs,
                )

                # sheet 名：例如 "s0.030_z-0.150"，保证长度 <= 31
                sheet_name = f"s{float(s_val):.3f}_z{float(z_val):.3f}"
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]

                df.to_excel(writer, sheet_name=sheet_name, index=False)
