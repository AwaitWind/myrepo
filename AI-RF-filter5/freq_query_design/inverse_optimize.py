"""
Layer 2A —— 逆向求解（CMA-ES + 可微梯度，M3）。

【本版重构核心改动】：
  1. 目标函数纳入「下降系数」：
       - 固定 target 模式：MSE + rolloff_penalty（skirt 哨兵硬约束）
       - 自由通带模式    ：passband_free_loss 内置阻带缺额 + rolloff 哨兵
  2. 支持 Ensemble 代理：
       - 用成员均值预测（降低单模型抖动）
       - std_penalty > 0 时把成员间 std 加到 objective（避免走到代理不确定的参数区）
  3. 目标"保守余量" margin_db：给通带/阻带门槛加上代理误差余量，反解出的解在真值上更稳
  4. Top-K 输出：CMA-ES 在所有 restart 里保留 K 个不重合的最优解 → validate 一次跑完 HFSS
  5. CMA-ES restarts 4→16，sigma0 0.25→0.35

【约束适配】_constraint_penalty 严格对照本电路 id_config.check_constraints
  （19 维：l1-l5 / w1-w4 / wc1-wc5 / h1-h3 / hd / hs，
   谐振器 x 越界 / via 与中心导体重叠 / via 阵列 y 长度）。

用法（推荐）:
    # 固定带通目标
    python inverse_optimize.py --f0 3.5 --bw 0.5 --passband_db -3 \\
           --stopband_db -25 --rolloff_ghz 0.15 --rolloff_db -15 \\
           --margin_db 1.0 --top_k 10 --restarts 16 \\
           --out id_data/inv_result.npz

    # 自由通带（不指定中心频率）
    python inverse_optimize.py --free --target_bw 0.5 --passband_db -3 \\
           --stopband_db -20 --rolloff_ghz 0.15 --rolloff_db -15 \\
           --top_k 10 --restarts 16 --out id_data/inv_free.npz
"""

import argparse
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import id_config as C
from forward_net import ForwardModel, FORWARD_CKPT
from dataio import split_y
import spec as spec_mod
from passband_spec import (
    passband_free_loss,
    passband_free_loss_torch,
    find_best_passband,
    format_passband_report,
)


# =====================================================================
# 目标函数组件
# =====================================================================
def _weighted_mse(y_pred, y_target, w):
    diff = (y_pred - y_target) * np.sqrt(np.maximum(w, 0.0))
    return float(np.mean(diff ** 2))


def _constraint_penalty(x_real, scale=1000.0):
    """
    几何约束线性罚，严格对照 id_config.check_constraints。

    本版参数已减维到 11 维（wc1=wc5=wc_end / wc2=wc3=wc4=wc_res / l1=l5=l_end /
    h1=h2=h3=h_res / l2=l3=l4=l_res）。这里先用 _expand_to_full 展开成 19 参数字典，
    再按原 filter5 的 4 类约束逐条检查（约束逻辑与 id_config 保持完全一致）：
      A) 全部参数 > 0
      B) 3 对谐振器 x 不越界：wc2/2+h1, wc3/2+h2, wc4/2+h3 <= PCB_X/2 - 0.2
      C) via 阵列不覆盖中心导体：pcb_x/2 - hd/2 > max(中心结构最外 x) + 0.1
      D) via 阵列 y 长度合法：l2+l3+l4+w1+w2+w3+w4 >= hd
    """
    full = C._expand_to_full(x_real)
    pen = 0.0

    # A) 全部 > 0
    for _name, v in full.items():
        if v <= 0:
            pen += (-v + 1e-6)

    # B) 3 对谐振器 x 越界（对称约束下三条其实等价，但保留循环便于扩展）
    x_limit = C.PCB_X / 2.0 - 0.2
    for wc_key, h_key in (("wc2", "h1"), ("wc3", "h2"), ("wc4", "h3")):
        x_used = full[wc_key] / 2.0 + full[h_key]
        if x_used > x_limit:
            pen += (x_used - x_limit)

    # C) via 阵列不能覆盖到中心导体
    via_outer_x = C.PCB_X / 2.0 - full["hd"] / 2.0
    max_center_x = max(
        full["wc1"] / 2.0, full["wc5"] / 2.0,
        full["wc2"] / 2.0 + full["h1"],
        full["wc3"] / 2.0 + full["h2"],
        full["wc4"] / 2.0 + full["h3"],
    )
    over_via = (max_center_x + 0.1) - via_outer_x
    if over_via > 0:
        pen += over_via

    # D) via 阵列 y 长度
    via_line_len = (full["l2"] + full["l3"] + full["l4"]
                    + full["w1"] + full["w2"] + full["w3"] + full["w4"])
    if via_line_len < full["hd"]:
        pen += (full["hd"] - via_line_len)

    return scale * pen


def _std_penalty(y_std, weight=0.0):
    """代理不确定度惩罚：ensemble std 越大越罚。y_std 长度 Y_DIM 或 (B,Y_DIM)。"""
    if weight <= 0 or y_std is None:
        return 0.0
    return float(weight * np.mean(np.abs(y_std)))


# =====================================================================
# 固定 target 目标函数
# =====================================================================
def make_objective(fm, y_target, w,
                   rolloff=None, std_weight=0.0):
    """
    rolloff = None 或 dict(f0=..., bw=..., rolloff_ghz=..., rolloff_db=..., weight=...)
    std_weight > 0 且 fm.is_ensemble 时启用不确定度惩罚
    """
    y_target = np.asarray(y_target, float)
    w = np.asarray(w, float)
    use_std = std_weight > 0 and fm.is_ensemble

    def f(x_real):
        x_real = np.clip(np.asarray(x_real, float), C.LOWER, C.UPPER)
        if use_std:
            y_mean, y_std = fm.predict(x_real, return_std=True)
            y_pred = y_mean[0]
            cost = _weighted_mse(y_pred, y_target, w)
            cost += _std_penalty(y_std[0], std_weight)
        else:
            y_pred = fm.predict(x_real)[0]
            cost = _weighted_mse(y_pred, y_target, w)

        if rolloff is not None:
            s21, _ = split_y(y_pred)
            cost += spec_mod.rolloff_penalty(
                s21, rolloff["f0"], rolloff["bw"],
                rolloff_ghz=rolloff["rolloff_ghz"],
                rolloff_db=rolloff["rolloff_db"],
                weight=rolloff.get("weight", 2.0),
            )

        cost += _constraint_penalty(x_real)
        return cost

    return f


# =====================================================================
# 自由通带目标函数
# =====================================================================
def make_free_objective(fm, target_bw, passband_db=-3.0,
                        stopband_db=None, guard_ghz=0.3, stop_weight=1.0,
                        rolloff_db=None, rolloff_ghz=0.15, rolloff_weight=1.5,
                        agg="mean", std_weight=0.0, f0=None, f0_tol=None):
    """f0/f0_tol 给定时即『软约束中心频率』(centered)：只在 f0±f0_tol 邻域找通带。"""
    use_std = std_weight > 0 and fm.is_ensemble

    def f(x_real):
        x_real = np.clip(np.asarray(x_real, float), C.LOWER, C.UPPER)
        if use_std:
            y_mean, y_std = fm.predict(x_real, return_std=True)
            y_pred = y_mean[0]
            std_part = _std_penalty(y_std[0], std_weight)
        else:
            y_pred = fm.predict(x_real)[0]
            std_part = 0.0

        s21, _ = split_y(y_pred)
        return (
            passband_free_loss(
                s21, target_bw, passband_db=passband_db,
                stopband_db=stopband_db, guard_ghz=guard_ghz, stop_weight=stop_weight,
                rolloff_db=rolloff_db, rolloff_ghz=rolloff_ghz, rolloff_weight=rolloff_weight,
                agg=agg, f0=f0, f0_tol=f0_tol,
            )
            + std_part
            + _constraint_penalty(x_real)
        )
    return f


# =====================================================================
# Top-K 收集器（去重：参数距离 < min_dist_unit 视为同一候选）
# =====================================================================
class TopKCollector:
    def __init__(self, k=10, min_dist_unit=0.05):
        self.k = k
        self.min_dist = min_dist_unit
        self.items = []       # list[(f_val, x_real, u01)]

    def _too_close(self, u, existing):
        return any(np.linalg.norm(u - eu) < self.min_dist for _, _, eu in existing)

    def add(self, f_val, x_real):
        u = (x_real - C.LOWER) / (C.UPPER - C.LOWER + 1e-12)
        if self._too_close(u, self.items):
            for i, (fi, _, ui) in enumerate(self.items):
                if np.linalg.norm(u - ui) < self.min_dist and f_val < fi:
                    self.items[i] = (f_val, x_real.copy(), u.copy())
                    break
        else:
            self.items.append((f_val, x_real.copy(), u.copy()))
        self.items.sort(key=lambda t: t[0])
        if len(self.items) > self.k:
            self.items = self.items[:self.k]

    def as_arrays(self):
        if not self.items:
            return None
        fs = np.array([t[0] for t in self.items], dtype=float)
        xs = np.stack([t[1] for t in self.items], axis=0)
        return fs, xs


# =====================================================================
# CMA-ES 求解（返回 Top-K）
# =====================================================================
def solve_cma(fm, obj_real,
              n_restarts=16, sigma0=0.35, maxiter=200, seed=42,
              top_k=10, verbose=True):
    def obj_unit(u):
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        x_real = C.LOWER + u * (C.UPPER - C.LOWER)
        return obj_real(x_real)

    collector = TopKCollector(k=top_k)

    try:
        import cma
        for r in range(n_restarts):
            x0 = np.random.default_rng(seed + r).random(C.N_PARAMS)
            es = cma.CMAEvolutionStrategy(
                x0, sigma0,
                {"bounds": [0.0, 1.0], "maxiter": maxiter, "verbose": -9, "seed": seed + r},
            )
            es.optimize(obj_unit)
            best_u = np.array(es.result.xbest, float)
            best_x = C.LOWER + np.clip(best_u, 0, 1) * (C.UPPER - C.LOWER)
            collector.add(float(es.result.fbest), best_x)
            if verbose:
                print(f"  [cma] restart {r+1:2d}/{n_restarts}  fbest={es.result.fbest:.5f}  "
                      f"top1={collector.items[0][0]:.5f}  |topK|={len(collector.items)}")
    except ImportError:
        if verbose:
            print("  [cma] 未安装 cma 库，回退到简单 ES。建议 pip install cma")
        _fallback_es(obj_unit, n_restarts, maxiter, seed, verbose, collector)

    return collector


def _fallback_es(obj_unit, n_restarts, maxiter, seed, verbose, collector):
    rng = np.random.default_rng(seed)
    pop = 16
    for r in range(n_restarts):
        mean = rng.random(C.N_PARAMS)
        sigma = 0.35
        cur_best_u, cur_best_f = mean.copy(), obj_unit(mean)
        for it in range(maxiter):
            cand = np.clip(mean + sigma * rng.standard_normal((pop, C.N_PARAMS)), 0, 1)
            fs = np.array([obj_unit(c) for c in cand])
            order = np.argsort(fs)
            elites = cand[order[:max(2, pop // 4)]]
            mean = elites.mean(0)
            if fs[order[0]] < cur_best_f:
                cur_best_f, cur_best_u = float(fs[order[0]]), cand[order[0]].copy()
            sigma *= 0.97
        best_x = C.LOWER + cur_best_u * (C.UPPER - C.LOWER)
        collector.add(cur_best_f, best_x)
        if verbose:
            print(f"  [es]  restart {r+1:2d}/{n_restarts}  fbest={cur_best_f:.5f}")


# =====================================================================
# 梯度求解（可微代理，[0,1] 空间 + clamp）
# =====================================================================
def solve_grad(fm, y_target, w, obj_real,
               n_restarts=16, n_steps=400, lr=0.05, seed=42,
               top_k=10, verbose=True,
               free=None, fixed_rolloff=None, std_weight=0.0):
    import torch
    is_free = free is not None

    if not is_free:
        yt_np = (np.asarray(y_target) - fm.y_mu_np) / fm.y_std_np
        yt = torch.tensor(yt_np, dtype=torch.float32, device=fm.device)
        wt = torch.tensor(np.asarray(w), dtype=torch.float32, device=fm.device)

    collector = TopKCollector(k=top_k)
    rng = np.random.default_rng(seed)
    use_std = std_weight > 0 and fm.is_ensemble

    y_mu_t = torch.tensor(fm.y_mu_np, dtype=torch.float32, device=fm.device)
    y_std_t = torch.tensor(fm.y_std_np, dtype=torch.float32, device=fm.device)

    for r in range(n_restarts):
        u = torch.tensor(rng.random(C.N_PARAMS), dtype=torch.float32,
                         device=fm.device, requires_grad=True)
        opt = torch.optim.Adam([u], lr=lr)
        for step in range(n_steps):
            uc = torch.clamp(u, 0.0, 1.0)
            if use_std:
                all_pred = fm.forward_unit_all(uc)          # (M,1,Y)
                yn = all_pred.mean(dim=0).squeeze(0)
                std_pen = all_pred.std(dim=0).abs().mean()
            else:
                yn = fm.forward_unit(uc).squeeze(0)
                std_pen = torch.tensor(0.0, device=fm.device)

            if is_free:
                y_db = yn * y_std_t + y_mu_t
                s21 = y_db[:C.N_FREQ]
                loss = passband_free_loss_torch(
                    s21, free["target_bw"], free["passband_db"],
                    stopband_db=free.get("stopband_db"),
                    guard_ghz=free.get("guard_ghz", 0.3),
                    stop_weight=free.get("stop_weight", 1.0),
                    rolloff_db=free.get("rolloff_db"),
                    rolloff_ghz=free.get("rolloff_ghz", 0.15),
                    rolloff_weight=free.get("rolloff_weight", 1.5),
                    tau=free.get("tau", 0.1),
                    f0=free.get("f0"), f0_tol=free.get("f0_tol"),
                )
            else:
                loss = (((yn - yt) ** 2) * wt).mean()
                if fixed_rolloff is not None:
                    y_db = yn * y_std_t + y_mu_t
                    s21 = y_db[:C.N_FREQ]
                    loss = loss + spec_mod.rolloff_penalty_torch(
                        s21, fixed_rolloff["f0"], fixed_rolloff["bw"],
                        rolloff_ghz=fixed_rolloff["rolloff_ghz"],
                        rolloff_db=fixed_rolloff["rolloff_db"],
                        weight=fixed_rolloff.get("weight", 2.0),
                    )
            loss = loss + std_weight * std_pen

            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                u.clamp_(0.0, 1.0)

        with torch.no_grad():
            uc = torch.clamp(u, 0.0, 1.0).cpu().numpy()
            x_real = C.LOWER + uc * (C.UPPER - C.LOWER)
            f_val = obj_real(x_real)
        collector.add(f_val, x_real)
        if verbose:
            print(f"  [grad] restart {r+1:2d}/{n_restarts}  obj={f_val:.5f}  "
                  f"top1={collector.items[0][0]:.5f}")

    return collector


# =====================================================================
# 顶层接口
# =====================================================================
def inverse_design(y_target, w, method="cma", ckpt=FORWARD_CKPT, verbose=True,
                   free_passband=None, fixed_rolloff=None,
                   std_weight=0.0, top_k=10,
                   **kw):
    fm = ForwardModel(ckpt)
    if verbose:
        tag = f"ensemble({len(fm.models)})" if fm.is_ensemble else "single"
        print(f"[inv] 代理模型: {tag}, device={fm.device}")

    if free_passband is not None:
        obj_real = make_free_objective(
            fm, free_passband["target_bw"], free_passband["passband_db"],
            stopband_db=free_passband.get("stopband_db"),
            guard_ghz=free_passband.get("guard_ghz", 0.3),
            stop_weight=free_passband.get("stop_weight", 1.0),
            rolloff_db=free_passband.get("rolloff_db"),
            rolloff_ghz=free_passband.get("rolloff_ghz", 0.15),
            rolloff_weight=free_passband.get("rolloff_weight", 1.5),
            agg=free_passband.get("agg", "mean"),
            std_weight=std_weight,
            f0=free_passband.get("f0"), f0_tol=free_passband.get("f0_tol"),
        )
    else:
        obj_real = make_objective(
            fm, y_target, w,
            rolloff=fixed_rolloff, std_weight=std_weight,
        )

    if method == "cma":
        collector = solve_cma(fm, obj_real, top_k=top_k, verbose=verbose, **kw)
    elif method == "grad":
        collector = solve_grad(
            fm, y_target, w, obj_real,
            top_k=top_k, verbose=verbose,
            free=free_passband, fixed_rolloff=fixed_rolloff,
            std_weight=std_weight, **kw,
        )
    else:
        raise ValueError(f"未知 method: {method}")

    if not collector.items:
        raise RuntimeError("求解失败：没有得到任何候选")

    fs, xs = collector.as_arrays()
    x_best = xs[0]
    f_best = float(fs[0])
    ok, reason = C.check_constraints(x_best)
    y_pred = fm.predict(x_best)[0]
    s21_pred, s11_pred = split_y(y_pred)

    y_pred_topk = fm.predict(xs)
    s21_topk = y_pred_topk[:, :C.N_FREQ]

    result = {
        "x": x_best,
        "params": C.param_dict(x_best),
        "objective": f_best,
        "constraint_ok": ok,
        "constraint_reason": reason,
        "y_pred": y_pred,
        "s21_pred": s21_pred,
        "s11_pred": s11_pred,
        "freqs": fm.freqs,
        "method": method,
        "topk_x": xs,
        "topk_obj": fs,
        "topk_y_pred": y_pred_topk,
        "topk_s21": s21_topk,
    }

    if free_passband is not None:
        result["passband"] = find_best_passband(
            s21_pred, free_passband["target_bw"], free_passband["passband_db"],
            stopband_db=free_passband.get("stopband_db"),
            guard_ghz=free_passband.get("guard_ghz", 0.3),
            stop_weight=free_passband.get("stop_weight", 1.0),
            rolloff_db=free_passband.get("rolloff_db"),
            rolloff_ghz=free_passband.get("rolloff_ghz", 0.15),
            rolloff_weight=free_passband.get("rolloff_weight", 1.5),
            freq=fm.freqs, agg=free_passband.get("agg", "mean"),
            f0=free_passband.get("f0"), f0_tol=free_passband.get("f0_tol"),
        )
        result["free_passband"] = free_passband
    if fixed_rolloff is not None:
        result["fixed_rolloff"] = fixed_rolloff
    return result


def print_result(res):
    print("\n========== 逆向设计结果 ==========")
    print(f"method      = {res['method']}")
    print(f"objective   = {res['objective']:.5f}")
    print(f"约束满足    = {res['constraint_ok']}  ({res['constraint_reason']})")
    print("Top-1 参数 x̂:")
    for k in C.PARAM_NAMES:
        print(f"  {k:8s} = {res['params'][k]:.4f}")
    print(f"Top-K 候选数量 = {len(res['topk_obj'])} (obj range "
          f"{res['topk_obj'].min():.4f} ~ {res['topk_obj'].max():.4f})")
    if "passband" in res:
        print("---- 代理预测的通带质量 ----")
        print(format_passband_report(res["passband"], tag="surrogate"))
    print("==================================")
    print("下一步: python validate.py --result <npz>  → 一次跑 HFSS 复验所有 Top-K")


# =====================================================================
# 反设计预测效果图（不依赖 HFSS，每次反设计都出一张）
# =====================================================================
def plot_prediction(res, out_path, y_target=None, free_pb=None):
    """画 Top-1 代理预测 S21/S11 + 目标/通带/f0±tol/下降哨兵标注。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[predict] 跳过出图 (matplotlib 不可用: {e})")
        return
    f = np.asarray(res["freqs"], float)
    s21_p = np.asarray(res["s21_pred"], float)
    s11_p = np.asarray(res["s11_pred"], float)
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top-K 其余候选浅色叠加，Top-1 高亮
    if "topk_s21" in res and len(res["topk_s21"]) > 1:
        for i in range(1, len(res["topk_s21"])):
            ax[0].plot(f, res["topk_s21"][i], color="gray", lw=0.6, alpha=0.35)
    ax[0].plot(f, s21_p, color="#FF9800", lw=1.9, label="S21 predicted (Top-1)")

    s11_t = None
    if y_target is not None:
        s21_t, s11_t = split_y(np.asarray(y_target))
        ax[0].plot(f, s21_t, "--", color="#9C27B0", lw=1.4, label="S21 target")

    diag = res.get("passband")
    if diag and diag.get("win_flo") is not None:
        ax[0].axvspan(diag["win_flo"], diag["win_fhi"], color="#4CAF50", alpha=0.15,
                      label=f"passband {diag['win_flo']:.2f}~{diag['win_fhi']:.2f}GHz")
    fp = free_pb or {}
    if fp.get("f0") is not None and fp.get("f0_tol") is not None:
        ax[0].axvspan(fp["f0"] - fp["f0_tol"], fp["f0"] + fp["f0_tol"],
                      color="#3F51B5", alpha=0.06, label="f0±tol")
    if fp.get("passband_db") is not None:
        ax[0].axhline(fp["passband_db"], color="green", ls=":", alpha=0.6,
                      label=f"IL target {fp['passband_db']:.1f}dB")
    if fp.get("rolloff_db") is not None:
        ax[0].axhline(fp["rolloff_db"], color="red", ls=":", alpha=0.6,
                      label=f"rolloff {fp['rolloff_db']:.1f}dB")
    ax[0].set_ylabel("S21 (dB)"); ax[0].set_ylim(-60, 3)
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8, loc="lower center", ncol=2)

    ax[1].plot(f, s11_p, color="#2196F3", lw=1.5, label="S11 predicted (Top-1)")
    if s11_t is not None:
        ax[1].plot(f, s11_t, "--", color="#9C27B0", lw=1.2, label="S11 target")
    ax[1].set_ylabel("S11 (dB)"); ax[1].set_xlabel("Freq (GHz)"); ax[1].set_ylim(-40, 3)
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)

    fig.suptitle(f"Inverse design predicted response (surrogate, obj={res['objective']:.3f})")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"[predict] 代理预测效果图 → {out_path}")


# =====================================================================
# CLI
# =====================================================================
def _apply_margin(passband_db, stopband_db, margin_db):
    """给通带/阻带门槛加代理误差余量，使反解出的解在真值上更稳。"""
    if margin_db <= 0:
        return passband_db, stopband_db
    return passband_db + margin_db, (stopband_db - margin_db) if stopband_db is not None else None


def main():
    ap = argparse.ArgumentParser(description="逆向设计求解 (Layer2A / M3) — filter5 电路")
    ap.add_argument("--ckpt", default=FORWARD_CKPT,
                    help="模型 ckpt 路径 / ensemble index / 目录 / glob 模式")
    ap.add_argument("--method", choices=["cma", "grad"], default="cma")

    # ---- 目标 A：高层规格 ----
    ap.add_argument("--f0", type=float, help="带通中心频率 GHz")
    ap.add_argument("--bw", type=float, help="带通带宽 GHz（centered 模式下即『最小带宽』）")
    ap.add_argument("--f0_tol", type=float, default=0.0,
                    help="中心频率允许波动 ±GHz；>0 时启用『软约束中心频率』(centered) 模式")
    ap.add_argument("--passband_db", type=float, default=-0.5,
                    help="通带插损目标 dB（自由通带建议 -1 ~ -3）")
    ap.add_argument("--stopband_db", type=float, default=-30.0)
    ap.add_argument("--transition_ghz", type=float, default=0.15)
    ap.add_argument("--trans_weight", type=float, default=0.5)

    # ---- 目标 B：已有 s2p ----
    ap.add_argument("--target_s2p", help="用一条现成 s2p 作为目标")

    # ---- 目标 C：自由通带 ----
    ap.add_argument("--free", action="store_true",
                    help="自由通带模式：只要求存在够宽、插损够低的通带，中心频率不限")
    ap.add_argument("--target_bw", type=float, default=0.5)
    ap.add_argument("--agg", choices=["mean", "max"], default="mean")
    ap.add_argument("--guard_ghz", type=float, default=0.3)
    ap.add_argument("--stop_weight", type=float, default=1.0)

    # ---- 下降系数（skirt / rolloff）显式约束 ----
    ap.add_argument("--rolloff_ghz", type=float, default=0.15)
    ap.add_argument("--rolloff_db", type=float, default=-15.0,
                    help="哨兵点 S21 上限；写 0 表示禁用")
    ap.add_argument("--rolloff_weight", type=float, default=2.0)
    ap.add_argument("--no_rolloff", action="store_true", help="禁用下降系数约束")

    # ---- 代理误差鲁棒性 ----
    ap.add_argument("--margin_db", type=float, default=0.0,
                    help="给 passband/stopband 加代理误差余量（dB），让真值更稳")
    ap.add_argument("--std_penalty", type=float, default=0.0,
                    help="Ensemble std 惩罚权重（>0 且加载了 ensemble 才生效）")

    # ---- 求解超参 ----
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--top_k", type=int, default=10, help="保留的候选数量")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out", help="把结果保存到 npz 的路径")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[错误] 未找到代理模型 {args.ckpt}，请先训练: python forward_net.py --train")
        sys.exit(1)

    pb_db, sb_db = _apply_margin(args.passband_db, args.stopband_db, args.margin_db)
    if args.margin_db > 0:
        print(f"[target] 代理余量 margin={args.margin_db} dB → "
              f"passband {args.passband_db}→{pb_db} dB, stopband {args.stopband_db}→{sb_db} dB")

    rolloff_enabled = (not args.no_rolloff) and (args.rolloff_db != 0)
    fixed_rolloff = None
    free_pb = None
    y_target, w = None, None

    # centered（软约束中心频率）：--f0 --bw 且 --f0_tol>0 → 复用自由通带机制 + 窗口中心约束
    centered_enabled = (not args.free and not args.target_s2p
                        and args.f0 is not None and args.bw is not None
                        and args.f0_tol is not None and args.f0_tol > 0)

    if args.free or centered_enabled:
        free_pb = {
            "target_bw": (args.bw if centered_enabled else args.target_bw),
            "passband_db": pb_db,
            "stopband_db": sb_db,
            "guard_ghz": args.guard_ghz,
            "stop_weight": args.stop_weight,
            "agg": args.agg,
        }
        if rolloff_enabled:
            free_pb.update({
                "rolloff_db": args.rolloff_db,
                "rolloff_ghz": args.rolloff_ghz,
                "rolloff_weight": args.rolloff_weight,
            })
        if centered_enabled:
            free_pb["f0"] = args.f0
            free_pb["f0_tol"] = args.f0_tol
            print(f"[target] 软约束中心频率(centered): f0={args.f0}±{args.f0_tol} GHz, "
                  f"最小带宽>={args.bw} GHz, 插损>={pb_db} dB, 阻带<={sb_db} dB, "
                  f"skirt={args.rolloff_db if rolloff_enabled else '关闭'} dB@{args.rolloff_ghz} GHz")
        else:
            print(f"[target] 自由通带模式: 带宽>={args.target_bw} GHz, "
                  f"插损>={pb_db} dB, 阻带<={sb_db} dB, "
                  f"skirt={args.rolloff_db if rolloff_enabled else '关闭'} dB@{args.rolloff_ghz} GHz")
    elif args.target_s2p:
        y_target, w = spec_mod.target_from_s2p(args.target_s2p)
        print(f"[target] 来自 s2p: {args.target_s2p}")
    elif args.f0 is not None and args.bw is not None:
        y_target, w = spec_mod.make_bandpass_target(
            f0=args.f0, bw=args.bw,
            passband_db=pb_db, stopband_db=sb_db,
            transition_ghz=args.transition_ghz,
            trans_weight=args.trans_weight,
        )
        print(f"[target] 带通规格: f0={args.f0} GHz, BW={args.bw} GHz, "
              f"trans={args.transition_ghz} GHz (w={args.trans_weight})")
        if rolloff_enabled:
            fixed_rolloff = {
                "f0": args.f0, "bw": args.bw,
                "rolloff_ghz": args.rolloff_ghz,
                "rolloff_db": args.rolloff_db,
                "weight": args.rolloff_weight,
            }
            print(f"[target] 显式下降沿: 通带边外 {args.rolloff_ghz} GHz 处 "
                  f"S21<={args.rolloff_db} dB (w={args.rolloff_weight})")
    else:
        print("[错误] 请提供目标：--free 或 --f0/--bw 或 --target_s2p")
        sys.exit(1)

    kw = {"n_restarts": args.restarts, "maxiter": args.maxiter, "seed": args.seed}
    if args.method == "grad":
        kw = {"n_restarts": max(args.restarts, 4),
              "n_steps": args.maxiter, "seed": args.seed}

    res = inverse_design(
        y_target, w, method=args.method, ckpt=args.ckpt,
        free_passband=free_pb, fixed_rolloff=fixed_rolloff,
        std_weight=args.std_penalty, top_k=args.top_k, **kw,
    )
    print_result(res)

    # 反设计预测效果图（不依赖 HFSS，每次反设计都出一张）
    _pred_png = (os.path.splitext(args.out)[0] + "_pred.png") if args.out \
        else os.path.join(C.DATA_DIR, "inv_pred.png")
    try:
        os.makedirs(os.path.dirname(_pred_png) or ".", exist_ok=True)
        plot_prediction(res, _pred_png, y_target=y_target, free_pb=free_pb)
    except Exception as e:
        print(f"[predict] 出图跳过: {e}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        save_kw = dict(
            x=res["x"], y_pred=res["y_pred"], freqs=res["freqs"],
            topk_x=res["topk_x"], topk_obj=res["topk_obj"],
            topk_y_pred=res["topk_y_pred"],
            param_names=np.array(C.PARAM_NAMES),
        )
        if free_pb is not None:
            save_kw.update(mode="free_passband",
                           target_bw=free_pb["target_bw"],
                           passband_db=free_pb["passband_db"],
                           stopband_db=(free_pb.get("stopband_db") if free_pb.get("stopband_db") is not None else np.nan),
                           rolloff_db=(free_pb.get("rolloff_db") if free_pb.get("rolloff_db") is not None else np.nan),
                           rolloff_ghz=free_pb.get("rolloff_ghz", 0.15),
                           guard_ghz=free_pb.get("guard_ghz", 0.3))
            # centered：额外存中心频率软约束，validate 自动沿用
            if free_pb.get("f0") is not None:
                save_kw.update(f0=free_pb["f0"], f0_tol=free_pb.get("f0_tol", 0.0))
        else:
            save_kw.update(mode="fixed_target", y_target=y_target, w=w)
            if args.f0 is not None and args.bw is not None:
                save_kw.update(f0=args.f0, bw=args.bw,
                               transition_ghz=args.transition_ghz)
            if fixed_rolloff is not None:
                save_kw.update(rolloff_db=fixed_rolloff["rolloff_db"],
                               rolloff_ghz=fixed_rolloff["rolloff_ghz"])
        np.savez(args.out, **save_kw)
        print(f"[save] 结果已保存 → {args.out}")


if __name__ == "__main__":
    main()
