"""
位置无关的「自由通带」评价（free passband spec）。

【本版重构，把"下降系数"显式引入 loss】：
  - 老版 loss = 通带缺额（越低越好）
  - 新版 loss = 通带缺额
                + stop_weight * 阻带越界（S21 > stopband_db 且离通带 >= guard_ghz）
                + rolloff_weight * skirt 缺额（通带边缘外 rolloff_ghz 处 S21 未掉到 rolloff_db）

  - stopband_db = None → 不加阻带项（老行为，向后兼容）
  - rolloff_db  = None → 不加 skirt 项

一句话概括：
  只满足"够宽 + 够低"的通带 → 会被"没有下降沿"的例子作弊；
  同时约束阻带 + skirt → 优化器被迫画出真正的带通形状。

对 filter5 尤为重要：3 阶带通 + via 传输零点的结构天然能做出陡下降沿，
显式约束 rolloff/stopband 能引导优化器充分利用这些物理优势。
"""

import numpy as np

import id_config as C


def bw_to_nwin(target_bw, step_ghz=C.FREQ_STEP_GHZ, n_freq=C.N_FREQ):
    """目标带宽 target_bw(GHz) → 覆盖它所需的连续频点数（>=2）。"""
    n = int(round(float(target_bw) / float(step_ghz))) + 1
    return int(np.clip(n, 2, n_freq))


def _ghz_to_pts(x_ghz, step_ghz=C.FREQ_STEP_GHZ):
    return max(0, int(round(float(x_ghz) / float(step_ghz))))


# =====================================================================
# numpy 版：滑动窗口最佳通带 + 阻带/skirt 惩罚
# =====================================================================
def passband_free_loss(
    s21_db, target_bw,
    passband_db=-3.0,
    stopband_db=None, guard_ghz=0.3, stop_weight=1.0,
    rolloff_db=None, rolloff_ghz=0.15, rolloff_weight=1.5,
    freq=None, agg="mean",
    f0=None, f0_tol=None,
):
    """
    自由通带综合 loss（越小越好，单位 dB）。

    通带部分：
      在长度 n_win 的滑动窗口里取「通带缺额」（S21 < passband_db 的量）最小者。

    阻带部分（stopband_db != None）：
      窗口外 guard_ghz 之外的频点，若 S21 > stopband_db，按超出量加惩罚。

    Skirt 部分（rolloff_db != None）：
      窗口左右各 rolloff_ghz 处的两个"哨兵点"，若 S21 > rolloff_db，按超出量加惩罚。
      这是"下降系数"最直接的显式约束。

    返回：最优窗口对应的总 cost。
    """
    s21 = np.asarray(s21_db, float)
    n_freq = s21.shape[0]
    n_win = bw_to_nwin(target_bw)
    if n_freq < n_win:
        return float(max(0.0, passband_db - s21.max()) + 1e3)

    # centered 模式：只考虑「窗口中心落在 [f0-f0_tol, f0+f0_tol]」的窗口
    fgrid = np.asarray(freq if freq is not None else C.FREQ_GHZ, float)
    _centered = (f0 is not None and f0_tol is not None)

    pass_def = np.maximum(passband_db - s21, 0.0)

    stop_exc = None
    if stopband_db is not None:
        stop_exc = np.maximum(s21 - stopband_db, 0.0)
    n_guard = _ghz_to_pts(guard_ghz)
    n_roll = _ghz_to_pts(rolloff_ghz)

    best = np.inf
    for start in range(0, n_freq - n_win + 1):
        end = start + n_win  # 半开区间 [start, end)

        # centered：窗口中心不在 f0±tol 内则跳过
        if _centered:
            fc = 0.5 * (fgrid[start] + fgrid[end - 1])
            if abs(fc - f0) > f0_tol + 1e-9:
                continue

        # 1) 通带缺额
        w = pass_def[start:end]
        pass_cost = float(w.mean()) if agg == "mean" else float(w.max())

        cost = pass_cost

        # 2) 阻带越界
        if stop_exc is not None and stop_weight > 0:
            lo = max(0, start - n_guard)
            hi = min(n_freq, end + n_guard)
            mask = np.ones(n_freq, dtype=bool)
            mask[lo:hi] = False
            if mask.any():
                stop_cost = float(stop_exc[mask].mean())
                cost += stop_weight * stop_cost

        # 3) skirt / rolloff 哨兵
        if rolloff_db is not None and rolloff_weight > 0 and n_roll > 0:
            left = start - 1 - n_roll     # 通带边缘外 rolloff_ghz 处的哨兵
            right = end + n_roll          # 半开区间 → end 即通带外第一个点
            skirt = 0.0
            n_taken = 0
            if 0 <= left < n_freq:
                skirt += max(0.0, s21[left] - rolloff_db)
                n_taken += 1
            if 0 <= right < n_freq:
                skirt += max(0.0, s21[right] - rolloff_db)
                n_taken += 1
            if n_taken > 0:
                cost += rolloff_weight * (skirt / n_taken)

        if cost < best:
            best = cost
    if not np.isfinite(best):
        # 没有窗口中心落入 f0±tol（tol 太小或超出频带）→ 大惩罚
        return float(1e3 + abs(passband_db))
    return best


# =====================================================================
# torch 可微版：soft-min over windows（供梯度逆向）
# =====================================================================
def passband_free_loss_torch(
    s21_db, target_bw,
    passband_db=-3.0,
    stopband_db=None, guard_ghz=0.3, stop_weight=1.0,
    rolloff_db=None, rolloff_ghz=0.15, rolloff_weight=1.5,
    tau=0.1, f0=None, f0_tol=None, freq=None,
):
    """
    torch 可微版本。对每个可能的通带位置计算综合 cost，然后用 soft-min 聚合，
    保持梯度对"最优窗口"位置可反传。
    """
    import torch

    n_freq = s21_db.shape[-1]
    n_win = bw_to_nwin(target_bw, n_freq=n_freq)
    n_guard = _ghz_to_pts(guard_ghz)
    n_roll = _ghz_to_pts(rolloff_ghz)
    pass_def = torch.relu(passband_db - s21_db)

    stop_exc = None
    if stopband_db is not None:
        stop_exc = torch.relu(s21_db - stopband_db)

    # 1) 通带项：所有窗口的 mean 缺额
    pass_windows = pass_def.unfold(0, n_win, 1)       # (num_win, n_win)
    win_pass = pass_windows.mean(dim=1)               # (num_win,)
    win_cost = win_pass

    # 2) 阻带项：每个窗口位置对应一个 mask 均值（用累积和加速，list + stack 避免 in-place）
    if stop_exc is not None and stop_weight > 0:
        cs = torch.cumsum(stop_exc, dim=0)
        cs_pad = torch.cat([cs.new_zeros(1), cs])
        total = cs[-1]
        num_win = win_pass.shape[0]
        stop_items = []
        for start in range(num_win):
            end = start + n_win
            lo = max(0, start - n_guard)
            hi = min(n_freq, end + n_guard)
            inside = cs_pad[hi] - cs_pad[lo]
            outside_sum = total - inside
            outside_n = n_freq - (hi - lo)
            if outside_n > 0:
                stop_items.append(outside_sum / outside_n)
            else:
                stop_items.append(s21_db.new_zeros(()))
        stop_cost = torch.stack(stop_items, dim=0)
        win_cost = win_cost + stop_weight * stop_cost

    # 3) skirt 哨兵（list + stack 避免 in-place）
    if rolloff_db is not None and rolloff_weight > 0 and n_roll > 0:
        num_win = win_pass.shape[0]
        skirt_items = []
        for start in range(num_win):
            end = start + n_win
            left = start - 1 - n_roll
            right = end + n_roll
            vals = s21_db.new_zeros(())
            cnt = 0
            if 0 <= left < n_freq:
                vals = vals + torch.relu(s21_db[left] - rolloff_db)
                cnt += 1
            if 0 <= right < n_freq:
                vals = vals + torch.relu(s21_db[right] - rolloff_db)
                cnt += 1
            if cnt > 0:
                skirt_items.append(vals / cnt)
            else:
                skirt_items.append(vals)
        skirt_cost = torch.stack(skirt_items, dim=0)
        win_cost = win_cost + rolloff_weight * skirt_cost

    # centered：窗口中心不在 f0±tol 内的，加大数使 soft-min 不选中
    if f0 is not None and f0_tol is not None:
        fgrid = np.asarray(freq if freq is not None else C.FREQ_GHZ, float)
        num_win = win_cost.shape[0]
        centers = 0.5 * (fgrid[:num_win] + fgrid[n_win - 1:n_win - 1 + num_win])
        invalid = np.abs(centers - f0) > f0_tol + 1e-9
        if invalid.any():
            add = torch.tensor(np.where(invalid, 1e6, 0.0), dtype=win_cost.dtype,
                               device=win_cost.device)
            win_cost = win_cost + add

    return -tau * torch.logsumexp(-win_cost / tau, dim=0)


# =====================================================================
# 诊断：找出最佳通带位置 + 下降系数指标
# =====================================================================
def find_best_passband(
    s21_db, target_bw,
    passband_db=-3.0,
    stopband_db=None, guard_ghz=0.3, stop_weight=1.0,
    rolloff_db=None, rolloff_ghz=0.15, rolloff_weight=1.5,
    freq=None, agg="mean",
    f0=None, f0_tol=None,
):
    """
    在 S21 曲线上找「最佳通带」，返回完整诊断 dict。

    额外报告的下降系数指标：
      - rolloff_left_db_per_ghz : 通带左边缘处的 |dS21/df|（dB/GHz）
      - rolloff_right_db_per_ghz: 通带右边缘处的 |dS21/df|
      - shape_factor            : 阻带带宽 / 通带带宽（≥1，越接近 1 越陡）
                                   （阻带 = S21 掉到 stopband_db 以下的最大连续段）
    """
    s21 = np.asarray(s21_db, float)
    f = np.asarray(freq if freq is not None else C.FREQ_GHZ, float)
    n_win = bw_to_nwin(target_bw)
    step = float(C.FREQ_STEP_GHZ)
    n_freq = s21.shape[0]

    out = {
        "loss": passband_free_loss(
            s21, target_bw, passband_db=passband_db,
            stopband_db=stopband_db, guard_ghz=guard_ghz, stop_weight=stop_weight,
            rolloff_db=rolloff_db, rolloff_ghz=rolloff_ghz, rolloff_weight=rolloff_weight,
            agg=agg, f0=f0, f0_tol=f0_tol,
        ),
        "target_bw": float(target_bw),
        "passband_db": float(passband_db),
        "stopband_db": None if stopband_db is None else float(stopband_db),
        "rolloff_db": None if rolloff_db is None else float(rolloff_db),
        "n_win": n_win,
        "f0": None if f0 is None else float(f0),
        "f0_tol": None if f0_tol is None else float(f0_tol),
    }

    # 找最佳通带位置（centered 时限制窗口中心在 f0±tol 内）
    _centered = (f0 is not None and f0_tol is not None)
    deficit = np.maximum(passband_db - s21, 0.0)
    best_cost, best_start = np.inf, None
    for start in range(0, n_freq - n_win + 1):
        if _centered:
            fc = 0.5 * (f[start] + f[start + n_win - 1])
            if abs(fc - f0) > f0_tol + 1e-9:
                continue
        w = deficit[start:start + n_win]
        cost = float(w.mean()) if agg == "mean" else float(w.max())
        if cost < best_cost:
            best_cost, best_start = cost, start
    if best_start is None:
        best_start = 0
    b0, b1 = best_start, best_start + n_win - 1
    out.update({
        "win_flo": float(f[b0]),
        "win_fhi": float(f[b1]),
        "win_f0": float(0.5 * (f[b0] + f[b1])),
        "win_min_s21": float(s21[b0:b1 + 1].min()),
        "win_mean_s21": float(s21[b0:b1 + 1].mean()),
    })

    # 实际达标通带
    ok = s21 >= passband_db
    best_len, best_i0 = 0, None
    cur_len, cur_i0 = 0, 0
    for i, flag in enumerate(ok):
        if flag:
            if cur_len == 0:
                cur_i0 = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_i0 = cur_len, cur_i0
        else:
            cur_len = 0
    if best_len >= 1 and best_i0 is not None:
        i0, i1 = best_i0, best_i0 + best_len - 1
        out.update({
            "achieved_bw": float((best_len - 1) * step) if best_len >= 2 else 0.0,
            "achieved_flo": float(f[i0]),
            "achieved_fhi": float(f[i1]),
        })
    else:
        out.update({"achieved_bw": 0.0, "achieved_flo": None, "achieved_fhi": None})

    # ---- 下降系数指标 ----
    n_probe = 2  # 取通带边缘外 2 个频点做一阶差分
    # 左边缘：b0 → b0 - n_probe
    if b0 - n_probe >= 0:
        d_left = (s21[b0] - s21[b0 - n_probe]) / (n_probe * step)   # 正常应该 > 0（进入通带时上升）
        out["rolloff_left_db_per_ghz"] = float(abs(d_left))
    else:
        out["rolloff_left_db_per_ghz"] = None
    # 右边缘：b1 → b1 + n_probe
    if b1 + n_probe < n_freq:
        d_right = (s21[b1] - s21[b1 + n_probe]) / (n_probe * step)  # 正常应该 > 0（离开通带时下降）
        out["rolloff_right_db_per_ghz"] = float(abs(d_right))
    else:
        out["rolloff_right_db_per_ghz"] = None

    # shape factor：阻带带宽（<= stopband_db）/ 通带带宽（>= passband_db）
    if stopband_db is not None:
        stop_mask = s21 <= stopband_db
        s_len, s_i0 = 0, None
        cur, ci0 = 0, 0
        for i, flag in enumerate(stop_mask):
            if flag:
                if cur == 0:
                    ci0 = i
                cur += 1
                if cur > s_len:
                    s_len, s_i0 = cur, ci0
            else:
                cur = 0
        stop_bw = (s_len - 1) * step if s_len >= 2 else 0.0
        out["stopband_bw"] = float(stop_bw)
        if out["achieved_bw"] > 0 and stop_bw > 0:
            out["shape_factor"] = float(stop_bw / out["achieved_bw"])
        else:
            out["shape_factor"] = None
    else:
        out["stopband_bw"] = None
        out["shape_factor"] = None

    return out


def format_passband_report(diag, tag="passband"):
    lines = [
        f"[{tag}] 目标: 带宽>={diag['target_bw']:.2f} GHz, 插损>={diag['passband_db']:.1f} dB "
        f"(窗口 {diag['n_win']} 点)",
        f"[{tag}] 综合 loss = {diag['loss']:.3f} dB  (0=完美达标)",
        f"[{tag}] 最佳通带窗口: {diag['win_flo']:.2f}~{diag['win_fhi']:.2f} GHz "
        f"(中心 {diag['win_f0']:.2f} GHz) | 窗口内最差 S21={diag['win_min_s21']:.2f} dB, "
        f"平均 S21={diag['win_mean_s21']:.2f} dB",
    ]
    if diag["achieved_flo"] is not None:
        lines.append(
            f"[{tag}] 实际达标(S21>={diag['passband_db']:.1f}dB)最大连续带宽 = "
            f"{diag['achieved_bw']:.2f} GHz ({diag['achieved_flo']:.2f}~{diag['achieved_fhi']:.2f} GHz)"
        )
    else:
        lines.append(f"[{tag}] 无任何频点达到插损目标 {diag['passband_db']:.1f} dB")

    rl = diag.get("rolloff_left_db_per_ghz")
    rr = diag.get("rolloff_right_db_per_ghz")
    if rl is not None or rr is not None:
        rl_s = f"{rl:.1f}" if rl is not None else "—"
        rr_s = f"{rr:.1f}" if rr is not None else "—"
        lines.append(f"[{tag}] 下降系数(通带边缘 |dS21/df|): 左={rl_s} dB/GHz, 右={rr_s} dB/GHz")
    if diag.get("shape_factor") is not None:
        lines.append(
            f"[{tag}] shape factor = stopBW/passBW = "
            f"{diag['stopband_bw']:.2f}/{diag['achieved_bw']:.2f} = "
            f"{diag['shape_factor']:.2f}  (越接近 1 越陡)"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    f = C.FREQ_GHZ
    s21 = np.full_like(f, -30.0)
    s21[(f >= 2.7) & (f <= 3.3)] = -0.5
    d1 = find_best_passband(s21, target_bw=0.5, passband_db=-3.0)
    print("=== 老行为（无阻带/skirt 项） ===")
    print(format_passband_report(d1, tag="old"))
    d2 = find_best_passband(s21, target_bw=0.5, passband_db=-3.0,
                            stopband_db=-20.0, guard_ghz=0.3, stop_weight=1.0,
                            rolloff_db=-15.0, rolloff_ghz=0.15, rolloff_weight=1.5)
    print("\n=== 新行为（含阻带 + skirt） ===")
    print(format_passband_report(d2, tag="new"))
