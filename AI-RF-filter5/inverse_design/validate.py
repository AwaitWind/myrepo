"""
闭环验证（M3c）—— 把逆向求出的参数 x̂ 代入「真 HFSS」仿真，与目标对比。

【本版重构核心改动】：
  1. 支持 Top-K 复验：一次跑 K 个候选，全部仿真后按"真值 loss"重排
     → 代理最优 ≠ 真值最优，K 个候选大大提升"至少一个真正好"的概率
  2. 生成汇总表格 + 叠加对比图
  3. 断点续跑：若某 idx 已有 .s2p 文件，可跳过 HFSS 仿真直接复用

注：filter5 项目的 HFSS 建模函数为 filter_layout_generate（CPW+3阶谐振器+via 阵列）。

用法:
    python validate.py --result id_data/inv_result.npz            # 只验 top-1
    python validate.py --result id_data/inv_result.npz --top_k 10 # 全部 top-K 验证
    python validate.py --x <19 个参数> --f0 3.5 --bw 0.5
    python validate.py --result id_data/inv_free.npz              # 自由通带模式自动沿用 npz 设置
"""

import argparse
import os
import sys

import numpy as np

# 路径处理：让本目录与上级项目目录都可被 import（filter5 平铺，无 PC25 层级）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_THIS_DIR)     # AI-RF-filter5/
for _p in (_THIS_DIR, _PROJ_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import id_config as C
from dataio import s2p_to_y, split_y
import spec as spec_mod
from passband_spec import find_best_passband, format_passband_report, passband_free_loss


# =====================================================================
# 用真 HFSS 仿真一组参数
# =====================================================================
def simulate_real(x_real, tag="validate", reuse=True):
    """
    对参数 x_real 用真 HFSS 仿真，返回 (s2p_path, y_real)。
    reuse=True 时若目标 s2p 已存在则直接复用（断点续跑）。
    """
    s2p_path = os.path.join(C.DATA_DIR, f"{tag}.s2p")
    if reuse and os.path.exists(s2p_path):
        y_real = s2p_to_y(s2p_path)
        print(f"[validate] 复用已有 s2p: {s2p_path}")
        return s2p_path, y_real

    from ansys.aedt.core import settings as pyaedt_settings
    from HFSS_funcs import init_hfss, filter_layout_generate, hfss_simulate

    pyaedt_settings.use_grpc_api = True
    C.ensure_dirs()

    hfss = None
    try:
        hfss, setup, setup_name, sweep_name, sweeps_name = init_hfss(C.CENTER_FREQ_GHZ)
        kwargs = C.param_to_kwargs(x_real)
        filter_layout_generate(hfss, **kwargs)
        hfss_simulate(hfss, s2p_path, setup_name=setup_name,
                      sweep_name=sweeps_name[0], cores=C.SIM_CORES, tasks=C.SIM_TASKS)
    finally:
        if hfss is not None:
            try:
                hfss.release_desktop()
            except Exception:
                pass

    y_real = s2p_to_y(s2p_path)
    return s2p_path, y_real


# =====================================================================
# 真值 loss（与 inverse_optimize 的目标函数保持一致）
# =====================================================================
def true_loss_fixed(s21_real, s21_target, w_s21):
    """固定 target 模式的真值 loss（加权 MSE，仅 S21）。"""
    diff = (s21_real - s21_target) * np.sqrt(np.maximum(w_s21, 0.0))
    return float(np.mean(diff ** 2))


def true_loss_free(s21_real, free_pb):
    """自由通带模式的真值 loss（passband + stopband + rolloff）。"""
    return passband_free_loss(
        s21_real, free_pb["target_bw"], free_pb["passband_db"],
        stopband_db=free_pb.get("stopband_db"),
        guard_ghz=free_pb.get("guard_ghz", 0.3),
        stop_weight=free_pb.get("stop_weight", 1.0),
        rolloff_db=free_pb.get("rolloff_db"),
        rolloff_ghz=free_pb.get("rolloff_ghz", 0.15),
        rolloff_weight=free_pb.get("rolloff_weight", 1.5),
        agg=free_pb.get("agg", "mean"),
    )


# =====================================================================
# 单点评估 + 画图
# =====================================================================
def plot_compare(y_target, y_pred, y_real, out_path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = C.FREQ_GHZ
    s21_t, _ = split_y(y_target) if y_target is not None else (None, None)
    s21_r, s11_r = split_y(y_real)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    if s21_t is not None:
        ax.plot(f, s21_t, label="target S21", color="#9C27B0", lw=2.0, ls="--")
    if y_pred is not None:
        s21_p, _ = split_y(y_pred)
        ax.plot(f, s21_p, label="surrogate S21", color="#FF9800", lw=1.5)
    ax.plot(f, s21_r, label="HFSS S21 (真值)", color="#F44336", lw=2.0)
    if s11_r is not None:
        ax.plot(f, s11_r, label="HFSS S11", color="#2196F3", lw=1.0, alpha=0.7)

    ax.axhline(-3, color="gray", ls=":", alpha=0.6)
    ax.axhline(-30, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S| (dB)")
    ax.set_ylim(-60, 2)
    ax.set_xlim(f[0], f[-1])
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(loc="lower center", ncol=2)
    ax.set_title(title or "Inverse design validation: target vs surrogate vs HFSS")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_free(y_real, y_pred, diag, passband_db, out_path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = C.FREQ_GHZ
    s21_r, s11_r = split_y(y_real)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(f, s21_r, label="HFSS S21 (真值)", color="#F44336", lw=2.0)
    if y_pred is not None:
        s21_p, _ = split_y(np.asarray(y_pred))
        ax.plot(f, s21_p, label="surrogate S21", color="#FF9800", lw=1.5)
    if s11_r is not None:
        ax.plot(f, s11_r, label="HFSS S11", color="#2196F3", lw=1.0, alpha=0.7)

    ax.axvspan(diag["win_flo"], diag["win_fhi"], color="#4CAF50", alpha=0.15,
               label=f"best passband {diag['win_flo']:.2f}~{diag['win_fhi']:.2f} GHz")
    ax.axhline(passband_db, color="green", ls=":", alpha=0.8,
               label=f"insertion-loss target {passband_db:.1f} dB")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S| (dB)")
    ax.set_ylim(-60, 2)
    ax.set_xlim(f[0], f[-1])
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(loc="lower center", ncol=2)
    ax.set_title(title or "Free-passband validation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_topk_overlay(topk_results, target_data, out_path, mode="fixed"):
    """把 K 个真值曲线叠在一张图上，用颜色深浅表示真值 loss 排名。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = C.FREQ_GHZ
    fig, ax = plt.subplots(figsize=(12, 6))

    # target
    if mode == "fixed" and target_data is not None:
        s21_t, _ = split_y(target_data)
        ax.plot(f, s21_t, label="target", color="#000000", lw=2.5, ls="--", zorder=10)

    # top-K
    cmap = plt.cm.viridis
    K = len(topk_results)
    for rank, r in enumerate(topk_results):
        y_real = r["y_real"]
        s21_r, _ = split_y(y_real)
        color = cmap(0.15 + 0.75 * rank / max(1, K - 1))
        lw = 2.2 if rank == 0 else 1.1
        alpha = 1.0 if rank == 0 else 0.55
        ax.plot(f, s21_r,
                label=f"rank{rank+1} obj={r['true_loss']:.3f}"
                      + (" ★" if rank == 0 else ""),
                color=color, lw=lw, alpha=alpha)

    if mode == "free" and target_data is not None:
        pb_db = target_data.get("passband_db")
        if pb_db is not None:
            ax.axhline(pb_db, color="green", ls=":", alpha=0.7,
                       label=f"passband target {pb_db:.1f} dB")
        sb_db = target_data.get("stopband_db")
        if sb_db is not None:
            ax.axhline(sb_db, color="red", ls=":", alpha=0.7,
                       label=f"stopband target {sb_db:.1f} dB")

    ax.axhline(-3, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S21| (dB)")
    ax.set_ylim(-60, 2)
    ax.set_xlim(f[0], f[-1])
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(loc="lower center", ncol=3, fontsize=8)
    ax.set_title(f"Top-{K} candidates HFSS validation (sorted by true loss)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =====================================================================
# 单点评估
# =====================================================================
def evaluate_one(x_real, y_pred=None, y_target=None, free_passband=None,
                 tag="validate", reuse=True, plot=True):
    x_real = np.asarray(x_real, float)
    ok, reason = C.check_constraints(x_real)

    s2p_path, y_real = simulate_real(x_real, tag=tag, reuse=reuse)
    s21_r, _ = split_y(y_real)

    info = {
        "x": x_real,
        "s2p": s2p_path,
        "y_real": y_real,
        "constraint_ok": ok,
        "constraint_reason": reason,
    }

    if free_passband is not None:
        info["true_loss"] = true_loss_free(s21_r, free_passband)
        diag = find_best_passband(
            s21_r, free_passband["target_bw"], free_passband["passband_db"],
            stopband_db=free_passband.get("stopband_db"),
            rolloff_db=free_passband.get("rolloff_db"),
            rolloff_ghz=free_passband.get("rolloff_ghz", 0.15),
            freq=C.FREQ_GHZ, agg=free_passband.get("agg", "mean"),
        )
        info["diag"] = diag
        if plot:
            out_png = os.path.join(C.DATA_DIR, f"{tag}_freepass.png")
            plot_free(y_real, y_pred, diag, free_passband["passband_db"],
                      out_png, title=tag)
            info["plot"] = out_png

    elif y_target is not None:
        s21_t, _ = split_y(y_target)
        w_s21 = np.where(s21_t >= s21_t.max() - 1e-6, 3.0, 2.0)
        info["true_loss"] = true_loss_fixed(s21_r, s21_t, w_s21)
        info["mae_target"] = float(np.mean(np.abs(s21_r - s21_t)))
        pass_mask = s21_t >= (s21_t.max() - 1e-6)
        info["mae_pass"] = float(np.mean(np.abs(s21_r[pass_mask] - s21_t[pass_mask]))) \
            if pass_mask.any() else None
        if plot:
            out_png = os.path.join(C.DATA_DIR, f"{tag}_compare.png")
            plot_compare(y_target, y_pred, y_real, out_png, title=tag)
            info["plot"] = out_png

    else:
        info["true_loss"] = None

    if y_pred is not None:
        s21_p, _ = split_y(np.asarray(y_pred))
        info["mae_surrogate"] = float(np.mean(np.abs(s21_r - s21_p)))

    return info


# =====================================================================
# 主流程：单点 or Top-K
# =====================================================================
def validate_topk(candidates_x, candidates_y_pred=None,
                  y_target=None, free_passband=None,
                  tag_prefix="validate", reuse=True):
    """
    candidates_x     shape (K, N_PARAMS)
    candidates_y_pred shape (K, Y_DIM) or None
    返回：list[info]（按 true_loss 排序）
    """
    results = []
    K = len(candidates_x)
    print(f"\n========== 开始 Top-{K} HFSS 复验 ==========")
    for i in range(K):
        tag = f"{tag_prefix}_rank{i+1:02d}"
        print(f"\n---------- [{i+1}/{K}] {tag} ----------")
        x = candidates_x[i]
        print(f"  x = {dict(zip(C.PARAM_NAMES, np.round(x,4).tolist()))}")
        y_pred = candidates_y_pred[i] if candidates_y_pred is not None else None
        try:
            info = evaluate_one(x, y_pred=y_pred, y_target=y_target,
                                free_passband=free_passband,
                                tag=tag, reuse=reuse, plot=True)
            info["orig_rank"] = i + 1
            if info["true_loss"] is not None:
                print(f"  → true_loss = {info['true_loss']:.4f}")
            results.append(info)
        except Exception as e:
            print(f"  [错误] 候选 {i+1} 仿真失败: {e}")

    # 按真值 loss 排序
    valid = [r for r in results if r.get("true_loss") is not None]
    valid.sort(key=lambda r: r["true_loss"])

    print("\n========== Top-K 复验汇总（按真值 loss 排序） ==========")
    header = f"{'rank':>4} {'orig':>5} {'true_loss':>10} {'mae_srg':>8} {'ok':>3}"
    if free_passband is not None:
        header += f" {'achv_bw':>8} {'win_min':>8}"
    print(header)
    for i, r in enumerate(valid):
        line = (f"{i+1:>4} {r['orig_rank']:>5} {r['true_loss']:>10.4f} "
                f"{r.get('mae_surrogate', float('nan')):>8.3f} "
                f"{'Y' if r['constraint_ok'] else 'N':>3}")
        if free_passband is not None and "diag" in r:
            d = r["diag"]
            line += f" {d.get('achieved_bw', 0):>8.2f} {d.get('win_min_s21', 0):>8.2f}"
        print(line)

    # 汇总图
    if valid:
        out = os.path.join(C.DATA_DIR, f"{tag_prefix}_topk_overlay.png")
        if free_passband is not None:
            plot_topk_overlay(valid, free_passband, out, mode="free")
        else:
            plot_topk_overlay(valid, y_target, out, mode="fixed")
        print(f"\n[validate] Top-K 叠加图 → {out}")

        # 打印真值最优的详细报告
        best = valid[0]
        print("\n========== 真值最优候选 ==========")
        print(f"orig_rank    = {best['orig_rank']}")
        print(f"true_loss    = {best['true_loss']:.4f}")
        print(f"约束满足     = {best['constraint_ok']} ({best['constraint_reason']})")
        print("参数 x̂:")
        for k, v in zip(C.PARAM_NAMES, best["x"]):
            print(f"  {k:8s} = {v:.4f}")
        if "diag" in best:
            print(format_passband_report(best["diag"], tag="HFSS真值"))
    else:
        print("\n[validate][警告] 所有候选都仿真失败")

    return valid


# =====================================================================
# CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="逆向结果真 HFSS 验证 (M3c) — filter5 电路")
    ap.add_argument("--x", type=float, nargs=C.N_PARAMS, metavar=tuple(C.PARAM_NAMES),
                    help=f"直接给 {C.N_PARAMS} 个参数")
    ap.add_argument("--result", help="inverse_optimize 保存的 npz")
    ap.add_argument("--top_k", type=int, default=1,
                    help="从 --result 里取前 K 个候选逐个仿真（默认 1）")
    ap.add_argument("--f0", type=float, help="目标带通中心(用于画 target)")
    ap.add_argument("--bw", type=float, help="目标带宽")
    ap.add_argument("--target_s2p", help="目标 s2p")
    ap.add_argument("--free", action="store_true",
                    help="自由通带评估")
    ap.add_argument("--target_bw", type=float, default=0.5)
    ap.add_argument("--passband_db", type=float, default=-3.0)
    ap.add_argument("--stopband_db", type=float, default=None)
    ap.add_argument("--rolloff_db", type=float, default=None)
    ap.add_argument("--rolloff_ghz", type=float, default=0.15)
    ap.add_argument("--agg", choices=["mean", "max"], default="mean")
    ap.add_argument("--tag", default="validate")
    ap.add_argument("--no_reuse", action="store_true",
                    help="强制重新跑 HFSS，即使已有 s2p 文件")
    args = ap.parse_args()

    y_pred = None
    y_target = None
    free_pb = None
    topk_x = None
    topk_y_pred = None

    # 1) 从 result npz 读取
    if args.result:
        d = np.load(args.result, allow_pickle=True)
        x_real = d["x"]
        if "y_pred" in d.files:
            y_pred = d["y_pred"]
        if "topk_x" in d.files:
            topk_x = np.asarray(d["topk_x"])
        if "topk_y_pred" in d.files:
            topk_y_pred = np.asarray(d["topk_y_pred"])

        mode = str(d["mode"]) if "mode" in d.files else None
        if mode == "free_passband":
            free_pb = {
                "target_bw": float(d["target_bw"]),
                "passband_db": float(d["passband_db"]),
                "agg": "mean",
            }
            for k in ("stopband_db", "rolloff_db", "rolloff_ghz", "guard_ghz"):
                if k in d.files:
                    v = float(d[k])
                    if not np.isnan(v):
                        free_pb[k] = v
        elif "y_target" in d.files:
            y_target = d["y_target"]
    elif args.x:
        x_real = np.array(args.x, float)
    else:
        print("[错误] 请用 --x 或 --result 指定参数")
        sys.exit(1)

    # 2) CLI 覆盖目标（可选）
    if args.free:
        free_pb = {"target_bw": args.target_bw, "passband_db": args.passband_db,
                   "agg": args.agg}
        if args.stopband_db is not None:
            free_pb["stopband_db"] = args.stopband_db
        if args.rolloff_db is not None:
            free_pb["rolloff_db"] = args.rolloff_db
            free_pb["rolloff_ghz"] = args.rolloff_ghz

    if free_pb is None:
        if args.target_s2p:
            y_target, _ = spec_mod.target_from_s2p(args.target_s2p)
        elif args.f0 is not None and args.bw is not None:
            y_target, _ = spec_mod.make_bandpass_target(f0=args.f0, bw=args.bw)

    reuse = not args.no_reuse

    # 3) 分发：单点 or Top-K
    if args.top_k > 1 and topk_x is not None and len(topk_x) >= 2:
        K = min(args.top_k, len(topk_x))
        print(f"[validate] Top-K 模式: K={K} (--result 里有 {len(topk_x)} 个候选)")
        validate_topk(
            topk_x[:K],
            candidates_y_pred=topk_y_pred[:K] if topk_y_pred is not None else None,
            y_target=y_target, free_passband=free_pb,
            tag_prefix=args.tag, reuse=reuse,
        )
    else:
        info = evaluate_one(x_real, y_pred=y_pred, y_target=y_target,
                            free_passband=free_pb, tag=args.tag, reuse=reuse)
        print("\n========== 单点验证结果 ==========")
        print(f"约束满足     = {info['constraint_ok']} ({info['constraint_reason']})")
        if info["true_loss"] is not None:
            print(f"true_loss    = {info['true_loss']:.4f}")
        if info.get("mae_target") is not None:
            print(f"MAE(HFSS vs target, 全频段) = {info['mae_target']:.3f} dB")
        if info.get("mae_pass") is not None:
            print(f"MAE(HFSS vs target, 通带内) = {info['mae_pass']:.3f} dB")
        if info.get("mae_surrogate") is not None:
            print(f"MAE(HFSS vs surrogate)     = {info['mae_surrogate']:.3f} dB")
        if "diag" in info:
            print(format_passband_report(info["diag"], tag="HFSS真值"))
        print(f"s2p          = {info['s2p']}")
        if "plot" in info:
            print(f"对比图        = {info['plot']}")


if __name__ == "__main__":
    main()
