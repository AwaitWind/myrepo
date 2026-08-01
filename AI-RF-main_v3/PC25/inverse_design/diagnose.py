"""
数据集 + 代理模型「体检」脚本（只读，不改任何现有文件）。

用来定位「代理预测 S21 > 0 / 完全不准」到底是：
  (A) 训练数据本身就有坏样本（HFSS 仿真垃圾结果混入）
  (B) 数据干净但模型欠拟合 / 外推乱飞

用法:
    python diagnose.py                         # 体检默认 dataset.npz + forward_net.pt
    python diagnose.py --data id_data/dataset.npz --ckpt id_data/forward_net.pt
    python diagnose.py --bad_thresh 0.5        # S21/S11 超过该 dB 视为坏样本(默认 0.5)
"""

import argparse
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import id_config as C
from dataio import load_dataset


def _seg_stats(name, seg):
    """打印一段曲线矩阵 (N, N_FREQ) 的统计。返回坏样本掩码相关信息。"""
    finite = np.isfinite(seg)
    n_nan = int((~finite).sum())
    safe = np.where(finite, seg, np.nan)
    print(f"\n  [{name}] shape={seg.shape}")
    print(f"    min / max / mean = "
          f"{np.nanmin(safe):.3f} / {np.nanmax(safe):.3f} / {np.nanmean(safe):.3f} dB")
    print(f"    nan/inf 元素个数  = {n_nan}")
    return safe


def diagnose_data(npz_path, bad_thresh):
    print("=" * 70)
    print(f"[数据体检] {npz_path}")
    print("=" * 70)
    d = load_dataset(npz_path)
    X, Y = np.asarray(d["X"], float), np.asarray(d["Y"], float)
    N = X.shape[0]
    print(f"  样本数 N = {N}, X={X.shape}, Y={Y.shape}, include_s11={d['include_s11']}")

    s21 = Y[:, :C.N_FREQ]
    s11 = Y[:, C.N_FREQ:2 * C.N_FREQ] if d["include_s11"] else None

    _seg_stats("S21", s21)
    if s11 is not None:
        _seg_stats("S11", s11)

    # ---- 逐样本坏值判定 ----
    # 无源器件 |S| <= 0 dB。留一点数值误差余量 bad_thresh。
    s21_max_per = np.where(np.isfinite(s21), s21, -np.inf).max(axis=1)
    nan_row = ~np.isfinite(Y).all(axis=1)
    over_row = s21_max_per > bad_thresh
    if s11 is not None:
        s11_max_per = np.where(np.isfinite(s11), s11, -np.inf).max(axis=1)
        over_row = over_row | (s11_max_per > bad_thresh)
    bad_row = nan_row | over_row

    n_bad = int(bad_row.sum())
    print("\n  " + "-" * 60)
    print(f"  坏样本判定 (阈值: S21/S11 > {bad_thresh} dB 或含 nan/inf):")
    print(f"    含 nan/inf 样本数 = {int(nan_row.sum())}")
    print(f"    S21/S11 > {bad_thresh} dB 样本数 = {int(over_row.sum())}")
    print(f"    坏样本合计        = {n_bad} / {N}  ({100.0*n_bad/max(1,N):.1f}%)")

    if n_bad > 0:
        order = np.argsort(-s21_max_per)
        print("\n    S21 峰值最高的前 15 个样本（idx: S21_max dB）:")
        for i in order[:15]:
            flag = "  <-- 坏" if bad_row[i] else ""
            print(f"      #{i:5d}: S21_max = {s21_max_per[i]:8.3f} dB{flag}")

    # ---- 归一化统计会不会被坏样本带偏 ----
    y_mu = Y.mean(axis=0)
    y_std = Y.std(axis=0) + 1e-6
    if n_bad > 0:
        Yc = Y[~bad_row]
        if Yc.shape[0] > 0:
            mu_c = Yc.mean(axis=0)
            std_c = Yc.std(axis=0) + 1e-6
            print("\n  归一化统计对比（全量 vs 剔除坏样本后）:")
            print(f"    y_mu  绝对差 最大 = {np.abs(y_mu - mu_c).max():.3f} dB")
            print(f"    y_std 比值   最大 = {(y_std / std_c).max():.2f}x "
                  f"(远大于 1 说明坏样本把方差撑大了)")

    return d, bad_row


def diagnose_model(npz_path, ckpt_path, d, bad_row, bad_thresh):
    if not os.path.exists(ckpt_path):
        print(f"\n[模型体检] 跳过：未找到 ckpt {ckpt_path}")
        return
    print("\n" + "=" * 70)
    print(f"[模型体检] {ckpt_path}")
    print("=" * 70)
    from forward_net import ForwardModel

    fm = ForwardModel(ckpt_path)
    X, Y = np.asarray(d["X"], float), np.asarray(d["Y"], float)
    pred = fm.predict(X)

    def _mae(mask=None):
        p, y = (pred, Y) if mask is None else (pred[mask], Y[mask])
        if p.shape[0] == 0:
            return float("nan"), float("nan")
        mae = float(np.mean(np.abs(p - y)))
        mae_s21 = float(np.mean(np.abs(p[:, :C.N_FREQ] - y[:, :C.N_FREQ])))
        return mae, mae_s21

    tag = f"ensemble({len(fm.models)})" if fm.is_ensemble else "single"
    mae, mae_s21 = _mae()
    print(f"  [{tag}] 全量 MAE = {mae:.3f} dB | 仅S21 MAE = {mae_s21:.3f} dB (N={X.shape[0]})")

    good = ~bad_row
    if good.any() and (~good).any():
        mae_g, mae_s21_g = _mae(good)
        print(f"  仅在 [干净样本] 上: MAE = {mae_g:.3f} dB | S21 MAE = {mae_s21_g:.3f} dB "
              f"(N={int(good.sum())})")

    # 预测里有多少 S21 > 0（物理不可能）
    pred_s21 = pred[:, :C.N_FREQ]
    frac_over = float((pred_s21 > bad_thresh).mean())
    print(f"  预测 S21 > {bad_thresh} dB 的频点占比 = {100*frac_over:.2f}% "
          f"(理想为 0；>0 说明输出层无物理约束在外推)")
    print(f"  预测 S21 最大值 = {pred_s21.max():.3f} dB")


def main():
    ap = argparse.ArgumentParser(description="数据集 + 代理模型体检")
    ap.add_argument("--data", default=C.DATASET_NPZ)
    ap.add_argument("--ckpt", default=os.path.join(C.DATA_DIR, "forward_net.pt"))
    ap.add_argument("--bad_thresh", type=float, default=0.5,
                    help="S21/S11 超过该 dB 判为坏样本（默认 0.5，留数值误差余量）")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"[错误] 未找到数据集 {args.data}")
        sys.exit(1)

    d, bad_row = diagnose_data(args.data, args.bad_thresh)
    diagnose_model(args.data, args.ckpt, d, bad_row, args.bad_thresh)

    print("\n" + "=" * 70)
    print("[结论提示]")
    print("  · 若『坏样本合计』占比明显 (>1%)  → 数据污染是主因，先清洗再重训")
    print("  · 若数据干净但『预测 S21>0 占比』高 → 模型外推问题，需加输出物理约束")
    print("  · 若『干净样本上的 MAE』仍很大    → 欠拟合/容量或超参问题")
    print("=" * 70)


if __name__ == "__main__":
    main()
