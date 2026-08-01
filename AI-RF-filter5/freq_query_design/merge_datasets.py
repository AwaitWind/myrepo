"""merge_datasets.py —— 把多个 dataset.npz 合并成一个（校验 + 去重 + 清洗 + 压缩）。

前提：每个输入 npz 内部的 X 与 Y 是**一一对应**的（即每个 .npz 各自是一次干净聚合）。
      若你是在同一个 id_data 里先 --n 2000 再 --n 8000 且没备份 params.npy，
      请先跑 `python inspect_data.py` 确认没有 X/Y 错配，再来合并。

用法：
    python merge_datasets.py <输出.npz> <输入1.npz> <输入2.npz> [更多...]

例（合并两批后直接覆盖成训练用 dataset.npz）：
    python merge_datasets.py id_data/dataset.npz \
        id_data/dataset_2000.npz id_data/dataset_8000.npz
    python forward_net.py --train --ensemble 5 --epochs 800

输出 schema 与 dataio.save_dataset 完全一致（X, Y, freqs, param_names, include_s11），
可被 forward_net.py --train 直接读取。
"""
import argparse
import os
import sys

import numpy as np


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    if "X" not in d.files or "Y" not in d.files:
        raise KeyError(f"{path} 缺少 X/Y 键，不是有效 dataset.npz")
    X = np.asarray(d["X"], float)
    Y = np.asarray(d["Y"], float)
    freqs = np.asarray(d["freqs"], float) if "freqs" in d.files else None
    pn = list(d["param_names"]) if "param_names" in d.files else None
    inc = bool(np.asarray(d["include_s11"]).ravel()[0]) if "include_s11" in d.files else None
    return X, Y, freqs, pn, inc


def main():
    ap = argparse.ArgumentParser(description="合并多个 dataset.npz → 一个（去重+校验+清洗）")
    ap.add_argument("out", help="输出 npz 路径")
    ap.add_argument("inputs", nargs="+", help="一个或多个输入 dataset.npz")
    ap.add_argument("--dedup-decimals", type=int, default=6,
                    help="按 X 去重时四舍五入到的小数位（默认 6）")
    ap.add_argument("--no-dedup", action="store_true", help="不按 X 去重")
    args = ap.parse_args()

    Xs, Ys = [], []
    ref = {"pdim": None, "ydim": None, "freqs": None, "pn": None, "inc": None}
    total_in = 0

    for p in args.inputs:
        if not os.path.exists(p):
            print(f"[跳过] 不存在: {p}")
            continue
        X, Y, freqs, pn, inc = load_npz(p)
        if X.shape[0] != Y.shape[0]:
            print(f"[错误] {p}: X 行数 {X.shape[0]} != Y 行数 {Y.shape[0]}，跳过")
            continue

        # 以第一个有效文件为基准，校验后续文件维度/频率网格一致
        if ref["pdim"] is None:
            ref.update(pdim=X.shape[1], ydim=Y.shape[1], freqs=freqs, pn=pn, inc=inc)
        else:
            if X.shape[1] != ref["pdim"]:
                print(f"[错误] {p}: 参数维 {X.shape[1]} != 基准 {ref['pdim']}，跳过")
                continue
            if Y.shape[1] != ref["ydim"]:
                print(f"[错误] {p}: Y 维 {Y.shape[1]} != 基准 {ref['ydim']}，跳过")
                continue
            if (ref["freqs"] is not None and freqs is not None
                    and not np.allclose(freqs, ref["freqs"], atol=1e-6)):
                print(f"[警告] {p}: 频率网格与基准不一致！合并结果可能无意义")

        # 清洗：丢弃 X/Y 含 NaN/Inf 的样本
        bad = ~np.isfinite(Y).all(axis=1) | ~np.isfinite(X).all(axis=1)
        if bad.any():
            print(f"[清洗] {p}: 丢弃 {int(bad.sum())} 个 NaN/Inf 坏样本")
            X, Y = X[~bad], Y[~bad]

        print(f"[加载] {p}: {X.shape[0]} 个样本  (X{X.shape}, Y{Y.shape})")
        total_in += X.shape[0]
        Xs.append(X)
        Ys.append(Y)

    if not Xs:
        print("[错误] 没有可用输入文件")
        sys.exit(1)

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    print(f"\n[拼接] 合计 {X.shape[0]} 个样本")

    if not args.no_dedup:
        key = np.round(X, args.dedup_decimals)
        _, uniq_idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        n_dup = X.shape[0] - len(uniq_idx)
        if n_dup > 0:
            # 完整性提示：若重复 X 对应的 Y 差异很大，多半是不同批次结果打架/错配
            worst = 0.0
            for g in range(len(uniq_idx)):
                rows = np.where(inv == g)[0]
                if rows.size > 1:
                    worst = max(worst, float(np.abs(Y[rows] - Y[rows[0]]).max()))
            print(f"[去重] {n_dup} 个重复 X → 保留 {len(uniq_idx)} 个唯一样本")
            if worst > 1.0:
                print(f"[警告] 重复 X 之间 Y 最大差异 {worst:.2f} dB（>1dB）——"
                      f"可能有批次结果不一致/错配，建议先跑 inspect_data.py 核查")
        keep = np.sort(uniq_idx)
        X, Y = X[keep], Y[keep]

    save_kw = {"X": X, "Y": Y}
    if ref["freqs"] is not None:
        save_kw["freqs"] = ref["freqs"]
    if ref["pn"] is not None:
        save_kw["param_names"] = np.array(ref["pn"])
    if ref["inc"] is not None:
        save_kw["include_s11"] = np.array([ref["inc"]])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **save_kw)
    print(f"\n[完成] 输入合计 {total_in} → 去重清洗后 {X.shape[0]} 个 → {args.out}")
    print(f"        X{X.shape}  Y{Y.shape}")
    print(f"下一步: python forward_net.py --train --ensemble 5 --epochs 800 --data {args.out}")


if __name__ == "__main__":
    main()
