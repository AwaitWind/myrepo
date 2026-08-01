"""
数据 I/O：
  - 解析 Touchstone (.s2p) → S11/S21 dB 曲线
  - 线性插值对齐到统一频率网格 FREQ_GHZ
  - 组装成定长 y 向量
  - 数据集（dataset.npz）与索引（index.csv）的读写

y 向量约定（见 id_config.INCLUDE_S11）：
  INCLUDE_S11=True  → y = [S21_dB(N_FREQ), S11_dB(N_FREQ)]，长度 2*N_FREQ
  INCLUDE_S11=False → y = [S21_dB(N_FREQ)]
"""

import csv
import os

import numpy as np

from id_config import (
    FREQ_GHZ,
    N_FREQ,
    INCLUDE_S11,
    Y_DIM,
    PARAM_NAMES,
    DATASET_NPZ,
    INDEX_CSV,
)


# =====================================================================
# 1. s2p 解析
# =====================================================================
def parse_s2p(s2p_path):
    """
    解析 Touchstone .s2p，返回 (freqs_ghz, s11_db, s21_db)。
    优先 scikit-rf；缺失回退手写 MA 解析。
    """
    try:
        import skrf as rf
        nw = rf.Network(s2p_path)
        f_ghz = nw.f / 1e9
        s11 = nw.s[:, 0, 0]
        s21 = nw.s[:, 1, 0] if nw.nports >= 2 else np.zeros_like(s11)
        s11_db = 20.0 * np.log10(np.abs(s11) + 1e-12)
        s21_db = 20.0 * np.log10(np.abs(s21) + 1e-12)
        return np.asarray(f_ghz, float), np.asarray(s11_db, float), np.asarray(s21_db, float)
    except ImportError:
        return _parse_s2p_ma(s2p_path)


def _parse_s2p_ma(s2p_path):
    """手写 Touchstone MA 解析（回退）。"""
    freqs, s11_db, s21_db = [], [], []
    unit_scale = 1e9

    with open(s2p_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                tokens = line.lower().split()
                if "hz" in tokens:
                    unit_scale = 1.0
                elif "khz" in tokens:
                    unit_scale = 1e3
                elif "mhz" in tokens:
                    unit_scale = 1e6
                elif "ghz" in tokens:
                    unit_scale = 1e9
                continue

            parts = line.split()
            if len(parts) < 9:
                continue
            freq_hz = float(parts[0]) * unit_scale
            s11_mag = float(parts[1])
            s21_mag = float(parts[3])
            freqs.append(freq_hz / 1e9)
            s11_db.append(20.0 * np.log10(s11_mag + 1e-12))
            s21_db.append(20.0 * np.log10(s21_mag + 1e-12))

    if not freqs:
        raise ValueError(f"未从 {s2p_path} 读到有效数据")
    return np.array(freqs), np.array(s11_db), np.array(s21_db)


# =====================================================================
# 2. 对齐到统一频率网格
# =====================================================================
def resample_to_grid(src_freqs, src_values, grid=FREQ_GHZ):
    src_freqs = np.asarray(src_freqs, float)
    src_values = np.asarray(src_values, float)

    if len(src_freqs) == len(grid) and np.allclose(src_freqs, grid, atol=1e-6):
        return src_values.copy()

    order = np.argsort(src_freqs)
    return np.interp(grid, src_freqs[order], src_values[order])


def s2p_to_y(s2p_path):
    """解析 s2p → 定长 y 向量（已对齐到 FREQ_GHZ）。"""
    f, s11_db, s21_db = parse_s2p(s2p_path)
    s21_g = resample_to_grid(f, s21_db)
    if INCLUDE_S11:
        s11_g = resample_to_grid(f, s11_db)
        y = np.concatenate([s21_g, s11_g])
    else:
        y = s21_g
    assert y.shape[0] == Y_DIM, f"y 维度 {y.shape[0]} != Y_DIM {Y_DIM}"
    return y


def split_y(y):
    """把 y 向量拆回 (s21_db, s11_db)；若不含 S11 则 s11_db 为 None。"""
    y = np.asarray(y, float)
    s21_db = y[:N_FREQ]
    s11_db = y[N_FREQ:2 * N_FREQ] if INCLUDE_S11 else None
    return s21_db, s11_db


# =====================================================================
# 3. 索引 CSV（断点续跑用）
# =====================================================================
INDEX_HEADER = ["idx", "status", "s2p", "sim_sec"] + PARAM_NAMES


def append_index_row(idx, status, s2p, sim_sec, x, csv_path=INDEX_CSV):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(INDEX_HEADER)
        w.writerow([idx, status, s2p, f"{sim_sec:.2f}"] + [f"{v:.6f}" for v in x])


def load_done_indices(csv_path=INDEX_CSV):
    """已成功（status=='ok'）的 idx 集合，用于断点续跑。"""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("status") == "ok":
                try:
                    done.add(int(row["idx"]))
                except (ValueError, KeyError):
                    pass
    return done


def load_attempts(csv_path=INDEX_CSV):
    """每个 idx 的尝试次数（status=='try' 的行数）；用于毒样本熔断。"""
    counts = {}
    if not os.path.exists(csv_path):
        return counts
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("status") == "try":
                try:
                    idx = int(row["idx"])
                    counts[idx] = counts.get(idx, 0) + 1
                except (ValueError, KeyError):
                    pass
    return counts


# =====================================================================
# 4. 聚合数据集 npz 读写
# =====================================================================
def save_dataset(X, Y, npz_path=DATASET_NPZ):
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(
        npz_path,
        X=np.asarray(X, float),
        Y=np.asarray(Y, float),
        freqs=FREQ_GHZ,
        param_names=np.array(PARAM_NAMES),
        include_s11=np.array([INCLUDE_S11]),
    )


def load_dataset(npz_path=DATASET_NPZ):
    d = np.load(npz_path, allow_pickle=True)
    return {
        "X": d["X"],
        "Y": d["Y"],
        "freqs": d["freqs"],
        "param_names": list(d["param_names"]),
        "include_s11": bool(d["include_s11"][0]),
    }


if __name__ == "__main__":
    print("inverse_design / dataio (filter5)")
    print(f"  统一频率网格点数 N_FREQ = {N_FREQ}")
    print(f"  y 向量维度 Y_DIM = {Y_DIM} (INCLUDE_S11={INCLUDE_S11})")
