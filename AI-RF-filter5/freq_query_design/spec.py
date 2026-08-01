"""
目标响应（spec）—— 把「人能描述的效果」转成逆向求解的目标曲线 y* + 权重 w。

【本版重构核心改动】：
  - transition_ghz 默认 0.4 → 0.15（过渡带更窄，逼出陡下降沿）
  - trans_weight   默认 0.0 → 0.5（过渡带不再"免罚"，鼓励贴近理想过渡曲线）
  - 新增 rolloff_penalty(...)：给固定 target 模式额外加一个显式的"下降沿"惩罚项

两种目标来源：
  1. make_bandpass_target(...)  高层带通规格 → 目标 S21 曲线模板
  2. target_from_s2p(path)      直接拿一条已有的理想 s2p 当目标

注：filter5 频率网格 0.01~9.91 GHz / 100 点（步长=0.1 GHz），
   transition_ghz / rolloff_ghz 会按 C.FREQ_STEP_GHZ 自动折算成格点数。
"""

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import id_config as C
from dataio import s2p_to_y, split_y


# =====================================================================
# 1. 高层带通规格 → 目标曲线
# =====================================================================
def make_bandpass_target(
    f0,
    bw,
    passband_db=-0.5,
    stopband_db=-30.0,
    transition_ghz=0.15,       # ← 默认从 0.4 收窄到 0.15
    s11_passband_db=-15.0,
    s11_other_db=-1.0,
    pass_weight=3.0,
    stop_weight=2.0,
    trans_weight=0.5,          # ← 默认从 0 提升到 0.5，过渡带被约束
    s11_weight=0.5,
):
    """
    构造带通滤波器目标。

    参数（GHz / dB）：
      f0              通带中心频率
      bw              通带带宽（通带 = [f0-bw/2, f0+bw/2]）
      passband_db     通带目标 S21（接近 0，如 -0.5）
      stopband_db     阻带目标 S21（如 -30）
      transition_ghz  过渡带宽度（通带两侧各 transition_ghz）
      s11_passband_db 通带内 S11 目标
      s11_other_db    通带外 S11 目标
      *_weight        各区段在 loss 中的权重（trans_weight > 0 时会约束下降沿）

    返回: (y_target, w) 两个长度 Y_DIM 的 np.ndarray。
    """
    f = C.FREQ_GHZ
    f_lo = f0 - bw / 2.0
    f_hi = f0 + bw / 2.0

    # ---- S21 目标模板 ----
    s21 = np.full(C.N_FREQ, stopband_db, dtype=float)
    w21 = np.full(C.N_FREQ, stop_weight, dtype=float)

    pass_mask = (f >= f_lo) & (f <= f_hi)
    s21[pass_mask] = passband_db
    w21[pass_mask] = pass_weight

    left_trans = (f >= f_lo - transition_ghz) & (f < f_lo)
    right_trans = (f > f_hi) & (f <= f_hi + transition_ghz)
    if transition_ghz > 0:
        s21[left_trans] = np.interp(
            f[left_trans], [f_lo - transition_ghz, f_lo], [stopband_db, passband_db]
        )
        s21[right_trans] = np.interp(
            f[right_trans], [f_hi, f_hi + transition_ghz], [passband_db, stopband_db]
        )
    w21[left_trans] = trans_weight
    w21[right_trans] = trans_weight

    if not C.INCLUDE_S11:
        return s21, w21

    s11 = np.full(C.N_FREQ, s11_other_db, dtype=float)
    s11[pass_mask] = s11_passband_db
    w11 = np.full(C.N_FREQ, s11_weight, dtype=float)

    y_target = np.concatenate([s21, s11])
    w = np.concatenate([w21, w11])
    return y_target, w


# =====================================================================
# 2. 从已有 s2p 提取目标
# =====================================================================
def target_from_s2p(s2p_path, pass_weight=3.0, stop_weight=2.0, s11_weight=0.5,
                    passband_thresh_db=-3.0):
    """
    用一条现有 s2p 作为目标曲线。权重按「该曲线自身的通带/阻带」自动划分：
      S21 >= passband_thresh_db 视为通带 → pass_weight
      其余 → stop_weight
    """
    y = s2p_to_y(s2p_path)
    s21, s11 = split_y(y)

    w21 = np.where(s21 >= passband_thresh_db, pass_weight, stop_weight)
    if not C.INCLUDE_S11:
        return s21.copy(), w21

    w11 = np.full(C.N_FREQ, s11_weight, dtype=float)
    w = np.concatenate([w21, w11])
    return y.copy(), w


# =====================================================================
# 3. 下降系数（skirt / rolloff）惩罚
# =====================================================================
def rolloff_penalty(
    s21_db,
    f0, bw,
    rolloff_ghz=0.15,
    rolloff_db=-15.0,
    weight=2.0,
):
    """
    显式的下降系数约束（供 inverse_optimize 加到 objective 上）。

    在通带边缘外 `rolloff_ghz` 处放两个哨兵频点：要求 S21 <= rolloff_db。
    每个哨兵超出量线性罚，最后乘 weight。

    与 make_bandpass_target 的 trans_weight 不同：
      - trans_weight 是「过渡带按目标曲线拟合」，容忍度大
      - rolloff_penalty 是「哨兵点必须掉到指定电平以下」，硬约束"下降沿要陡"

    返回：float（>=0，越大说明下降沿越平缓）。
    """
    s21 = np.asarray(s21_db, float)
    f = C.FREQ_GHZ
    f_lo = f0 - bw / 2.0
    f_hi = f0 + bw / 2.0

    def _nearest(fx):
        return int(np.clip(np.argmin(np.abs(f - fx)), 0, len(f) - 1))

    i_left = _nearest(f_lo - rolloff_ghz)
    i_right = _nearest(f_hi + rolloff_ghz)

    pen = 0.0
    n_taken = 0
    if 0 <= i_left < len(f):
        pen += max(0.0, s21[i_left] - rolloff_db)
        n_taken += 1
    if 0 <= i_right < len(f):
        pen += max(0.0, s21[i_right] - rolloff_db)
        n_taken += 1
    if n_taken > 0:
        pen /= n_taken
    return weight * pen


def rolloff_penalty_torch(
    s21_db, f0, bw,
    rolloff_ghz=0.15, rolloff_db=-15.0, weight=2.0,
):
    """torch 可微版本（供梯度逆向）。"""
    import torch
    f = C.FREQ_GHZ
    f_lo = f0 - bw / 2.0
    f_hi = f0 + bw / 2.0

    def _nearest(fx):
        return int(np.clip(np.argmin(np.abs(f - fx)), 0, len(f) - 1))

    i_left = _nearest(f_lo - rolloff_ghz)
    i_right = _nearest(f_hi + rolloff_ghz)

    vals, cnt = 0.0, 0
    if 0 <= i_left < len(f):
        vals = vals + torch.relu(s21_db[i_left] - rolloff_db)
        cnt += 1
    if 0 <= i_right < len(f):
        vals = vals + torch.relu(s21_db[i_right] - rolloff_db)
        cnt += 1
    if cnt > 0:
        vals = vals / cnt
    return weight * vals


# =====================================================================
# 4. 工具：把目标曲线画出来看看（不依赖 HFSS）
# =====================================================================
def plot_target(y_target, w=None, out_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s21, s11 = split_y(y_target)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(C.FREQ_GHZ, s21, label="target S21 (dB)", color="#F44336", lw=1.8)
    if s11 is not None:
        ax.plot(C.FREQ_GHZ, s11, label="target S11 (dB)", color="#2196F3", lw=1.2)
    if w is not None:
        ws21, _ = split_y(w) if w.shape[0] == C.Y_DIM else (w, None)
        ax.fill_between(C.FREQ_GHZ, -60, 0, where=(ws21 >= ws21.max() - 1e-9),
                        color="orange", alpha=0.12, label="high-weight (passband)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S| (dB)")
    ax.set_ylim(-60, 2)
    ax.set_xlim(C.FREQ_GHZ[0], C.FREQ_GHZ[-1])
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend()
    ax.set_title("Inverse-design target spec (filter5)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"[spec] 目标曲线已保存 → {out_path}")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    y, w = make_bandpass_target(f0=3.5, bw=0.5)
    print(f"[spec] 生成带通目标: Y_DIM={y.shape[0]}, "
          f"通带点数={(w[:C.N_FREQ]==w[:C.N_FREQ].max()).sum()}, "
          f"trans_weight={0.5} transition={0.15} GHz")
    out = os.path.join(C.DATA_DIR, "target_preview.png")
    C.ensure_dirs()
    try:
        plot_target(y, w, out)
    except Exception as e:
        print(f"[spec] 预览图绘制跳过: {e}")
