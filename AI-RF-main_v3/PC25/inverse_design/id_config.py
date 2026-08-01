"""
inverse_design 的统一配置（single source of truth）。

针对 PC25 第三版滤波器（``HFSS_funcs.filter3_layout_generate``）建模的反向设计。
第三版结构：中心 CPW + 4 边矩形螺旋谐振器 + 中心叉指电容 + 可选多单元级联。

集中定义：
  - 9 个「可变」设计参数的范围
  - 若干「固定」结构参数（n, N, cpw_length, pcb_*, port_*）
  - 几何硬约束（对应 HFSS_funcs.filter3_layout_generate 里各条隐式假设）
  - 频率网格（与 HFSS_funcs.py 里第三版扫频完全一致：0.01 ~ 5.0 GHz，步长 0.1）
  - 路径、仿真核数等

所有其他脚本从这里导入常量，避免"参数对不上 HFSS 工程"的问题。
"""

import os

import numpy as np


# =====================================================================
# 1. PCB 与其他"固定"结构参数（与 main.py filter3_main 默认值一致）
# =====================================================================
PCB_X = 30.0              # PCB x 方向长度 (mm)
PCB_Y = 14.5              # PCB y 方向长度 (mm) —— 内部会被 N*p + 2*port_back + 2*cpw_length 覆盖
PCB_Z = 0.762             # 介质厚度 (mm)
PORT_BACK = 2.0           # 端口后侧地面 y 向延伸 (mm)
PORT_WIDTH = 0.15         # 集总端口 y 向厚度 (mm)
CENTER_FREQ_GHZ = 3.5     # init_hfss 的自适应网格中心频率

# 逆向设计中作为"超参数"固定的量（不进采样空间）：
N_FINGERS_FIXED = 10      # n = 叉指电容数量（改变阶数需要重新训代理）
N_CELLS_FIXED = 1         # N = 单元个数
CPW_LENGTH_FIXED = 2.0    # 两侧 CPW 馈线长度 (mm)


# =====================================================================
# 2. 可变设计参数（9 个，顺序固定）
#    与 HFSS_funcs.filter3_layout_generate 的对应关系见 param_to_kwargs()
#
#    w0/g/d0/lf 控制中心 CPW + 叉指电容；
#    a/b1/b2 控制 4 边矩形螺旋谐振器；
#    d/p0 控制 y 方向单元周期长度 p = d + 2*a + p0。
# =====================================================================
PARAM_NAMES = ["w0", "g", "d0", "lf", "a", "b1", "b2", "d", "p0"]
N_PARAMS = len(PARAM_NAMES)

# 各参数范围 (下界, 上界)，单位 mm
# 中心默认值来自 main.py filter3_main（w0=2.2, g=0.2, d0=0.6, lf=0.4,
# a=3.1, b1=0.3, b2=0.3, d=1.4, p0=0.5），上下界按 ±50%~2× 铺开。
PARAM_BOUNDS = {
    "w0": (1.0, 4.0),    # 中心 CPW 导带宽度
    "g":  (0.1, 0.5),    # CPW 中心导带两侧缝隙
    "d0": (0.3, 1.5),    # 中心间隔线长（叉指电容纵向可用长度）
    "lf": (0.15, 0.9),   # 叉指长度（受 d0 约束: lf < d0）
    "a":  (2.0, 5.0),    # 螺旋外中心线长（决定螺旋整体尺寸）
    "b1": (0.15, 0.6),   # 螺旋条带宽度
    "b2": (0.15, 0.6),   # 螺旋条带间距
    "d":  (1.0, 3.0),    # y 方向 gnd 半宽度参数
    "p0": (0.2, 1.5),    # y 方向单元额外间距
}

LOWER = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES], dtype=float)
UPPER = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES], dtype=float)


# =====================================================================
# 3. 几何硬约束（不满足则 HFSS 建模会自相交 / 越界 → 丢弃）
# =====================================================================
def param_dict(x):
    """N 维向量 → {名字: 值} 字典。"""
    return {name: float(v) for name, v in zip(PARAM_NAMES, x)}


def check_constraints(x, n=N_FINGERS_FIXED, pcb_x=PCB_X):
    """
    检查一组参数是否满足所有几何硬约束。返回 (ok: bool, reason: str)。

    约束来自 HFSS_funcs.filter3_layout_generate 的实际建模：
      A) 所有参数必须为正
      B) 叉指条带宽 w0 / (2n - 1) >= 0.02 mm
      C) 叉指长度必须能塞进中心间隔线: lf <= d0 - 0.05
      D) 螺旋生成合法性: a + b2 > b1/2 + 0.02
         （对应 get_rect_spiral_vertices 里 outer_center_len > 0）
      E) x 方向不越界: w0/2 + g + a + b2 <= pcb_x/2 - 0.5
         （左/右两侧螺旋各自在 x 方向的外沿）
      F) 螺旋条带间距至少要能容纳: b2 >= 0.05
    """
    p = param_dict(x)
    w0, g, d0, lf = p["w0"], p["g"], p["d0"], p["lf"]
    a, b1, b2 = p["a"], p["b1"], p["b2"]
    d, p0 = p["d"], p["p0"]

    # A) 所有正
    if min(w0, g, d0, lf, a, b1, b2, d, p0) <= 0:
        return False, "存在非正参数"

    # B) 叉指条带宽下限
    finger_w = w0 / (2 * n - 1)
    if finger_w < 0.02:
        return False, f"叉指条带宽 {finger_w:.4f} < 0.02 mm (w0={w0}, n={n})"

    # C) 叉指必须能塞入中心间隔
    if lf > d0 - 0.05:
        return False, f"lf={lf:.3f} > d0-0.05={d0-0.05:.3f}"

    # D) 螺旋外中心线长必须为正
    outer_center_len = a + b2 - b1 / 2.0
    if outer_center_len <= 0.02:
        return False, (f"螺旋无效: a+b2-b1/2={outer_center_len:.3f} <= 0.02 "
                       f"(a={a}, b1={b1}, b2={b2})")

    # E) x 方向不越界（左/右两侧螺旋对称，最外沿 = w0/2 + g + a + b2）
    x_half = w0 / 2.0 + g + a + b2
    if x_half > pcb_x / 2.0 - 0.5:
        return False, (f"x方向越界: w0/2+g+a+b2={x_half:.3f} > pcb_x/2-0.5"
                       f"={pcb_x/2-0.5:.3f}")

    # F) 螺旋条带间距下限（避免条带相互粘连）
    if b2 < 0.05:
        return False, f"b2={b2:.3f} < 0.05 mm（螺旋条带间距过小）"

    return True, "ok"


# =====================================================================
# 4. 参数向量 → filter3_layout_generate 关键字参数
#    统一在这里做「设计变量 + 固定超参」的组装
# =====================================================================
def param_to_kwargs(x):
    """把 N 维设计参数向量 x 转成 filter3_layout_generate 所需的完整 kwargs。"""
    p = param_dict(x)
    return dict(
        pcb_x=PCB_X, pcb_y=PCB_Y, pcb_z=PCB_Z,
        lf=p["lf"], n=N_FINGERS_FIXED, d=p["d"], a=p["a"],
        b1=p["b1"], b2=p["b2"], d0=p["d0"], g=p["g"],
        w0=p["w0"], p0=p["p0"],
        N=N_CELLS_FIXED, cpw_length=CPW_LENGTH_FIXED,
        port_width=PORT_WIDTH, port_back=PORT_BACK,
    )


# =====================================================================
# 5. 频率网格（必须与 HFSS_funcs 里的扫频完全一致）
#    第三版扫频：Frequency_Start=0.01, Frequency_Stop=5.0, Frequency_Step=0.1
#    → 共 51 个离散频点
# =====================================================================
FREQ_START_GHZ = 0.01
FREQ_STOP_GHZ = 5.0
FREQ_STEP_GHZ = 0.1
FREQ_GHZ = np.round(np.arange(FREQ_START_GHZ, FREQ_STOP_GHZ + 1e-9,
                              FREQ_STEP_GHZ), 4)
N_FREQ = len(FREQ_GHZ)

INCLUDE_S11 = True
Y_DIM = (2 if INCLUDE_S11 else 1) * N_FREQ

# 训练时 S 参数 dB 值的截断区间 [Y_DB_FLOOR, 0]：
#   上界 0  —— 无源器件 |S|<=1 → dB<=0，是硬物理上界；
#   下界 floor —— 深阻带低于该值对滤波器设计无实际意义，却会撑大逐维方差、
#                 抢占网络容量、放大 MSE，反而拖累通带/边缘拟合精度。
# clip 到该区间可显著提升代理在通带/过渡带的精度。
Y_DB_FLOOR = -60.0


# =====================================================================
# 6. 路径与仿真设置
# =====================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_THIS_DIR, "id_data")
S2P_DIR = os.path.join(DATA_DIR, "s2p")
DATASET_NPZ = os.path.join(DATA_DIR, "dataset.npz")
PARAMS_NPY = os.path.join(DATA_DIR, "params.npy")
INDEX_CSV = os.path.join(DATA_DIR, "index.csv")

# HFSS 仿真核数（按你机器实际算力调整）
SIM_CORES = 32
SIM_TASKS = 2

# 数据生成默认样本数
DEFAULT_N_SAMPLES = 300

# 采样随机种子（固定 → 断点续跑时采样点不变）
SAMPLING_SEED = 42


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(S2P_DIR, exist_ok=True)


if __name__ == "__main__":
    print("inverse_design / id_config (第三版 filter3)")
    print(f"  参数顺序 ({N_PARAMS} 维): {PARAM_NAMES}")
    print(f"  下界 LOWER: {LOWER}")
    print(f"  上界 UPPER: {UPPER}")
    print(f"  固定超参: n={N_FINGERS_FIXED}, N={N_CELLS_FIXED}, "
          f"cpw_length={CPW_LENGTH_FIXED}")
    print(f"  频率网格: {FREQ_START_GHZ}~{FREQ_STOP_GHZ} GHz, {N_FREQ} 点, "
          f"步长={FREQ_STEP_GHZ}")
    print(f"  y 向量维度 Y_DIM = {Y_DIM}  (INCLUDE_S11={INCLUDE_S11})")
    print(f"  数据目录 DATA_DIR = {DATA_DIR}")
