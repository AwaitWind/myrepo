"""
inverse_design 的统一配置（single source of truth）。

【本版参数减维（对称约束）】
  - wc1 = wc5          → 1 变量 `wc_end`  （输入/输出中心线宽对称）
  - wc2 = wc3 = wc4    → 1 变量 `wc_res`  （3 谐振段中心线宽相同）
  - l1  = l5           → 1 变量 `l_end`   （输入/输出侧对称）
  - h1  = h2 = h3      → 1 变量 `h_res`   （3 对谐振器等宽）
  - l2  = l3 = l4      → 1 变量 `l_res`   （3 谐振段等长）

原始 19 维 → 减少到 **11 维**，搜索空间小 ~42%。

其他脚本（forward_net/inverse_optimize/validate）都通过 C.N_PARAMS / C.PARAM_NAMES
自适应，不需改。HFSS 建模函数 filter_layout_generate 仍需 19 个参数，
由 _expand_to_full() 从 11 维紧凑向量还原成完整 19 参数字典。

集中定义：
  - 13 个「可变」设计参数的范围
  - 若干「固定」结构参数（PCB_X, PCB_Z）
  - 几何硬约束（对应 HFSS_funcs.filter_layout_generate 里的隐式假设）
  - 频率网格（0.01 ~ 9.91 GHz, 100 点，与 HFSS_funcs 扫频完全一致）
  - 路径、仿真核数等
"""

import os
import numpy as np


# =====================================================================
# 1. PCB 与其他"固定"结构参数（与 main.py 默认值一致）
# =====================================================================
PCB_X = 17.5           # PCB x 方向长度 (mm)
PCB_Z = 0.762          # 介质厚度 (mm)
CENTER_FREQ_GHZ = 3.5  # init_hfss 的自适应网格中心频率


# =====================================================================
# 2. 可变设计参数（13 个紧凑维度，顺序固定）
#
#    与 filter_layout_generate 的 19 参数映射见 _expand_to_full()
#
#    - l_end   : l1 = l5           输入/输出侧进线长度（对称）
#    - l_res   : l2 = l3 = l4      3 个谐振段沿 y 方向长度（对称）
#    - w1..w4  : （独立）           段间距
#    - wc_end  : wc1 = wc5         输入/输出段中心导带宽度（对称）
#    - wc_res  : wc2 = wc3 = wc4   3 谐振段中心导带宽度（对称）
#    - h_res   : h1 = h2 = h3      3 对矩形谐振器水平延伸（对称）
#    - hd      : via 打孔直径
#    - hs      : 相邻 via 边缘间距
# =====================================================================
PARAM_NAMES = [
    "l_end",         # l1 == l5
    "l_res",         # l2 == l3 == l4
    "w1", "w2", "w3", "w4",
    "wc_end",        # wc1 == wc5
    "wc_res",        # wc2 == wc3 == wc4
    "h_res",         # h1 == h2 == h3
    "hd", "hs",
]
N_PARAMS = len(PARAM_NAMES)   # = 11

# 参数默认值（来自 main.py 硬编码值，对应对称组的公共值）
PARAM_DEFAULTS = {
    "l_end": 5.25,       # l1=l5=5.25
    "l_res": 5.20,       # l2=l3=l4=5.20
    "w1": 1.00, "w2": 2.00, "w3": 2.00, "w4": 1.00,
    "wc_end": 1.00,      # wc1=wc5=1.00
    "wc_res": 1.00,      # wc2=wc3=wc4=1.00
    "h_res": 5.00,       # h1=h2=h3=5.00
    "hd": 0.80,
    "hs": 0.50,
}

# 参数范围 (下界, 上界)，单位 mm —— 默认值 ±30%
PARAM_BOUNDS = {
    "l_end":  (3.68, 6.83),
    "l_res":  (3.64, 6.76),
    "w1":  (0.70, 1.30),
    "w2":  (1.40, 2.60),
    "w3":  (1.40, 2.60),
    "w4":  (0.70, 1.30),
    "wc_end": (0.70, 1.30),
    "wc_res": (0.70, 1.30),
    "h_res":  (3.50, 6.50),
    "hd":  (0.56, 1.04),
    "hs":  (0.35, 0.65),
}

LOWER = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES], dtype=float)
UPPER = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES], dtype=float)


# =====================================================================
# 3. 展开：紧凑 11 维 → 完整 19 参数字典
# =====================================================================
def param_dict(x):
    """N 维紧凑向量 → {紧凑名字: 值} 字典。"""
    return {name: float(v) for name, v in zip(PARAM_NAMES, x)}


def _expand_to_full(x):
    """
    把 11 维紧凑向量展开成 filter_layout_generate 需要的完整 19 参数字典
    （名字对应 HFSS 函数签名）：
        {l1, l2, l3, l4, l5, w1..w4, wc1..wc5, h1, h2, h3, hd, hs}
    """
    p = param_dict(x)
    return {
        # 对称组展开
        "l1": p["l_end"],  "l5": p["l_end"],
        "l2": p["l_res"],  "l3": p["l_res"],  "l4": p["l_res"],
        "w1": p["w1"], "w2": p["w2"], "w3": p["w3"], "w4": p["w4"],
        "wc1": p["wc_end"], "wc5": p["wc_end"],
        "wc2": p["wc_res"], "wc3": p["wc_res"], "wc4": p["wc_res"],
        "h1": p["h_res"], "h2": p["h_res"], "h3": p["h_res"],
        "hd": p["hd"], "hs": p["hs"],
    }


# =====================================================================
# 4. 几何硬约束（不满足则 HFSS 建模会自相交 / 越界 → 丢弃）
# =====================================================================
_EDGE_MARGIN = 0.2   # 谐振器到 PCB 边缘的最小留白（mm）


def check_constraints(x, pcb_x=PCB_X):
    """
    检查一组参数是否满足所有几何硬约束。返回 (ok: bool, reason: str)。

    约束来自 HFSS_funcs.filter_layout_generate 的隐式建模假设：
      A) 所有参数 > 0
      B) 3 对矩形谐振器 x 方向不越出 PCB 边缘 + 留 0.2mm 安全边距：
           wc2/2 + h1 <= PCB_X/2 - 0.2
           wc3/2 + h2 <= PCB_X/2 - 0.2
           wc4/2 + h3 <= PCB_X/2 - 0.2
         （由于 wc2=wc3=wc4=wc_res 且 h1=h2=h3=h_res，等价于单条约束
            wc_res/2 + h_res <= PCB_X/2 - 0.2）
      C) via 阵列 x 位置合理（via 中心在 pcb_x/2 - hd 处），需保证：
           pcb_x/2 - hd - hd/2 >= max(wc1/2, wc5/2, wc2/2+h1, ...) + 0.1
      D) via 阵列长度合法（get_circle_layout_info 要求）：
           l2 + l3 + l4 + w1 + w2 + w3 + w4 >= hd（至少能放一个 via）
    """
    # 展开成完整字典（用原始 filter_layout_generate 的参数名，方便直接对照 HFSS 建模）
    full = _expand_to_full(x)

    # A) 全部 > 0
    for name, v in full.items():
        if v <= 0:
            return False, f"参数 {name}={v:.4f} <= 0"

    # B) 三对谐振器不越界（因对称约束，其实只需检查一次，但保留循环便于扩展）
    x_limit = pcb_x / 2.0 - _EDGE_MARGIN
    for i, (wc_key, h_key) in enumerate([("wc2", "h1"), ("wc3", "h2"), ("wc4", "h3")], start=1):
        x_used = full[wc_key] / 2.0 + full[h_key]
        if x_used > x_limit:
            return False, (f"第 {i} 对谐振器 x 越界: {wc_key}/2+{h_key}={x_used:.3f} "
                           f"> PCB_X/2-{_EDGE_MARGIN}={x_limit:.3f}")

    # C) via 不覆盖中心导带 / 谐振器
    via_outer_x = pcb_x / 2.0 - full["hd"] / 2.0
    max_center_x = max(
        full["wc1"] / 2.0, full["wc5"] / 2.0,
        full["wc2"] / 2.0 + full["h1"],
        full["wc3"] / 2.0 + full["h2"],
        full["wc4"] / 2.0 + full["h3"],
    )
    if via_outer_x <= max_center_x + 0.1:
        return False, (f"via 阵列与中心导体重叠: via 内边 x={via_outer_x:.3f} "
                       f"<= 中心结构最外 x={max_center_x:.3f} + 0.1")

    # D) via 阵列的 y 方向长度必须能放下至少 1 个 via
    via_line_len = (full["l2"] + full["l3"] + full["l4"]
                    + full["w1"] + full["w2"] + full["w3"] + full["w4"])
    if via_line_len < full["hd"]:
        return False, (f"via 阵列 y 长度 {via_line_len:.3f} < hd={full['hd']:.3f}，"
                       "无法放下任何 via")

    return True, "ok"


# =====================================================================
# 5. 参数向量 → filter_layout_generate 关键字参数
# =====================================================================
def param_to_kwargs(x):
    """把 11 维紧凑设计参数向量 x 转成 filter_layout_generate 所需的完整 kwargs。"""
    full = _expand_to_full(x)
    # pcb_y 由 filter_layout_generate 内部重算，这里传占位（实际会被覆盖）
    pcb_y_placeholder = (
        full["l1"] + full["l2"] + full["l3"] + full["l4"] + full["l5"]
        + full["w1"] + full["w2"] + full["w3"] + full["w4"]
    )
    return dict(
        pcb_x=PCB_X, pcb_y=pcb_y_placeholder, pcb_z=PCB_Z,
        **full,
    )


# =====================================================================
# 6. 频率网格（必须与 HFSS_funcs 的扫频完全一致）
#    HFSS_funcs.py: Frequency_Start=0.01, Frequency_Stop=10.0, Frequency_Step=0.1
#    → np.round(np.arange(0.01, 10.0+1e-9, 0.1), 2) = [0.01, 0.11, ..., 9.91]
#    共 100 点，步长 0.1 GHz
# =====================================================================
_FREQ_START_RAW = 0.01
_FREQ_STOP_RAW = 10.0
FREQ_STEP_GHZ = 0.1
FREQ_GHZ = np.round(np.arange(_FREQ_START_RAW, _FREQ_STOP_RAW + 1e-9, FREQ_STEP_GHZ), 2)
N_FREQ = int(len(FREQ_GHZ))                       # = 100
FREQ_START_GHZ = float(FREQ_GHZ[0])               # 0.01
FREQ_STOP_GHZ = float(FREQ_GHZ[-1])               # 9.91

INCLUDE_S11 = True
Y_DIM = (2 if INCLUDE_S11 else 1) * N_FREQ         # = 200


# =====================================================================
# 7. 路径与仿真设置
# =====================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_THIS_DIR, "id_data")
S2P_DIR = os.path.join(DATA_DIR, "s2p")
DATASET_NPZ = os.path.join(DATA_DIR, "dataset.npz")
PARAMS_NPY = os.path.join(DATA_DIR, "params.npy")
INDEX_CSV = os.path.join(DATA_DIR, "index.csv")

SIM_CORES = 8
SIM_TASKS = 2
DEFAULT_N_SAMPLES = 300
SAMPLING_SEED = 42


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(S2P_DIR, exist_ok=True)


if __name__ == "__main__":
    print("inverse_design / id_config (filter5, 减维版 11 维)")
    print(f"  紧凑参数顺序 ({N_PARAMS} 维): {PARAM_NAMES}")
    print(f"  下界 LOWER: {np.round(LOWER, 3).tolist()}")
    print(f"  上界 UPPER: {np.round(UPPER, 3).tolist()}")
    print(f"  默认值 (来自 main.py 对称组): {[PARAM_DEFAULTS[k] for k in PARAM_NAMES]}")
    print(f"  PCB: X={PCB_X} mm, Z={PCB_Z} mm, PCB_Y=动态(参数决定)")
    print(f"  频率网格: {FREQ_START_GHZ}~{FREQ_STOP_GHZ} GHz, {N_FREQ} 点, "
          f"步长={FREQ_STEP_GHZ} GHz")
    print(f"  y 向量维度 Y_DIM = {Y_DIM}  (INCLUDE_S11={INCLUDE_S11})")
    print(f"  数据目录 DATA_DIR = {DATA_DIR}")

    # 展开示例
    x_default = np.array([PARAM_DEFAULTS[k] for k in PARAM_NAMES])
    full = _expand_to_full(x_default)
    print(f"\n  11 维默认 → 完整 19 参数展开:")
    for k in ["l1", "l2", "l3", "l4", "l5", "w1", "w2", "w3", "w4",
              "wc1", "wc2", "wc3", "wc4", "wc5", "h1", "h2", "h3", "hd", "hs"]:
        print(f"    {k:6s} = {full[k]:.4f}")

    ok, reason = check_constraints(x_default)
    print(f"\n  默认参数约束: {ok} ({reason})")
