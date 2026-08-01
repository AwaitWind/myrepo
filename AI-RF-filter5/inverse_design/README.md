# inverse_design —— filter5（CPW + 3段矩形谐振器 + via 打孔阵列）· 反向设计

给一条 S 参数目标（中心频率 / 带宽 / 插损 / 阻带抑制 / 下降沿），求几何参数。

## 架构

```
Layer 0  数据生成    sampling → gen_dataset(进程隔离) → dataset.npz
Layer 1  正向代理    forward_net: ResMLP + Ensemble  x → S 参数曲线（毫秒级替代 HFSS）
Layer 2  逆向求解    inverse_optimize: CMA-ES / 梯度  给 y* 求 x̂（输出 Top-K）
         目标定义    spec (固定 target) / passband_spec (自由通带 soft-min)
验证              validate: 真 HFSS 复验 Top-K，按真值 loss 重排
```

## 本电路特性 → 网络定制（重要）

filter5 = **中心 CPW（tl_1/tl_2/tl_c） + 3 对矩形谐振器（re_1_x, re_2_x, re_3_x）+ via 打孔阵列**：
- 3 对谐振器从主导带 tl_c 挖出 → **三阶带通结构**（通带内 3 个匹配谷）
- via 阵列（沿 PCB 边缘打孔）→ **传输零点**（S21 深陷波）
- 0.01~9.91 GHz / 100 点，曲线呈现「一个通带 + 通带内多匹配谷 + 通带外陷波 + 高频寄生」的形态

| 电路/曲线特性 | 网络设计 |
|---|---|
| via 传输零点 + S11 匹配谷（比普通通带边更陡） | 频段自适应加权 MSE，`sharp_boost=6.0` 放大陡变频点 |
| 多谐振峰 + 传输零点（Y_DIM=200） | ResMLP `hidden=512, depth=6`（GELU + skip） |
| S21/S11 动态范围大 | **逐维 mean/std 归一化**（S21 段 / S11 段各自尺度）|
| 深谷/尖峰是局部极值，斜率关键 | 一阶差分正则 `diff_reg=0.15`（斜率对齐）|
| HFSS 偶发 -70dB 数值噪声深谷 | `clip_db=-60` 截断，robust 化 loss |
| 参数对零点位置强非线性 | Ensemble 多 seed（均值降 MAE，std 做不确定度惩罚）|

## 参数空间（11 维，含对称减维）

原始 19 维参数经**对称约束合并**后减到 **11 维**（搜索空间小 ~42%）：

| 合并组 | 紧凑变量 | 对应原始参数 | 物理含义 |
|---|---|---|---|
| 输入/输出段长对称 | `l_end` | `l1 = l5` | 首末段进线长度 |
| 3 谐振段等长 | `l_res` | `l2 = l3 = l4` | 3 谐振段沿 y 长度 |
| 输入/输出中心线宽对称 | `wc_end` | `wc1 = wc5` | 输入/输出段中心导带宽度 |
| 3 谐振段中心线宽相同 | `wc_res` | `wc2 = wc3 = wc4` | 3 谐振段中心导带宽度 |
| 3 对谐振器等宽 | `h_res` | `h1 = h2 = h3` | 3 对矩形谐振器水平延伸 |

**完整参数表（11 维）：**

| 分组 | 参数 | 含义 | 范围 (mm) | 默认 |
|---|---|---|---|---|
| **段长度** | `l_end` | 首=末段进线长（对称） | 3.68 ~ 6.83 | 5.25 |
| | `l_res` | 3 谐振段长（对称） | 3.64 ~ 6.76 | 5.20 |
| **段间距** | `w1` | 输入段↔第 1 谐振段 | 0.70 ~ 1.30 | 1.00 |
| | `w2`, `w3` | 相邻谐振段间距 | 1.40 ~ 2.60 | 2.00 |
| | `w4` | 第 3 谐振段↔输出段 | 0.70 ~ 1.30 | 1.00 |
| **中心线宽** | `wc_end` | 输入=输出段中心线宽（对称） | 0.70 ~ 1.30 | 1.00 |
| | `wc_res` | 3 谐振段中心线宽（对称） | 0.70 ~ 1.30 | 1.00 |
| **谐振器宽** | `h_res` | 3 对谐振器水平延伸（对称） | 3.50 ~ 6.50 | 5.00 |
| **via 打孔** | `hd` | via 直径 | 0.56 ~ 1.04 | 0.80 |
| | `hs` | 相邻 via 边缘间距 | 0.35 ~ 0.65 | 0.50 |

**展开**：`id_config._expand_to_full(x)` 把 11 维紧凑向量还原成 filter_layout_generate
需要的完整 19 参数字典（`l1..l5`, `w1..w4`, `wc1..wc5`, `h1..h3`, `hd`, `hs`）。

**固定超参**：`PCB_X=17.5 mm`（`PCB_Y` 由参数动态计算），`PCB_Z=0.762 mm`。

## 硬约束

`check_constraints(x)` 先用 `_expand_to_full` 展开成 19 参数字典，再拒绝：
- 任何参数 ≤ 0
- `wc_res/2 + h_res > PCB_X/2 - 0.2` 谐振器 x 越界（对称约束下 3 条等价合一）
- via 阵列与中心导体重叠：`PCB_X/2 - hd/2 <= max(wc/2 + h) + 0.1`
- via 阵列 y 长度 < `hd`（`l2+l3+l4+w1+w2+w3+w4 < hd`）

## 频率网格

`0.01 ~ 9.91 GHz`，`100 点`（`np.arange`），步长 `0.1 GHz`。
必须与 `HFSS_funcs.py:19-21` 的 `Frequency_Start/Stop/Step` 完全一致（`dataio` 会自动重采样对齐）。

## 用法（端到端）

见 [`run.sh`](run.sh)。核心 4 步：

```bash
cd AI-RF-filter5/inverse_design

# ① 生成数据（300 样本快速起步，1000+ 更好）
python gen_dataset.py --n 300

# ② 训代理（推荐 Ensemble×5）
python forward_net.py --train --ensemble 5 --epochs 800
python forward_net.py --eval

# ③ 逆向求参（固定带通 3.5 GHz，输出 Top-10）
python inverse_optimize.py --f0 3.5 --bw 0.5 \
    --passband_db -3 --stopband_db -25 \
    --rolloff_ghz 0.15 --rolloff_db -15 \
    --margin_db 1.0 --std_penalty 0.5 \
    --top_k 10 --restarts 16 --out id_data/inv_result.npz

# ④ 真 HFSS 复验 Top-10（按真值 loss 重排 + 叠加图）
python validate.py --result id_data/inv_result.npz --top_k 10
```

## 关键参数速查（inverse_optimize）

| 参数 | 作用 | 建议 |
|---|---|---|
| `--stopband_db` | 阻带门槛（自由通带模式必设） | -20 ~ -25 |
| `--rolloff_db` / `--rolloff_ghz` | 下降沿哨兵（边外 X GHz 处 S21 须 ≤ Y dB） | -15 dB @ 0.15 GHz |
| `--margin_db` | 代理误差余量（通带/阻带门槛加保守量） | 1.0；不达标就调小到 0.5 |
| `--std_penalty` | Ensemble 不确定度惩罚（须先训 ensemble） | 0.5 |
| `--top_k` | 输出候选数（validate 逐个复验） | 10 |
| `--restarts` | CMA-ES 重启次数（19 维盆地多，宁多勿少） | 16；卡住加到 24 |

## 数据文件布局

```
AI-RF-filter5/inverse_design/id_data/
├── params.npy              固定采样点 (N,19)，断点续跑复用
├── index.csv               每个样本的仿真状态（try/ok/fail）
├── s2p/sample_XXXXX.s2p    每个样本的原始 Touchstone
├── dataset.npz             聚合训练数据：X, Y, freqs, param_names, include_s11
├── forward_net.pt          单模型 ckpt 或 ensemble 元 index
├── forward_net_seed*.pt    ensemble 各成员
├── inv_result.npz          逆向结果（含 topk_x / topk_y_pred / 模式）
├── validate_rankNN.s2p     Top-K 各候选的 HFSS 真值
└── validate_topk_overlay.png  Top-K 叠加对比图（按真值 loss 排名着色）
```

## 依赖

- `numpy`, `scipy`, `matplotlib`, `scikit-rf` （通用）
- `pyaedt` + Ansys AEDT （只有 gen_dataset 和 validate 需要）
- `torch` （forward_net, inverse_optimize）
- `cma` （inverse_optimize --method cma，缺则回退简单 ES）

## 常见问题

**Q: main.py 或 gen_dataset 第 2 次运行就崩，报"same name"或"port already exists"。**
A: `HFSS_funcs.filter_layout_generate` 首行已加 `_cleanup_previous_geometry(hfss)`，
理论上不应发生。如仍报错，试试：(a) `print(hfss.modeler.object_names)` 看是不是有意外
残留；(b) 把残留对象名前缀加进 `HFSS_funcs.py` 的 `_STATIC_OBJECT_NAMES` 或 `_cleanup_previous_geometry`
的前缀列表。

**Q: 训练后 val MAE 依旧很大（>2~3 dB）。**
A: 首选 (a) `--ensemble 8 --hidden 640`；(b) 加大 `--sharp_boost 8`；(c) 补数据到 1000+；
(d) 若怀疑个别深谷噪声主导，试 `--huber_delta 1.0`。

**Q: Top-K 全部 `constraint_ok=N`（约束不满足）。**
A: 目标太苛刻。放宽 `--margin_db 0.5`，或先用 `--f0 3.5 --bw 0.6`（更宽通带）确认有解空间。

**Q: 代理说很好（objective 低），HFSS 真值却差很多。**
A: 代理"画大饼"。(a) 提高 `--margin_db` 到 1.5~2；(b) 训 ensemble 并开 `--std_penalty 0.5`
避开代理不确定区；(c) Top-K 里挑真值最优那个（validate 已自动重排）。

**Q: 想改参数范围。**
A: 编辑 `id_config.py` 的 `PARAM_BOUNDS`，然后重新 `gen_dataset` + `forward_net --train`
（旧数据 X 与新 LOWER/UPPER 不一定匹配）。
