# inverse_design —— PC25 第三版 CPW 滤波器 · 反向设计

给一条 S 参数目标（中心频率 / 带宽 / 插损 / 阻带抑制 / **下降系数**），求几何参数。
底层建模复用 [`../HFSS_funcs.filter3_layout_generate`](../HFSS_funcs.py)（4 边矩形螺旋谐振器 + 中心叉指电容）。

---

## 🔥 v2 重构说明（2026-07）：显式引入"下降系数" + 代理升级

原始版本存在两个结构性问题：

1. **代理网络学不到谐振尖点**：Y 用全局标量归一化 → S11 被 S21 压死；MSE 无频段加权 → 阻带 flat 压平谐振尖点
2. **目标函数没有"下降沿"约束**：自由通带只看通带、固定 target 过渡带权重 = 0 → 优化器可能给出"平缓下降"的伪滤波器

本次重构（**Batch A + B + C**）针对这些缺陷做了系统性修复：

| 层面 | 改动 | 收益 |
|---|---|---|
| **代理网络** | ResMLP (hidden=384, depth=5, skip)、逐维 mean/std 归一化、频段自适应加权 MSE + 一阶差分正则、Ensemble 训练 | val MAE 预计降 30~50%，谐振尖点/下降沿可辨 |
| **loss 目标** | `passband_free_loss` 加"阻带缺额" + "skirt 哨兵"两项；`make_bandpass_target` 过渡带 `trans_weight=0→0.5, transition_ghz=0.4→0.15` | "下降系数"变成显式一等指标，逼出真正的带通形状 |
| **逆向求解** | CMA-ES `restarts=4→16`；输出 **Top-K 候选**；代理误差余量 `--margin_db`；Ensemble 不确定度惩罚 `--std_penalty` | 从"1 个代理最优解"变成"K 个真值筛选"，真值命中率翻倍 |
| **HFSS 验证** | `validate.py` 一次跑 K 个候选，按真值 loss 重排 + 叠加图 | 一步到位选出真正的最优设计 |

**关键新参数速览**：

```
--stopband_db -25       # 阻带上限（自由通带模式下必须给，否则退化）
--rolloff_db -15        # 通带边外 skirt 哨兵点的 S21 上限（"下降沿要陡"）
--rolloff_ghz 0.15      # 哨兵点距通带边缘的频率
--margin_db 1.0         # 通带/阻带门槛留 1 dB 代理误差余量（真值更稳）
--top_k 10              # 输出 10 个候选参数
--std_penalty 0.5       # Ensemble 不确定度惩罚（需 --ensemble 训练）
```

---

## 架构

```
Layer 0  数据生成    sampling → gen_dataset(进程隔离) → dataset.npz
Layer 1  正向代理    forward_net: ResMLP + Ensemble  x → S 参数曲线（毫秒级替代 HFSS）
Layer 2  逆向求解    inverse_optimize: CMA-ES / 梯度  给 y* 求 top-K 个 x̂
         目标定义    spec (固定 target + rolloff 哨兵) / passband_spec (自由通带 + 阻带 + skirt)
Layer 3  真值验证    validate: 一次跑 K 个候选 HFSS，按真值 loss 重排
```

## 参数空间（9 维）

| 参数 | 含义 | 范围 (mm) | 默认 |
|---|---|---|---|
| `w0` | 中心 CPW 导带宽度 | 1.0 ~ 4.0 | 2.2 |
| `g`  | CPW 中心导带两侧缝隙 | 0.1 ~ 0.5 | 0.2 |
| `d0` | 中心间隔线长（叉指纵向可用长度） | 0.3 ~ 1.5 | 0.6 |
| `lf` | 叉指长度（受 d0 约束） | 0.15 ~ 0.9 | 0.4 |
| `a`  | 螺旋外中心线长（决定螺旋整体尺寸） | 2.0 ~ 5.0 | 3.1 |
| `b1` | 螺旋条带宽度 | 0.15 ~ 0.6 | 0.3 |
| `b2` | 螺旋条带间距 | 0.15 ~ 0.6 | 0.3 |
| `d`  | y 方向 gnd 半宽度参数 | 1.0 ~ 3.0 | 1.4 |
| `p0` | y 方向单元额外间距 | 0.2 ~ 1.5 | 0.5 |

**固定超参**（不进采样空间）：`n=10`（叉指电容数量）, `N=1`（单元数）, `cpw_length=2`, `PCB=30×14.5×0.762`。

## 硬约束

`check_constraints(x)` 会拒绝：
- 存在非正参数
- `w0 / (2n-1) < 0.02 mm` 叉指条带宽下限
- `lf > d0 - 0.05` 叉指长度超过中心间隔线
- `a + b2 - b1/2 <= 0.02` 螺旋外中心线长非正
- `w0/2 + g + a + b2 > pcb_x/2 - 0.5` 螺旋在 x 方向越出 PCB
- `b2 < 0.05` 螺旋条带间距过小

## 频率网格

`0.01 ~ 5.0 GHz`，`步长 0.1`，共 `51 点`。与 [`../HFSS_funcs.py`](../HFSS_funcs.py) 完全一致。

---

## 用法（端到端，v2 推荐）

见 [`run.sh`](run.sh)。核心 4 步：

```bash
# ① 生成数据（目标 2000 样本；断点续跑）
python gen_dataset.py --n 2000

# ② 训练 Ensemble 代理（5 个 seed，val MAE 大幅降低）
python forward_net.py --train --ensemble 5 --epochs 800
python forward_net.py --eval

# ③ 逆向求参（固定带通 + 显式下降沿 + Top-10）
python inverse_optimize.py \
    --f0 3.5 --bw 0.5 \
    --passband_db -3 --stopband_db -25 \
    --rolloff_db -15 --rolloff_ghz 0.15 --rolloff_weight 2.0 \
    --margin_db 1.0 --std_penalty 0.5 \
    --top_k 10 --restarts 16 \
    --out id_data/inv_result.npz

# ④ 真 HFSS 验证 Top-10，按真值 loss 重排
python validate.py --result id_data/inv_result.npz --top_k 10
```

自由通带模式（不指定 f0，让优化器自己找）：

```bash
python inverse_optimize.py --free \
    --target_bw 0.5 --passband_db -3 \
    --stopband_db -20 --rolloff_db -15 \
    --top_k 10 --restarts 16 \
    --out id_data/inv_free.npz

python validate.py --result id_data/inv_free.npz --top_k 10
```

## 目标函数详解

### 固定目标模式（`--f0 --bw`）

`loss = 加权MSE(y_pred, y_target) + rolloff_penalty(通带边外哨兵) + 约束罚 + std_penalty`

其中 `y_target` 由 [`spec.make_bandpass_target()`](spec.py) 生成：

- 通带 `[f0-bw/2, f0+bw/2]` → 目标 = `passband_db`，权重 3.0
- 过渡带 `±transition_ghz`（默认 0.15 GHz） → 线性插值，权重 0.5（v1 是 0）
- 阻带 → 目标 = `stopband_db`，权重 2.0
- **rolloff 哨兵**：通带边外 `rolloff_ghz` 处必须 ≤ `rolloff_db`

### 自由通带模式（`--free`）

`loss = passband_deficit + stop_weight·stopband_excess + rolloff_weight·skirt_deficit`

在长度 `target_bw` 的滑动窗口上取最优位置：
- 通带缺额：窗口内 S21 < passband_db 的量
- 阻带越界：窗口外 + guard 之外 S21 > stopband_db 的量
- Skirt 哨兵：窗口左右各 `rolloff_ghz` 处 S21 > rolloff_db 的量

**v1 只有通带缺额** → v2 三项联合约束才能逼出真正的带通形状。

## 数据文件布局

```
inverse_design/id_data/
├── params.npy              固定采样点 (N,9)，断点续跑复用
├── index.csv               每个样本的仿真状态（try/ok/fail）
├── s2p/
│   └── sample_00000.s2p …  每个样本的原始 Touchstone
├── dataset.npz             聚合训练数据
├── forward_net.pt          单模型 or Ensemble index（v2）
├── forward_net_seed{0..4}.pt Ensemble 成员（--ensemble N）
├── inv_result.npz          逆向求解结果（含 topk_x / topk_y_pred / topk_obj）
├── inv_result_rank{01..10}.s2p  Top-K 各自的 HFSS 仿真结果
├── inv_result_rank{01..10}_compare.png  各候选对比图
└── inv_result_topk_overlay.png  Top-K 叠加对比图（真值 loss 排名）
```

## 依赖

- `numpy`, `scipy`, `matplotlib`, `scikit-rf`
- `pyaedt` + Ansys AEDT （`gen_dataset` / `validate` 用）
- `torch` （`forward_net` / `inverse_optimize --method grad`）
- `cma` （`inverse_optimize --method cma`，缺则回退简单 ES）

## 与第二版实现的差异

- **建模函数**：`filter3_layout_generate`（4 边矩形螺旋 + 叉指电容）
- **参数**：9 维（w0/g/d0/lf/a/b1/b2/d/p0）
- **频率网格**：51 点 `arange(0.01, 5.0, 0.1)`

## 常见问题

**Q: 从 v1 升级到 v2，旧 `forward_net.pt` 还能用吗？**
A: 可以直接加载（`ForwardModel` 自动兼容标量归一化 ckpt），但**强烈建议重新训练**享受 ResMLP + 逐维归一化 + Ensemble 的红利。

**Q: 什么时候用 `--std_penalty`？**
A: **只有 `--ensemble N (N>=2)` 训练的模型才生效**。开启后 objective 会惩罚"代理成员之间预测不一致"的参数区域，代理误差大的地方被自动避开。推荐值 `0.5 ~ 2.0`。

**Q: `--margin_db` 设多少合适？**
A: 看代理 val MAE：val MAE ≈ 1 dB → `margin_db=0.5`；val MAE ≈ 2 dB → `margin_db=1.0`；val MAE > 3 dB → `margin_db=2.0`（但更建议先补数据/加深模型）。

**Q: `--rolloff_db` 怎么定？**
A: 一般 `passband_db - 12 ~ 15 dB` 是合理起点。例如 `passband_db=-3` → `rolloff_db=-15`。想要更陡的下降沿可以设 `-20 ~ -25`（但代理必须能画出陡的下降沿，否则优化器找不到解）。

**Q: Top-K 全部 constraint_ok=N 怎么办？**
A: 说明代理找的"最优"都落在约束外。加大 `--restarts` 到 32，或者 `_constraint_penalty` 的 `scale` 从 1000 提到 5000（在 [`inverse_optimize.py`](inverse_optimize.py) 里改）。

**Q: 逆向 Top-K 真值 loss 都很高，但代理 loss 很低。**
A: 典型的"代理欠拟合" —— 优化器在代理的"幻觉低谷"里疯狂优化。三步走：
1. 加大 `--std_penalty` 到 2.0（惩罚代理不确定区）
2. 加大 `--margin_db` 到 2.0（更保守目标）
3. 补数据（重新跑 `gen_dataset.py --n 2000`）+ 重训 Ensemble

**Q: 想改叉指数 `n` 或单元数 `N`。**
A: 改 [`id_config.N_FINGERS_FIXED / N_CELLS_FIXED`](id_config.py)，然后**清空 `id_data/` 重跑**（旧数据的响应曲线对应不同的电路，代理必须重训）。
