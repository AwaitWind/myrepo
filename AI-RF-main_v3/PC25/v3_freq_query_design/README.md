# freq_query_design —— PC25 第三版(filter3) 反向设计（Frequency-Query 代理版）

与 [`../inverse_design/`](../inverse_design) 是**同一个反向设计任务、同一份数据、同一评估口径**，
唯一区别是把 **正向代理内核** 从 `ResMLP 整曲线回归` 换成了 **Frequency-Query 条件网络（FQ-Net）**。
目的：用一个原理完全不同、对多螺旋谐振/陷波曲线更友好的模型，做「模型异化」并与 ResMLP 公平对比效果。

> 物理结构（filter3）：中心 CPW + 4 边矩形螺旋谐振器 + 中心叉指电容 + 可选多单元级联。
> 9 个设计变量、频率 0.01–5.0 GHz / 51 点、Y = [S21(51), S11(51)] 共 102 维。

## 核心区别（模型）

| 维度 | `../inverse_design`（ResMLP） | 本版 `freq_query_design`（FQ-Net） |
|---|---|---|
| 映射 | `x(9维) → 整条 102 维曲线` | `(x, f) → 该频点 (S21,S11)`，逐频点查询后拼成曲线 |
| 频率处理 | 隐式（输出维=频点下标） | **傅里叶特征** `γ(f)=[sin(2ᵏπf),cos(2ᵏπf)]`，频带对齐奈奎斯特 |
| 参数注入 | 拼进输入向量 | **FiLM 调制**：超网络由 `x` 生成每层 `(scale,shift)` 仿射调制频率坐标网络 |
| 物理约束 | 输出端 softplus 软上界 | **输出 `dB=-softplus(-raw)` 恒 ≤ 0**，同样杜绝 >0 dB 假通带 |
| 频率分辨率 | 固定 51 点 | 任意频率可查询、天然平滑 |

> 灵感来源：`../../../AI-RF-main/papers/` 中
> *Frequency-Query Enhanced EM Surrogate Modeling for Bandpass Filter Inverse Design*。
> FQ-Net 实现移植自 [`AI-RF-filter5/freq_query_design`](../../../AI-RF-filter5/freq_query_design)，
> 因全程从 `id_config` 取维度，放到本目录后自动适配 9 参数 / 51 频点。

**对外接口不变**：`forward_net.ForwardModel` 提供与 ResMLP 版完全一致的
`forward_unit / forward_unit_all / predict / y_mu_np / y_std_np / device / is_ensemble / freqs`，
因此 [`inverse_optimize.py`](inverse_optimize.py) 与 [`validate.py`](validate.py) **无需改动**即可复用，
**软约束中心频率（centered）等三种目标模式一并继承**。

## 数据复用（无需重跑 HFSS）

训练时若本目录 `id_data/dataset.npz` 不存在，[`forward_net.py`](forward_net.py) 会**自动回退**
到 `../inverse_design/id_data/dataset.npz`。所以直接沿用 ResMLP 那套已仿真好的数据，
ckpt 输出到本目录 `id_data/`，两套模型可在同一数据上公平对比。

## 三种反向设计目标（与 ResMLP 版一致）

- **centered（推荐）**：`--f0 3.5 --f0_tol 0.3 --bw 0.5` —— 中心频率允许 ±0.3 GHz 波动、最小带宽 0.5 GHz
- **fixed**：`--f0 3.5 --bw 0.5`（严格固定中心频率模板）
- **free**：`--free --target_bw 0.5`（完全不限中心频率）
- **下降速度**（三者通用）：`--rolloff_ghz 0.15 --rolloff_db -15`（通带边外 0.15GHz 处 S21 ≤ -15dB）

## 用法（端到端）

```bash
cd AI-RF-main_v3/PC25/freq_query_design

# ① 训练 FQ 代理（自动复用 ../inverse_design 的 dataset.npz）
python forward_net.py --train --ensemble 5 --epochs 800 --n_bands 10
python forward_net.py --eval          # 全曲线/仅S21 MAE + 预测 S21>0 占比(应为 0%)

# ② 软约束中心频率反设计（Top-K）
python inverse_optimize.py --ckpt id_data \
    --f0 3.5 --f0_tol 0.3 --bw 0.5 \
    --passband_db -3 --stopband_db -25 \
    --rolloff_ghz 0.15 --rolloff_db -15 \
    --margin_db 1.0 --std_penalty 0.5 \
    --top_k 10 --restarts 16 --out id_data/inv_centered.npz

# ③ 真 HFSS 复验 Top-10（自动沿用 npz 里的 f0/f0_tol）
python validate.py --result id_data/inv_centered.npz --top_k 10 --tag inv_centered
```

## FQ 专属超参（forward_net.py）

| 参数 | 作用 | 建议 |
|---|---|---|
| `--n_bands` | 傅里叶频率编码频带数（越大越能表达高频振荡；过大易过拟合噪声） | 8 ~ 12 |
| `--hidden` | FiLM 主干隐藏维 | 256（欠拟合可 384/512） |
| `--depth` | FiLM 主干层数 | 5 ~ 6 |
| `--sharp_boost` | 陡变频点（谐振/陷波）加权强度 | 6.0 |
| `--diff_reg` | 一阶差分正则（斜率/陷波对齐） | 0.15 |
| `--clip_db` | 深谷数值噪声截断 | -60 |
| `--huber_delta` | 对极端残差鲁棒（0=纯加权 MSE） | 0 或 1.0 |

## 与 ResMLP 版对比效果

两者共用同一 `dataset.npz` 与同一 `validate.py`，可直接对比：

```bash
cd ../inverse_design   && python forward_net.py --eval   # ResMLP
cd ../freq_query_design && python forward_net.py --eval   # FQ-Net (本目录)
```
重点看 `仅S21 MAE`、`预测 S21>0 占比`（FQ 应为 0%），以及各自逆向 Top-K 经 `validate.py` 复验后的真值 loss。

## 与 ResMLP 版共享的文件

`id_config.py / dataio.py / sampling.py / gen_dataset.py / spec.py / passband_spec.py /
inverse_optimize.py / validate.py` 均从 [`../inverse_design`](../inverse_design) 复制、逻辑一致
（含软约束中心频率 centered），仅 [`forward_net.py`](forward_net.py) 为 Frequency-Query 实现。
其余说明（参数空间、约束、频率网格、centered 细节）参见 [`../inverse_design/README.md`](../inverse_design/README.md)。
