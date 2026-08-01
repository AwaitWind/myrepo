# freq_query_design —— filter5 反向设计（Frequency-Query 代理版）

与 [`../inverse_design/`](../inverse_design) 是**同一个反向设计任务、同一份数据、同一评估口径**，
唯一区别是把 **正向代理内核** 从 `ResMLP 整曲线回归` 换成了 **Frequency-Query 条件网络**。
目的：用一个原理完全不同、且对多谐振/传输零点曲线更友好的模型，取得更好的效果，并能与旧方案公平对比。

## 核心区别（模型）

| 维度 | 旧版 `inverse_design`（ResMLP） | 本版 `freq_query_design`（FQ-Net） |
|---|---|---|
| 映射 | `x(11维) → 整条 200 维曲线` | `(x, f) → 该频点 (S21, S11)`，逐频点查询后拼成曲线 |
| 频率处理 | 隐式（输出维=频点下标） | **傅里叶特征** `γ(f)=[sin(2ᵏπf), cos(2ᵏπf)]`，频带对齐奈奎斯特 |
| 参数注入 | 拼进输入向量 | **FiLM 调制**：超网络由 `x` 生成每层 `(scale, shift)` 仿射调制频率坐标网络 |
| 物理约束 | 无（会外推出 >0 dB 假通带） | **输出 `dB=-softplus(-raw)` 恒 ≤ 0**，杜绝假通带（逆向被欺骗的元凶） |
| 频率分辨率 | 固定 100 点 | 任意频率可查询、天然平滑 |

> 灵感来源：`../../AI-RF-main/papers/` 中
> *Frequency-Query Enhanced EM Surrogate Modeling for Bandpass Filter Inverse Design*。

**对外接口不变**：`forward_net.ForwardModel` 提供与旧版完全一致的
`forward_unit / forward_unit_all / predict / y_mu_np / y_std_np / device / is_ensemble / freqs`，
因此 [`inverse_optimize.py`](inverse_optimize.py) 与 [`validate.py`](validate.py) **无需改动**即可复用。

## 数据复用（无需重跑 HFSS）

训练时若本目录 `id_data/dataset.npz` 不存在，[`forward_net.py`](forward_net.py) 会**自动回退**
到 `../inverse_design/id_data/dataset.npz`（见 `DATASET_NPZ` 逻辑）。
所以可直接沿用旧目录已仿真好的数据，ckpt 则输出到本目录 `id_data/`。

## 用法（端到端）

```bash
cd AI-RF-filter5/freq_query_design

# ① 训练 Frequency-Query 代理（自动复用 ../inverse_design 的 dataset.npz）
python forward_net.py --train --ensemble 5 --epochs 800
python forward_net.py --eval          # 看 全曲线/仅S21 MAE + 预测 S21>0 占比(应为 0)

# ② 逆向求参（与旧版命令完全一致）
python inverse_optimize.py --f0 3.5 --bw 0.5 \
    --passband_db -3 --stopband_db -25 \
    --rolloff_ghz 0.15 --rolloff_db -15 \
    --margin_db 1.0 --std_penalty 0.5 \
    --top_k 10 --restarts 16 --out id_data/inv_result.npz

# ③ 真 HFSS 复验 Top-10
python validate.py --result id_data/inv_result.npz --top_k 10
```

## 新增/专属超参（forward_net.py）

| 参数 | 作用 | 建议 |
|---|---|---|
| `--n_bands` | 傅里叶频率编码的频带数（越大越能表达高频振荡；过大易过拟合噪声） | 8 ~ 12 |
| `--hidden` | FiLM 主干隐藏维 | 256（欠拟合可 384/512） |
| `--depth` | FiLM 主干层数 | 5 ~ 6 |
| `--sharp_boost` | 陡变频点（传输零点/匹配谷）加权强度 | 6.0 |
| `--diff_reg` | 一阶差分正则（斜率/陷波对齐） | 0.15 |
| `--clip_db` | 深谷数值噪声截断 | -60 |
| `--huber_delta` | 对极端残差鲁棒（0=纯加权 MSE） | 0 或 1.0 |

## 与旧方案对比效果

两者共用同一 `dataset.npz` 与同一 `validate.py`，因此可直接对比：

```bash
# 旧
cd ../inverse_design && python forward_net.py --eval
# 新
cd ../freq_query_design && python forward_net.py --eval
```

重点看 `仅S21 MAE` 与 `预测 S21>0 占比`（新版应为 0%），
以及各自逆向 Top-K 经 `validate.py` 复验后的真值 loss。

## 与旧目录共享的文件

`id_config.py / dataio.py / sampling.py / gen_dataset.py / spec.py / passband_spec.py /
inverse_optimize.py / validate.py` 均从 `../inverse_design` 复制而来、逻辑一致，
仅 [`forward_net.py`](forward_net.py) 为全新实现。其余说明（参数空间、约束、频率网格、常见问题）
参见 [`../inverse_design/README.md`](../inverse_design/README.md)。
