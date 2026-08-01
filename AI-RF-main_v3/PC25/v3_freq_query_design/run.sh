#!/usr/bin/env bash
# 反向设计端到端命令清单（PC25 第三版 filter3 · Frequency-Query 代理版）
#
# 与 ../inverse_design 是同一任务、同一数据、同一评估口径，唯一区别：
#   正向代理内核 = Frequency-Query 网络（FQ-Net），而非 ResMLP。
#   FQ-Net：(x, f) → 该频点 (S21,S11)；傅里叶频率特征 + FiLM 参数调制 + dB<=0 物理约束。
#
# 数据复用：本目录若无 id_data/dataset.npz，forward_net.py 会自动回退到
#           ../inverse_design/id_data/dataset.npz —— 不必重跑 HFSS，且能与 ResMLP 公平对比。
#
# 反向设计三种目标（inverse_optimize/validate 与 ResMLP 版完全相同，含软约束中心频率）：
#   - centered（推荐）：--f0 + --f0_tol + --bw + rolloff 下降速度
#   - fixed：--f0 --bw（严格固定中心频率模板）
#   - free：--free（完全不限中心频率）

set -e

# ============================================================
# ① 训练 Frequency-Query 代理（自动复用 ../inverse_design 的 dataset.npz）
#    推荐 Ensemble×5；FQ 专属超参：--n_bands（傅里叶频带数）
# ============================================================
python forward_net.py --train --ensemble 5 --epochs 800 \
    --n_bands 10 --hidden 256 --depth 5 --sharp_boost 6.0 --diff_reg 0.15

python forward_net.py --eval          # 看 全曲线/仅S21 MAE + 预测 S21>0 占比(应为 0%)

# 欠拟合就加大主干：--hidden 384 --depth 6；高频振荡多就 --n_bands 12

# ============================================================
# ② 逆向求参（M3）—— 与 ResMLP 版命令完全一致；--ckpt id_data 用 Ensemble
# ============================================================

# A+. 软约束中心频率带通（推荐）：中心 3.5±0.3 GHz、最小带宽 0.5、边外 0.15GHz 压到 -15dB
python inverse_optimize.py --ckpt id_data \
    --f0 3.5 --f0_tol 0.3 --bw 0.5 \
    --passband_db -3 --stopband_db -25 --guard_ghz 0.3 \
    --rolloff_db -15 --rolloff_ghz 0.15 --rolloff_weight 2.0 \
    --margin_db 1.0 --std_penalty 0.5 \
    --top_k 10 --restarts 16 --maxiter 200 \
    --out id_data/inv_centered.npz

# A. 固定中心频率带通（严格）
# python inverse_optimize.py --ckpt id_data --f0 3.5 --bw 0.5 \
#     --passband_db -3 --stopband_db -25 --rolloff_db -15 --rolloff_ghz 0.15 \
#     --top_k 10 --restarts 16 --out id_data/inv_result.npz

# C. 自由通带（不限中心频率）
# python inverse_optimize.py --ckpt id_data --free --target_bw 0.5 --passband_db -3 \
#     --stopband_db -20 --rolloff_db -15 --rolloff_ghz 0.15 \
#     --top_k 10 --restarts 16 --out id_data/inv_free.npz

# ============================================================
# ③ 真 HFSS 验证（M3c）—— 一次跑 Top-K，按真值 loss 重排；自动沿用 npz 里 f0/f0_tol
# ============================================================
python validate.py --result id_data/inv_centered.npz --top_k 10 --tag inv_centered

# ============================================================
# ④ 与 ResMLP 版效果对比（同一 dataset.npz、同一 validate.py）
# ============================================================
#   代理精度：
#     cd ../inverse_design   && python forward_net.py --eval   # ResMLP
#     cd ../freq_query_design && python forward_net.py --eval   # FQ-Net (本目录)
#   重点看『仅S21 MAE』与『预测 S21>0 占比(FQ 应为 0%)』，以及各自 Top-K 真值 loss。
