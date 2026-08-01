#!/usr/bin/env bash
# 反向设计端到端命令清单（PC25 第三版 CPW 滤波器 · filter3_layout_generate）
#
# 前置：cd 到 PC25/inverse_design 目录后再执行
#       Windows PowerShell 请把 python 换成对应可执行文件
#
# ============================================================
# 【v2 重构（2026-07）】显式引入"下降系数"，代理网络升级
#   - forward_net.py：ResMLP、逐维归一化、频段加权+差分正则、Ensemble
#   - inverse_optimize.py：Top-K、rolloff/stopband/margin/std_penalty
#   - validate.py：一次跑 K 个候选、按真值 loss 重排
# ============================================================

set -e

# ============================================================
# ① 生成数据（M1）—— 进程隔离批量仿真
# ============================================================
python gen_dataset.py --n 2000                    # 目标 2000 条；断点续跑
# python gen_dataset.py --aggregate-only          # 只把已有 s2p 聚合成 dataset.npz


# ============================================================
# ② 训练正向代理（M2）—— 强烈建议用 Ensemble
# ============================================================
# 推荐：5 个 seed 的集成（预估 val MAE 下降 30%~50%，稳定性大幅提升）
python forward_net.py --train --ensemble 5 --epochs 800

# 若只想快速验证：单模型（更快，效果差一点）
# python forward_net.py --train --epochs 800

python forward_net.py --eval                       # 查看数据集上 MAE(dB)


# ============================================================
# ③ 逆向求参（M3）—— 三种目标模式；都启用 Top-K 输出
# ============================================================

# ------------------------------------------------------------
# A. 固定中心频率带通（中心 3.5 GHz，带宽 0.5 GHz，业界常用 -3 dB 带宽）
#    - --rolloff_db -15 --rolloff_ghz 0.15  → 通带边外 150 MHz 处必须 ≤ -15 dB（下降沿硬约束）
#    - --margin_db 1.0                      → 给通带/阻带留 1 dB 代理误差余量
#    - --std_penalty 0.5                    → Ensemble 不确定度惩罚（只在 --ensemble 训练后生效）
#    - --top_k 10 --restarts 16             → 输出 10 个候选，16 次随机重启
# ------------------------------------------------------------
python inverse_optimize.py \
    --f0 3.5 --bw 0.5 \
    --passband_db -3 --stopband_db -25 \
    --transition_ghz 0.15 --trans_weight 0.5 \
    --rolloff_db -15 --rolloff_ghz 0.15 --rolloff_weight 2.0 \
    --margin_db 1.0 --std_penalty 0.5 \
    --top_k 10 --restarts 16 --maxiter 200 \
    --out id_data/inv_result.npz

# ------------------------------------------------------------
# A+. 软约束中心频率带通（推荐）—— 中心频率允许 ±f0_tol 波动
#    --f0 3.5 --f0_tol 0.3   中心 3.5 GHz，允许 ±0.3 GHz 漂移（不必卡死在 3.5）
#    --bw 0.5                最小带宽（该邻域内需有 >=0.5 GHz 的通带）
#    --rolloff_ghz 0.15 --rolloff_db -15   下降速度（通带边外 0.15GHz 处压到 -15dB）
#    内部复用自由通带机制 + 「窗口中心必须落在 f0±tol」约束，输出 Top-K；
#    validate 会自动沿用 npz 里的 f0/f0_tol。
# ------------------------------------------------------------
python inverse_optimize.py \
    --f0 3.5 --f0_tol 0.3 --bw 0.5 \
    --passband_db -3 --stopband_db -25 --guard_ghz 0.3 \
    --rolloff_db -15 --rolloff_ghz 0.15 --rolloff_weight 2.0 \
    --margin_db 1.0 --std_penalty 0.5 \
    --top_k 10 --restarts 16 --maxiter 200 \
    --out id_data/inv_centered.npz

# B. 用现成 s2p 作目标
# python inverse_optimize.py --target_s2p /path/to/good.s2p \
#        --top_k 10 --restarts 16 --out id_data/inv_from_s2p.npz

# ------------------------------------------------------------
# C. 自由通带（位置无关）—— 中心频率不限，只要求存在一段"好"通带
#    重点：--stopband_db 一定要设，否则会退化到"全域平坦"
# ------------------------------------------------------------
python inverse_optimize.py --free \
    --target_bw 0.5 --passband_db -3 \
    --stopband_db -20 --stop_weight 1.0 --guard_ghz 0.3 \
    --rolloff_db -15 --rolloff_ghz 0.15 --rolloff_weight 1.5 \
    --margin_db 1.0 --std_penalty 0.5 \
    --top_k 10 --restarts 16 --maxiter 200 \
    --out id_data/inv_free.npz

# 梯度法（速度更快，但依赖代理可微 + 局部性强，建议只在 CMA 之后做微调）
# python inverse_optimize.py --f0 3.5 --bw 0.5 --method grad --restarts 8 --maxiter 400


# ============================================================
# ④ 真 HFSS 验证（M3c）—— 一次跑 Top-K，按真值 loss 重排
# ============================================================

# 固定 target 模式：跑全部 10 个候选
python validate.py --result id_data/inv_result.npz --top_k 10 --tag inv_result

# 软约束中心频率模式：自动沿用 npz 里的 f0/f0_tol（在 f0±tol 内评价真值通带）
python validate.py --result id_data/inv_centered.npz --top_k 10 --tag inv_centered

# 自由通带模式：自动沿用 npz 里的目标设置
python validate.py --result id_data/inv_free.npz --top_k 10 --tag inv_free

# 只验 Top-1（原来的默认行为）
# python validate.py --result id_data/inv_result.npz

# 直接给一组参数验证（顺序: w0 g d0 lf a b1 b2 d p0）
# python validate.py --x 2.2 0.2 0.6 0.4 3.1 0.3 0.3 1.4 0.5 --f0 3.5 --bw 0.5


# ============================================================
# 【调参速查】
# ============================================================
#   代理 val MAE > 3 dB  → 补数据到 2000+，或加深 --hidden 512 --depth 6
#   代理 val MAE < 2 dB  → 直接进入逆向阶段，Top-K 大概率能出好设计
#   逆向 Top-K 真值 loss 都很高 → 拉大 --margin_db（1.0 → 2.0），或换 --free 模式
#   Top-K 都很相似 → CMA-ES 陷入盆地，加大 --restarts（16 → 32）或 --seed
