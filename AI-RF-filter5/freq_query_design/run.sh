#!/usr/bin/env bash
# 反向设计端到端命令清单（filter5 —— CPW + 3段矩形谐振器 + via 打孔阵列）
#
# 前置：cd 到 AI-RF-filter5/inverse_design 目录后再执行
#       Windows PowerShell 请把 python 换成对应可执行文件
#
# 参数顺序（11 维紧凑，对称减维）：
#   l_end  l_res  w1 w2 w3 w4  wc_end  wc_res  h_res  hd  hs
#   （id_config._expand_to_full 会自动展开成 filter_layout_generate 需要的 19 参数）
# 频率网格：0.01~9.91 GHz / 100 点（步长=0.1 GHz）
#
# 【本版网络针对本电路定制】
#   - 3 阶带通结构 + via 传输零点 → 通带内多匹配谷 + 通带外陷波
#   - 代理：ResMLP(hidden512/depth6) + 逐维归一化 + 频段加权(sharp_boost6) +
#           差分正则 + clip_db robust + Ensemble
#   - 目标：显式约束阻带 + 下降沿(rolloff)，输出 Top-K 供真值重排
#   - 每次反设计自动输出「代理预测效果图」<out>_pred.png（不依赖 HFSS，立刻可看）
#   - 软约束中心频率(centered)：--f0 X --f0_tol Y --bw Z（中心可波动，见下方 A+）

set -e

# ============================================================
# ① 生成数据（M1）—— 进程隔离批量仿真
# ============================================================
python gen_dataset.py --n 300                     # 采样 300 个（快速起步）
# python gen_dataset.py --n 1000                  # 大规模（推荐 1000+）
# python gen_dataset.py --batch-size 20           # 每个 worker 处理 20 个（默认 40）
# python gen_dataset.py --aggregate-only          # 只把已有 s2p 聚合成 dataset.npz


# ============================================================
# ② 训练正向代理（M2）—— 推荐 Ensemble×5
# ============================================================
python forward_net.py --train --ensemble 5 --epochs 800
python forward_net.py --eval                       # 查看 val MAE(dB) / 仅S21 MAE
# 若显存不足或想更快：--ensemble 3 --hidden 384
# 若欠拟合（val MAE 偏大）：--ensemble 8 --hidden 640 --sharp_boost 8
# 关闭深谷截断：--clip_db -200


# ============================================================
# ③ 逆向求参（M3）—— 三种目标任选其一，均输出 Top-K
# ============================================================

# ------------------------------------------------------------
# A. 固定中心频率带通（中心 3.5 GHz，带宽 0.5 GHz）
# ------------------------------------------------------------
python inverse_optimize.py --f0 3.5 --bw 0.5 \
    --passband_db -3 --stopband_db -25 \
    --rolloff_ghz 0.15 --rolloff_db -15 --rolloff_weight 2.0 \
    --margin_db 1.0 --std_penalty 0.5 \
    --top_k 10 --restarts 16 --maxiter 200 \
    --out id_data/inv_result.npz

# ------------------------------------------------------------
# A+. 软约束中心频率带通（推荐）—— 中心频率允许 ±f0_tol 波动
#    --f0 3.5 --f0_tol 0.3  中心 3.5 GHz，允许 ±0.3 GHz 漂移（不必卡死）
#    --bw 0.5               最小带宽（该邻域内需有 >=0.5 GHz 的通带）
#    复用自由通带机制 + 「窗口中心落在 f0±tol」约束，输出 Top-K；
#    反设计完自动出 id_data/inv_centered_pred.png（代理预测效果），validate 自动沿用 f0/f0_tol
# ------------------------------------------------------------
python inverse_optimize.py \
    --f0 3.5 --f0_tol 0.3 --bw 0.5 \
    --passband_db -3 --stopband_db -25 --guard_ghz 0.3 \
    --rolloff_db -15 --rolloff_ghz 0.15 --rolloff_weight 2.0 \
    --margin_db 1.0 --std_penalty 0.5 \
    --top_k 10 --restarts 16 --maxiter 200 \
    --out id_data/inv_centered.npz

# ------------------------------------------------------------
# B. 用现成 s2p 作目标
# ------------------------------------------------------------
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
python validate.py --result id_data/inv_result.npz --top_k 10

# 软约束中心频率模式：自动沿用 npz 里的 f0/f0_tol（在 f0±tol 内评价真值通带）
python validate.py --result id_data/inv_centered.npz --top_k 10

# 自由通带模式：自动沿用 npz 里的设置
python validate.py --result id_data/inv_free.npz --top_k 10

# 断点续跑：中断后重跑会复用已有 s2p；加 --no_reuse 强制重仿真
# 直接给一组参数验证（11 个数字，顺序：l_end l_res w1 w2 w3 w4 wc_end wc_res h_res hd hs）
# python validate.py --x 5.25 5.2 1 2 2 1 1 1 5 0.8 0.5 --f0 3.5 --bw 0.5
