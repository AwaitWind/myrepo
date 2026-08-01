#!/usr/bin/env bash
# 诊断实验:pic_a3_oracle 在 warmup 覆盖 1/2/3 个文档段时的精度对比。
# 目的:定位 oracle isolated-hidden inject 从哪个文档开始崩(答案在 C2)。
#   docs=1 → 只 warmup C1;measure 时 C2/C3 走 miss recompute(不 inject)
#   docs=2 → warmup C1+C2;C2 开始被 inject(isolated 缺 C1 上下文)
#   docs=3 → warmup C1+C2+C3(= 当前默认,已知 FDT=0 崩)
# 每个实验:full_recompute(FDT baseline)+ pic_a3(已知正确对照)+ pic_a3_oracle(被测)。
set -x
cd /root/sglang
export PATH=/opt/dynamo/venv/bin:$PATH
export PYTHONPATH=/root/sglang/python
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1

for D in 1 2 3; do
  echo "########################################################"
  echo "########## ORACLE WARMUP DOCS = $D  ($(date)) ##########"
  echo "########################################################"
  python quick_test_online.py \
    --modes full_recompute pic_a3 pic_a3_oracle \
    --oracle-warmup-docs "$D" \
    --tp 8 2>&1 | tee "oracle_wu${D}.log"
done
echo "########## ALL DONE $(date) ##########"
