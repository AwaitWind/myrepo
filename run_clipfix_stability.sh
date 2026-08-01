#!/usr/bin/env bash
# clip 稳定性:warmup 发布修复后,跑 3 次真 clip 看 FDT 是否稳定(修复前 2/5 抖动)。
set -x
cd /root/sglang
export PATH=/opt/dynamo/venv/bin:$PATH
export PYTHONPATH=/root/sglang/python
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_PIC_A3_CLIP_CAPTURE=1
for R in 1 2 3; do
  echo "###### clip stability RUN $R $(date) ######"
  python quick_test_online.py \
    --modes full_recompute pic_a3_oracle \
    --oracle-warmup-docs 3 \
    --tp 8 2>&1 | tee "clipfix_run${R}.log"
done
echo "###### ALL DONE $(date) ######"
