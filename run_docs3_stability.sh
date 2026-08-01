#!/usr/bin/env bash
# docs=3 oracle 稳定性:同配置(prepop 修 + inject on + 全 warmup)重跑 3 次,
# 量化 GLM FP8 temp=0 跨重启的 FDT 波动(观察到 5 ↔ 0)。
set -x
cd /root/sglang
export PATH=/opt/dynamo/venv/bin:$PATH
export PYTHONPATH=/root/sglang/python
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
for R in 1 2 3; do
  echo "###### docs=3 stability RUN $R $(date) ######"
  python quick_test_online.py \
    --modes full_recompute pic_a3_oracle \
    --oracle-warmup-docs 3 \
    --tp 8 2>&1 | tee "docs3_run${R}.log"
done
echo "###### ALL DONE $(date) ######"
