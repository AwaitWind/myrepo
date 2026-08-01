#!/usr/bin/env bash
# 追 FDT5→6 gap:验证 gap 来自 C2/C3 的 isolated≠in-context。
# prepop 已修 + inject on。--oracle-warmup-docs 控制哪些段走 isolated inject:
#   docs=1: C1 isolated inject(=in-context 完美) + C2/C3 fresh recompute(in-context) → 应最干净
#   docs=2: +C2 isolated inject(缺 C1 上下文) → 引入 gap
#   docs=3: +C2/C3 isolated inject → 现状 FDT5
set -x
cd /root/sglang
export PATH=/opt/dynamo/venv/bin:$PATH
export PYTHONPATH=/root/sglang/python
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
for D in 1 2 3; do
  echo "###### ORACLE DOCS=$D (inject on, prepop fixed) $(date) ######"
  python quick_test_online.py \
    --modes full_recompute pic_a3_oracle \
    --oracle-warmup-docs "$D" \
    --tp 8 2>&1 | tee "oracle_docs${D}_fixed.log"
done
echo "###### ALL DONE $(date) ######"
