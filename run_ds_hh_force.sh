#!/usr/bin/env bash
# Force run, ORACLE ONLY (pic_a3 single-layer overflows its ratio-sized imp pool
# under FORCE_HIT_HEAD on multi-chunk; oracle's keepalive bumps max_imp_len to
# full_len so it's safe). hotpotqa pool N=20, FORCE_HIT_HEAD=128.
set -x
cd /root/sglang
export PATH=/opt/dynamo/venv/bin:$PATH
export PYTHONPATH=/root/sglang/python
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export PIC_BENCH_TOKENIZER=/workspace/models/GLM-5.2-FP8
export PIC_A3_FORCE_HIT_HEAD_IMP=128
DS="${1:-hotpotqa}"; N="${2:-20}"
echo "###### FORCE-ORACLE HEAD=128 $DS pool N=$N START $(date) ######"
python quick_test_online.py --dataset "$DS" --n-samples "$N" \
  --modes pic_a3_oracle --oracle-layers 1 20 40 60 --tp 8
echo "###### DONE rc=$? $(date) ######"
