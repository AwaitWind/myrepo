#!/usr/bin/env bash
# Does FORCE_HIT_HEAD rescue the oracle's multi-chunk dataset collapse?
# hotpotqa pool: memory baseline oracle F1=0.000 (collapse), pic_a3=0.807.
# baseline (no force) vs FORCE_HIT_HEAD=128. NOTE force now also hits pic_a3
# (pick_imp gained the hook) — bonus: does head-force help pic_a3 too?
set -x
cd /root/sglang
export PATH=/opt/dynamo/venv/bin:$PATH
export PYTHONPATH=/root/sglang/python
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export PIC_BENCH_TOKENIZER=/workspace/models/GLM-5.2-FP8
DS="${1:-hotpotqa}"; N="${2:-20}"
unset PIC_A3_FORCE_HIT_HEAD_IMP
unset PIC_A3_FORCE_ALL_IMP
unset PIC_A3_CLIP_REAL_L1        # default 1 (real L1)

echo "###### BASELINE $DS pool N=$N START $(date) ######"
python quick_test_online.py --dataset "$DS" --n-samples "$N" \
  --modes full_recompute pic_a3 pic_a3_oracle --oracle-layers 1 20 40 60 --tp 8
echo "###### BASELINE DONE rc=$? $(date) ######"

export PIC_A3_FORCE_HIT_HEAD_IMP=128
echo "###### FORCE HEAD=128 $DS pool N=$N START $(date) ######"
python quick_test_online.py --dataset "$DS" --n-samples "$N" \
  --modes pic_a3 pic_a3_oracle --oracle-layers 1 20 40 60 --tp 8
echo "###### FORCE DONE rc=$? $(date) ######"
unset PIC_A3_FORCE_HIT_HEAD_IMP
echo "###### ALL DONE $(date) ######"
