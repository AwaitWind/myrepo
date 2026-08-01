#!/usr/bin/env bash
# Decisive: is the oracle-vs-pic_a3 front-layer (2-19) K difference from a DIFFERENT
# imp set, or just FP8 run-to-run noise? Dump imp indices + K for pic_a3 & oracle,
# plus a SECOND pic_a3 run (separate server) as the FP8 noise floor.
set -x
cd /root/sglang
export PATH=/opt/dynamo/venv/bin:$PATH
export PYTHONPATH=/root/sglang/python
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export PIC_BENCH_TOKENIZER=/workspace/models/GLM-5.2-FP8
export SGLANG_PIC_KDUMP_ALL=1
export SGLANG_PIC_KDUMP_IMP=1
unset PIC_A3_CLIP_REAL_L1
unset PIC_A3_FORCE_ALL_IMP
unset PIC_A3_FORCE_HIT_HEAD_IMP

OUT=/tmp/kd_imp2; B=/tmp/kd_imp2b
rm -rf "$OUT" "$B"; mkdir -p "$OUT"

# run 1: pic_a3 + oracle (real-L1)
export SGLANG_PIC_KDUMP_DIR="$OUT"
echo "###### run1 [pic_a3 pic_a3_oracle] START $(date) ######"
python quick_test_online.py --modes pic_a3 pic_a3_oracle --oracle-layers 1 20 40 60 --tp 8
echo "###### run1 DONE rc=$? $(date) ######"

# run 2: pic_a3 again (separate server) = FP8 noise floor
export SGLANG_PIC_KDUMP_DIR="$B"
echo "###### run2 [pic_a3 noise-floor] START $(date) ######"
python quick_test_online.py --modes pic_a3 --oracle-layers 1 20 40 60 --tp 8
echo "###### run2 DONE rc=$? $(date) ######"
n=0
for f in "$B"/pic_a3_L*.pt "$B"/pic_a3_IMP_*.pt; do
  [ -e "$f" ] || continue
  b=$(basename "$f"); cp "$f" "$OUT/pic_a3b_${b#pic_a3_}"; n=$((n+1))
done
echo "###### merged $n pic_a3b files ######"

echo "###### inventory ######"
ls "$OUT" | sed -E 's/_L[0-9].*//;s/_IMP.*//;s/_Q_.*//;s/_A_.*//;s/_H_.*//' | sort | uniq -c
export SGLANG_PIC_KDUMP_DIR=""
echo "###### ANALYSIS ######"
python kdump_imp_diag.py
echo "###### ALL DONE $(date) ######"
