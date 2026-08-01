#!/usr/bin/env bash
# Test: does recomputing the segment HEADS (the "bright"/attention-sink positions)
# reduce error? baseline oracle vs oracle+PIC_A3_FORCE_HIT_HEAD_IMP=128, vs
# aligned ref (pic_a3+FORCE_ALL_IMP). Measures weighted-K error + output ids.
set -x
cd /root/sglang
export PATH=/opt/dynamo/venv/bin:$PATH
export PYTHONPATH=/root/sglang/python
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export PIC_BENCH_TOKENIZER=/workspace/models/GLM-5.2-FP8
export SGLANG_PIC_KDUMP_ALL=1
export SGLANG_PIC_KDUMP_IMP=1
unset PIC_A3_CLIP_REAL_L1        # default 1 = real L1
unset PIC_A3_FORCE_ALL_IMP
unset PIC_A3_FORCE_HIT_HEAD_IMP

OUT=/tmp/kd_hh; HH=/tmp/kd_hh_force; RF=/tmp/kd_hh_ref
rm -rf $OUT $HH $RF; mkdir -p $OUT

# R1 baseline: full_recompute (text ref) + pic_a3 + pic_a3_oracle (real L1, no force)
export SGLANG_PIC_KDUMP_DIR=$OUT
echo "###### R1 baseline START $(date) ######"
python quick_test_online.py --modes full_recompute pic_a3 pic_a3_oracle --oracle-layers 1 20 40 60 --tp 8
echo "###### R1 DONE rc=$? $(date) ######"

# R2 force: full_recompute + pic_a3_oracle with FORCE_HIT_HEAD=128
export SGLANG_PIC_KDUMP_DIR=$HH
export PIC_A3_FORCE_HIT_HEAD_IMP=128
echo "###### R2 force-hit-head=128 START $(date) ######"
python quick_test_online.py --modes full_recompute pic_a3_oracle --oracle-layers 1 20 40 60 --tp 8
echo "###### R2 DONE rc=$? $(date) ######"
unset PIC_A3_FORCE_HIT_HEAD_IMP
n=0
for f in $HH/pic_a3_oracle_L*.pt $HH/pic_a3_oracle_Q_*.pt $HH/pic_a3_oracle_IMP_*.pt; do
  [ -e "$f" ] || continue; b=$(basename "$f"); cp "$f" "$OUT/oracle_hh_${b#pic_a3_oracle_}"; n=$((n+1))
done
echo "###### merged $n oracle_hh files ######"

# R3 aligned ref: pic_a3 + FORCE_ALL_IMP
export SGLANG_PIC_KDUMP_DIR=$RF
export PIC_A3_FORCE_ALL_IMP=1
echo "###### R3 ref FORCE_ALL_IMP START $(date) ######"
python quick_test_online.py --modes pic_a3 --oracle-layers 1 20 40 60 --tp 8
echo "###### R3 DONE rc=$? $(date) ######"
unset PIC_A3_FORCE_ALL_IMP
n=0
for f in $RF/pic_a3_L*.pt $RF/pic_a3_Q_*.pt; do
  [ -e "$f" ] || continue; b=$(basename "$f"); cp "$f" "$OUT/fullref_${b#pic_a3_}"; n=$((n+1))
done
echo "###### merged $n fullref files ######"

echo "###### inventory ######"
ls $OUT | sed -E 's/_L[0-9].*//;s/_Q_.*//;s/_IMP.*//;s/_A_.*//;s/_H_.*//' | sort | uniq -c

export SGLANG_PIC_KDUMP_DIR=""
echo "###### WKV suppression + weighted error: baseline vs force (vs fullref) ######"
KD=$OUT REF=fullref MODES=pic_a3_oracle,oracle_hh OUT=$OUT/wkv_hh.png python kdump_wkv_clear.py
echo "###### segment-head raw K error: baseline vs force ######"
python kdump_hh_headcheck.py
echo "###### ALL DONE $(date) ######"
