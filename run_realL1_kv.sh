#!/usr/bin/env bash
# KV-error dashboard for the "layer-1 = real in-context (like pic_a3)" change.
# A/B: pic_a3 (single-layer baseline) | pic_a3_oracle (NEW real-L1 + deep isolated)
#      | oracle_iso (OLD isolated-L1, PIC_A3_CLIP_REAL_L1=0) | full_recompute (ref).
# Ref = pic_a3 + FORCE_ALL_IMP (all positions fresh on the padded prompt =
# byte-identical to full recompute, position-aligned). Plots cos + euclidean + relL2.
set -x
cd /root/sglang
export PATH=/opt/dynamo/venv/bin:$PATH
export PYTHONPATH=/root/sglang/python
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export PIC_BENCH_TOKENIZER=/workspace/models/GLM-5.2-FP8
export SGLANG_PIC_KDUMP_ALL=1
unset PIC_A3_FORCE_HIT_HEAD_IMP

OUT=/tmp/kd_realL1
ISO=/tmp/kd_realL1_iso
REF=/tmp/kd_realL1_ref
rm -rf "$OUT" "$ISO" "$REF"
mkdir -p "$OUT"

# ---- Run A: pic_a3 + pic_a3_oracle (NEW default: real in-context L1) ----
export SGLANG_PIC_KDUMP_DIR="$OUT"
unset PIC_A3_CLIP_REAL_L1        # code default is now =1 (real L1)
unset PIC_A3_FORCE_ALL_IMP
echo "###### A real-L1 [pic_a3 pic_a3_oracle] START $(date) ######"
python quick_test_online.py --modes pic_a3 pic_a3_oracle --oracle-layers 1 20 40 60 --tp 8
echo "###### A DONE rc=$? $(date) ######"

# ---- Run B: pic_a3_oracle OLD behavior (isolated L1) ----
export SGLANG_PIC_KDUMP_DIR="$ISO"
export PIC_A3_CLIP_REAL_L1=0
echo "###### B iso-L1 [pic_a3_oracle] START $(date) ######"
python quick_test_online.py --modes pic_a3_oracle --oracle-layers 1 20 40 60 --tp 8
echo "###### B DONE rc=$? $(date) ######"
unset PIC_A3_CLIP_REAL_L1
n=0
for f in "$ISO"/pic_a3_oracle_L*.pt; do
  [ -e "$f" ] || continue
  b=$(basename "$f"); cp "$f" "$OUT/oracle_iso_${b#pic_a3_oracle_}"; n=$((n+1))
done
echo "###### merged $n iso-L1 K files ######"

# ---- Run C: ground-truth ref = pic_a3 + FORCE_ALL_IMP (all fresh) ----
export SGLANG_PIC_KDUMP_DIR="$REF"
export PIC_A3_FORCE_ALL_IMP=1
echo "###### C ref FORCE_ALL_IMP [pic_a3] START $(date) ######"
python quick_test_online.py --modes pic_a3 --oracle-layers 1 20 40 60 --tp 8
echo "###### C DONE rc=$? $(date) ######"
unset PIC_A3_FORCE_ALL_IMP
n=0
for f in "$REF"/pic_a3_L*.pt; do
  [ -e "$f" ] || continue
  b=$(basename "$f"); cp "$f" "$OUT/full_recompute_${b#pic_a3_}"; n=$((n+1))
done
echo "###### merged $n ref K files ######"

# ---- inventory ----
echo "###### inventory (tag -> file count) ######"
ls "$OUT" | sed -E 's/_L[0-9].*//;s/_Q_.*//;s/_A_.*//;s/_H_.*//' | sort | uniq -c

# ---- plots: cosine, raw euclidean, relative-L2 ----
export SGLANG_PIC_KDUMP_DIR=""
for M in cos euclidean l2; do
  echo "###### PLOT METRIC=$M ######"
  KD=$OUT REF=full_recompute MODES=pic_a3,pic_a3_oracle,oracle_iso METRIC=$M \
    OUT=$OUT/kv_err_$M.png python kdump_heatmap.py
done
echo "###### PLOTS DONE rc=$? ######"
ls -la "$OUT"/kv_err_*.png
echo "###### ALL DONE $(date) ######"
