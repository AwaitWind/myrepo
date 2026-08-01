#!/usr/bin/env bash
# run_pic_a3_layer01_error.sh — 一键跑 pic_a3 vs full_recompute layer 0/1 K 误差诊断
#
# 用法:
#   ./scripts/run_pic_a3_layer01_error.sh
#
# 常用变体:
#   MODEL=/other/model ./scripts/run_pic_a3_layer01_error.sh
#   TP=4 ./scripts/run_pic_a3_layer01_error.sh
#   SKIP_CAPTURE=1 ./scripts/run_pic_a3_layer01_error.sh    # 秒级出图
#   LAYERS="0 1 2 3" ./scripts/run_pic_a3_layer01_error.sh  # 扩展到更多层
#   CSV=err.csv PLOT=err.png ./scripts/run_pic_a3_layer01_error.sh

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,15p' "$0" | sed 's/^#\s\{0,1\}//'
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAG_PY="${SCRIPT_DIR}/pic_a3_layer01_error.py"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "$DIAG_PY" ]]; then
    echo "[ERROR] 找不到诊断脚本: $DIAG_PY" >&2
    exit 1
fi

# ── 默认配置 ──
MODEL="${MODEL:-/workspace/models/GLM-5.2-FP8}"
TP="${TP:-8}"
PORT="${PORT:-30001}"
MEM_FRAC="${MEM_FRAC:-0.82}"
DUMP_ROOT="${DUMP_ROOT:-/tmp/pic_a3_l01_err}"
LAYERS="${LAYERS:-0 1}"
SKIP_CAPTURE="${SKIP_CAPTURE:-0}"
CSV="${CSV:-}"
PLOT="${PLOT:-}"

# ── 保证 sglang 从 /root/sglang 加载 (绕开 RedKnot editable install) ──
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
# ── 强制覆盖 CVD, 不用 :- fallback (调用方 shell 里可能有 CVD='0' 单卡污染,
# 用 :- 会保留污染值; 用户想跑单卡就设 TP=1 而不是自己改 CVD)。可显式传
# CUDA_VISIBLE_DEVICES_KEEP=1 保留调用方的值。──
if [[ "${CUDA_VISIBLE_DEVICES_KEEP:-0}" != "1" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
# ── SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS 同样强制覆盖 (默认 1: 每 worker
# 只看自己那张卡, 更稳)。可显式传 SGLANG_ONE_KEEP=1 保留调用方的值。──
if [[ "${SGLANG_ONE_KEEP:-0}" != "1" ]]; then
    export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
fi
# ── CUDA 13 兼容 & DeepGEMM fast warmup (与 quick_test_online.py 一致) ──
CU13_LIB="/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${CU13_LIB}:${LD_LIBRARY_PATH:-}"
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP="${SGLANG_JIT_DEEPGEMM_FAST_WARMUP:-1}"

cat <<EOF
================================================================
  pic_a3 vs full_recompute — layer 0/1 K error diagnostic
================================================================
  Model             : $MODEL
  TP                : $TP
  Layers to compare : $LAYERS
  Mem fraction      : $MEM_FRAC
  Dump root         : $DUMP_ROOT  (<root>_full_recompute / <root>_pic_a3)
  Skip capture      : $SKIP_CAPTURE
$(if [[ -n "$CSV"  ]]; then echo "  CSV output        : $CSV";  fi)
$(if [[ -n "$PLOT" ]]; then echo "  PNG output        : $PLOT"; fi)

Env:
  PYTHONPATH prefix                     = ${REPO_ROOT}/python
  CUDA_VISIBLE_DEVICES                  = $CUDA_VISIBLE_DEVICES
  SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS = $SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS
  SGLANG_JIT_DEEPGEMM_FAST_WARMUP       = $SGLANG_JIT_DEEPGEMM_FAST_WARMUP
================================================================
EOF

CMD=(
    python "$DIAG_PY"
    --model        "$MODEL"
    --tp           "$TP"
    --port         "$PORT"
    --mem-fraction "$MEM_FRAC"
    --dump-root    "$DUMP_ROOT"
    --layers       $LAYERS
)
[[ "$SKIP_CAPTURE" == "1" ]] && CMD+=(--skip-capture)
[[ -n "$CSV"  ]] && CMD+=(--output-csv  "$CSV")
[[ -n "$PLOT" ]] && CMD+=(--output-plot "$PLOT")

echo
echo "→ ${CMD[*]}"
echo

T0=$(date +%s)
"${CMD[@]}"
RC=$?
DT=$(( $(date +%s) - T0 ))

echo
if [[ $RC -eq 0 ]]; then
    echo "================================================================"
    echo "  ✓ DONE in ${DT}s"
    [[ -n "$CSV"  && -f "$CSV"  ]] && echo "    CSV : $CSV  ($(du -h "$CSV"  | cut -f1))"
    [[ -n "$PLOT" && -f "$PLOT" ]] && echo "    PNG : $PLOT ($(du -h "$PLOT" | cut -f1))"
    echo "================================================================"
else
    echo "[FAIL] 退出码 $RC" >&2
    exit $RC
fi
