#!/usr/bin/env bash
# run_pic_hit_kv_error_diag.sh — 一键跑 PIC 命中缓存 KV 误差诊断
#
# 用法(所有配置都可用环境变量覆盖):
#   ./scripts/run_pic_hit_kv_error_diag.sh
#
# 常用变体:
#   # 换模型
#   MODEL=/path/to/model ./scripts/run_pic_hit_kv_error_diag.sh
#
#   # 换 tp
#   TP=4 ./scripts/run_pic_hit_kv_error_diag.sh
#
#   # 只重画,复用已有 capture(秒级出图)
#   SKIP_CAPTURE=1 ./scripts/run_pic_hit_kv_error_diag.sh
#
#   # 导出 CSV
#   CSV=err.csv ./scripts/run_pic_hit_kv_error_diag.sh
#
#   # 不出 nope/pe 分开的子图
#   SEPARATE_NOPE_PE=0 ./scripts/run_pic_hit_kv_error_diag.sh
#
#   # 传额外参数给 python 脚本(比如自定义 capture 目录)
#   ./scripts/run_pic_hit_kv_error_diag.sh -- --capture-dir-a /custom/dir

set -euo pipefail

# ── 打印帮助 ──
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,25p' "$0" | sed 's/^#\s\{0,1\}//'
    exit 0
fi

# ── 定位诊断脚本(相对本脚本路径,允许从任何 cwd 调用)──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAG_PY="${SCRIPT_DIR}/../test/manual/pic_hit_kv_error_diag.py"

if [[ ! -f "$DIAG_PY" ]]; then
    echo "[ERROR] 找不到诊断脚本: $DIAG_PY" >&2
    exit 1
fi

# ── 默认配置(与 quick_test_online.py 保持一致) ──
MODEL="${MODEL:-/workspace/models/GLM-5.2-FP8}"
TP="${TP:-8}"
OUTPUT="${OUTPUT:-pic_hit_kv_error.png}"
MEM_FRAC="${MEM_FRAC:-0.82}"
CAPTURE_DIR_A="${CAPTURE_DIR_A:-/tmp/errdiag_warmup2}"
CAPTURE_DIR_B="${CAPTURE_DIR_B:-/tmp/errdiag_test}"
CSV="${CSV:-}"
SEPARATE_NOPE_PE="${SEPARATE_NOPE_PE:-1}"   # 1=出 nope/pe 分开子图(推荐), 0=只出 2 面板
SKIP_CAPTURE="${SKIP_CAPTURE:-0}"           # 1=跳过 Engine 捕获,直接用现有 capture 目录
KEEP_SHARDS="${KEEP_SHARDS:-0}"             # 1=跑完不清理 capture 目录

# ── 检查模型 ──
if [[ "$SKIP_CAPTURE" != "1" && ! -d "$MODEL" ]]; then
    echo "[ERROR] 模型目录不存在: $MODEL" >&2
    echo "        用 MODEL=/path/to/model 覆盖,或跳过 capture: SKIP_CAPTURE=1" >&2
    exit 2
fi

# ── CUDA 13 兼容库路径 & DeepGEMM fast warmup ──
# 与 quick_test_online.py::_start_server 保持一致,防止 sglang.Engine 起不来。
CU13_LIB="/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
LD_PATH="/usr/local/cuda-13.0/compat:${CU13_LIB}"
export LD_LIBRARY_PATH="${LD_PATH}:${LD_LIBRARY_PATH:-}"
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP="${SGLANG_JIT_DEEPGEMM_FAST_WARMUP:-1}"

# ── 打印配置 ──
cat <<EOF
================================================================
  PIC hit-KV error diagnostic
================================================================
  Model             : $MODEL
  TP                : $TP
  Output PNG        : $OUTPUT
  Mem fraction      : $MEM_FRAC
  Capture dir A     : $CAPTURE_DIR_A   (warmup2 = SYS+C3+Q)
  Capture dir B     : $CAPTURE_DIR_B   (test    = SYS+C1+C2+C3+Q)
  Separate nope/pe  : $SEPARATE_NOPE_PE
  Skip capture      : $SKIP_CAPTURE
  Keep shards       : $KEEP_SHARDS
$(if [[ -n "$CSV" ]]; then echo "  CSV output        : $CSV"; fi)
================================================================

Env:
  LD_LIBRARY_PATH prefix: ${LD_PATH}
  SGLANG_JIT_DEEPGEMM_FAST_WARMUP=${SGLANG_JIT_DEEPGEMM_FAST_WARMUP}

EOF

# ── 组装 python 命令 ──
CMD=(
    python "$DIAG_PY"
    --model         "$MODEL"
    --tp            "$TP"
    --output        "$OUTPUT"
    --mem-fraction  "$MEM_FRAC"
    --capture-dir-a "$CAPTURE_DIR_A"
    --capture-dir-b "$CAPTURE_DIR_B"
)

[[ "$SEPARATE_NOPE_PE" == "1" ]] && CMD+=(--separate-nope-pe)
[[ "$SKIP_CAPTURE"     == "1" ]] && CMD+=(--skip-capture)
[[ "$KEEP_SHARDS"      == "1" ]] && CMD+=(--keep-shards)
[[ -n "$CSV"                   ]] && CMD+=(--csv "$CSV")

# 允许 `-- --extra --args-to-python` 透传
if [[ "${1:-}" == "--" ]]; then
    shift
    CMD+=("$@")
fi

# ── 打印最终命令 & 计时执行 ──
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
    echo "    Output PNG: $OUTPUT"
    [[ -n "$CSV" ]] && echo "    CSV      : $CSV"
    if [[ -f "$OUTPUT" ]]; then
        echo "    Size     : $(du -h "$OUTPUT" | cut -f1)"
    fi
    echo "================================================================"
else
    echo "[FAIL] 退出码 $RC" >&2
    exit $RC
fi
