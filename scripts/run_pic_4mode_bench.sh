#!/usr/bin/env bash
# run_pic_4mode_bench.sh — 一键跑 full_recompute / pic / pic_a3 / pic_cacheblend
# 四模式 TTFT 对比,产出终端汇总表 + PNG 图片(TTFT + 命中率 + 段命中 + 加速比)。
#
# 用法(所有配置都可用环境变量覆盖):
#   ./scripts/run_pic_4mode_bench.sh
#
# 常用变体:
#   # 换模型
#   MODEL=/path/to/model ./scripts/run_pic_4mode_bench.sh
#
#   # 换 tp
#   TP=4 ./scripts/run_pic_4mode_bench.sh
#
#   # 换输出 PNG 路径
#   OUTPUT=/tmp/mybench.png ./scripts/run_pic_4mode_bench.sh
#
#   # 只跑部分 mode
#   MODES="full_recompute pic" ./scripts/run_pic_4mode_bench.sh
#
#   # 换 recomp_ratio(pic_a3 / pic_cacheblend 重算比例)
#   RECOMP_RATIO=0.30 ./scripts/run_pic_4mode_bench.sh
#
#   # 透传额外参数给 quick_test_online.py(注意 `--` 分隔)
#   ./scripts/run_pic_4mode_bench.sh -- --dataset hotpotqa --n-samples 5

set -euo pipefail

# ── 打印帮助 ──
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,25p' "$0" | sed 's/^#\s\{0,1\}//'
    exit 0
fi

# ── 定位主脚本(相对本脚本路径,允许从任何 cwd 调用)──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_PY="${REPO_ROOT}/quick_test_online.py"

if [[ ! -f "$BENCH_PY" ]]; then
    echo "[ERROR] 找不到主脚本: $BENCH_PY" >&2
    exit 1
fi

# ── 默认配置(与 quick_test_online.py 保持一致) ──
MODEL="${MODEL:-/workspace/models/GLM-5.2-FP8}"
TP="${TP:-8}"
PORT="${PORT:-30001}"
MEM_FRAC="${MEM_FRAC:-0.82}"
OUTPUT="${OUTPUT:-pic_4mode_bench.png}"
MODES="${MODES:-full_recompute pic pic_a3 pic_cacheblend}"
RECOMP_RATIO="${RECOMP_RATIO:-0.15}"

# ── 检查模型 ──
if [[ ! -d "$MODEL" ]]; then
    echo "[ERROR] 模型目录不存在: $MODEL" >&2
    echo "        用 MODEL=/path/to/model 覆盖" >&2
    exit 2
fi

# ── CUDA 13 兼容库路径 & DeepGEMM fast warmup ──
# 与 quick_test_online.py::_start_server 保持一致,防止 sglang server 起不来。
CU13_LIB="/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
LD_PATH="/usr/local/cuda-13.0/compat:${CU13_LIB}"
export LD_LIBRARY_PATH="${LD_PATH}:${LD_LIBRARY_PATH:-}"
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP="${SGLANG_JIT_DEEPGEMM_FAST_WARMUP:-1}"

# ★ 强制 sglang 从 /root/sglang 加载(而不是 venv 里 editable 装的 /root/RedKnot 版本)。
# venv site-packages 里的 __editable__.sglang-0.0.0.dev1+ge7768efcd 会让
# `import sglang` 默认走 /root/RedKnot,那里没有 --pic-enable / --enable-a3 等 flag,
# 服务器 argparse 会直接 error 退出(returncode=2)。
# 加 PYTHONPATH 前置就能让本地改动的 /root/sglang 版本优先被 import。
# 允许用户通过 SGLANG_REPO_ROOT 覆盖(比如切到别的 clone)。
SGLANG_REPO_ROOT="${SGLANG_REPO_ROOT:-${REPO_ROOT}}"
export PYTHONPATH="${SGLANG_REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

# quick_test_online.py 也读这几个环境变量做默认值,导出让主进程/子进程都拿到。
export PIC_MODEL="$MODEL"
export PIC_PORT="$PORT"
export MEM_FRAC

# ── 打印配置 ──
cat <<EOF
================================================================
  PIC 4-mode TTFT benchmark
================================================================
  Model             : $MODEL
  TP                : $TP
  Port              : $PORT
  Mem fraction      : $MEM_FRAC
  Output PNG        : $OUTPUT
  Modes             : $MODES
  a3 recomp_ratio   : $RECOMP_RATIO
================================================================

Env:
  LD_LIBRARY_PATH prefix: ${LD_PATH}
  SGLANG_JIT_DEEPGEMM_FAST_WARMUP=${SGLANG_JIT_DEEPGEMM_FAST_WARMUP}

EOF

# ── 组装 python 命令 ──
# MODES 用 read -a 展开成数组,支持 "full_recompute pic pic_a3 pic_cacheblend"
# 这种以空格分隔的多值 env。
read -r -a MODE_ARR <<< "$MODES"

CMD=(
    python "$BENCH_PY"
    --model         "$MODEL"
    --tp            "$TP"
    --port          "$PORT"
    --modes         "${MODE_ARR[@]}"
    --a3-recomp-ratio "$RECOMP_RATIO"
    --plot-output   "$OUTPUT"
)

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
    if [[ -f "$OUTPUT" ]]; then
        echo "    Size     : $(du -h "$OUTPUT" | cut -f1)"
    fi
    echo "================================================================"
else
    echo "[FAIL] 退出码 $RC" >&2
    exit $RC
fi
