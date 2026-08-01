#!/usr/bin/env bash
# run_pic_ratio_sweep.sh
# ─────────────────────────────────────────────────────────────────────────────
# 对 full_recompute / pic / pic_a3 / pic_cacheblend 四模式,在
#   --a3-recomp-ratio ∈ {0.15, 0.3, 0.5, 0.8}
# 下各跑一次 quick_test_online.py (dataset=hotpotqa, n-samples=100, tp=8),
# 每次把完整输出存日志,最后解析出 F1 / TTFT / 命中率 的汇总表 + CSV。
#
# 用法:
#   scripts/run_pic_ratio_sweep.sh
#   RATIOS="0.15 0.3" N_SAMPLES=50 scripts/run_pic_ratio_sweep.sh   # 覆盖
#
# 说明:
#   full_recompute / pic 不依赖 --a3-recomp-ratio,默认每个 ratio 都重跑(带各自
#   baseline,反映 FP8 运行间抖动)。若只想它们跑一次、pic_a3/pic_cacheblend 才扫
#   ratio(省 ~一半时间),设 SKIP_RATIO_INDEP=1。
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail   # 不用 -e:某个 ratio 失败也要继续跑完其余,最后一并汇总

# ── 可覆盖配置 ────────────────────────────────────────────────────────────────
MODEL="${MODEL:-/workspace/models/GLM-5.2-FP8}"
TP="${TP:-8}"
PORT="${PORT:-30001}"
N_SAMPLES="${N_SAMPLES:-100}"
DATASET="${DATASET:-hotpotqa}"
MODES="${MODES:-full_recompute pic pic_a3 pic_cacheblend}"
RATIOS="${RATIOS:-0.15 0.3 0.5 0.8}"
OUTDIR="${OUTDIR:-/tmp/pic_ratio_sweep}"
SKIP_RATIO_INDEP="${SKIP_RATIO_INDEP:-0}"   # 1 = full_recompute/pic 只在首个 ratio 跑

# ── 定位仓库 + 主脚本(相对本脚本路径,允许任意 cwd 调用)──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_PY="${REPO_ROOT}/quick_test_online.py"
[[ -f "$BENCH_PY" ]] || { echo "[ERROR] 找不到主脚本: $BENCH_PY" >&2; exit 1; }
[[ -d "$MODEL" ]]    || { echo "[ERROR] 模型目录不存在: $MODEL (用 MODEL=... 覆盖)" >&2; exit 2; }

# ── 运行环境(与 run_pic_4mode_bench.sh 一致)──
# ★ PYTHONPATH 前置:venv 里 sglang 是 editable 装到别处(/root/RedKnot),
#   不加这个 `import sglang` 走不到本仓库、server 的 --pic-enable 会 argparse 报错。
export PATH="/opt/dynamo/venv/bin:${PATH}"
CU13_LIB="/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${CU13_LIB}:${LD_LIBRARY_PATH:-}"
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP="${SGLANG_JIT_DEEPGEMM_FAST_WARMUP:-1}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

# dataset 用相对路径 data/raw|processed/<name>,必须在仓库根跑
cd "$REPO_ROOT"
mkdir -p "$OUTDIR"

read -r -a RATIO_ARR <<< "$RATIOS"

echo "=================================================================="
echo "  PIC recomp-ratio sweep"
echo "    model     : $MODEL"
echo "    modes     : $MODES"
echo "    ratios    : $RATIOS"
echo "    n-samples : $N_SAMPLES   dataset: $DATASET   tp: $TP   port: $PORT"
echo "    outdir    : $OUTDIR"
echo "    skip-indep: $SKIP_RATIO_INDEP"
echo "=================================================================="

T0=$(date +%s)

# ── 逐 ratio 跑 ──────────────────────────────────────────────────────────────
_first=1
for ratio in "${RATIO_ARR[@]}"; do
    # 选择本 ratio 要跑的 modes
    run_modes="$MODES"
    if [[ "$SKIP_RATIO_INDEP" == "1" && "$_first" != "1" ]]; then
        # 非首个 ratio 只跑吃 ratio 的两个模式
        run_modes="$(echo "$MODES" | tr ' ' '\n' | grep -E '^(pic_a3|pic_cacheblend)$' | tr '\n' ' ')"
    fi
    read -r -a RUN_MODE_ARR <<< "$run_modes"

    log="${OUTDIR}/ratio_${ratio}.log"
    echo
    echo ">>> [$(date '+%F %T')] ratio=${ratio}  modes=[${run_modes}]  → ${log}"
    python "$BENCH_PY" \
        --modes "${RUN_MODE_ARR[@]}" \
        --dataset "$DATASET" \
        --n-samples "$N_SAMPLES" \
        --a3-recomp-ratio "$ratio" \
        --tp "$TP" \
        --port "$PORT" \
        --model "$MODEL" \
        > "$log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "    ✓ done (rc=0)  $(date '+%T')"
    else
        echo "    ✗ [WARN] rc=$rc — 见 $log 末尾"
        tail -n 15 "$log" | sed 's/^/      | /'
    fi
    _first=0
done

DT=$(( $(date +%s) - T0 ))
echo
echo "=================================================================="
echo "  所有 ratio 跑完,用时 ${DT}s ($((DT/60)) 分钟)。解析汇总……"
echo "=================================================================="

# ── 解析每个 ratio 日志的汇总块,拼成 F1 / TTFT / 命中率 表 + CSV ──
python - "$OUTDIR" "$MODES" "$RATIOS" <<'PY'
import sys, os

outdir, modes_s, ratios_s = sys.argv[1], sys.argv[2], sys.argv[3]
modes  = modes_s.split()
ratios = ratios_s.split()

# data[metric][mode][ratio] = 字符串值
data = {"f1": {}, "ttft": {}, "hit": {}}
for d in data.values():
    for m in modes:
        d[m] = {}

for ratio in ratios:
    log = os.path.join(outdir, f"ratio_{ratio}.log")
    if not os.path.exists(log):
        continue
    txt = open(log, errors="replace").read()
    # 取最后一个「测试结果汇总」之后的块(避免抓到逐样本输出里的 mode 名)
    idx = txt.rfind("测试结果汇总")
    block = txt[idx:] if idx >= 0 else txt
    for line in block.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        m = parts[0]
        if m not in modes:
            continue
        # ERROR 行:  "<mode>  ERROR: ..."
        if parts[1].startswith("ERROR"):
            for k in data:
                data[k][m][ratio] = "ERR"
            continue
        # 汇总表数据行(≥6 列): mode f1_mean f1_median ttft_p50 ttft_p95 hit pic_seg
        # 加速比/Δ 行只有 4 列且 parts[1] 是 ":" → 被下面的 float() 过滤掉
        if len(parts) < 6:
            continue
        try:
            f1   = float(parts[1])
            ttft = float(parts[3])
        except ValueError:
            continue
        data["f1"][m][ratio]   = f"{f1:.3f}"
        data["ttft"][m][ratio] = f"{ttft:.3f}"
        data["hit"][m][ratio]  = parts[5]

def render(title, metric):
    colw = 12
    hdr = f"{'mode':<18}" + "".join(f"{'r='+r:<{colw}}" for r in ratios)
    print(f"\n### {title}")
    print(hdr)
    print("-" * len(hdr))
    for m in modes:
        row = f"{m:<18}"
        for r in ratios:
            row += f"{data[metric][m].get(r, '-'):<{colw}}"
        print(row)

render("F1_mean  (越高越好)",              "f1")
render("TTFT_p50 秒  (越低越好)",          "ttft")
render("PIC 命中率_mean",                  "hit")

# CSV(方便后续画图/贴表)
csv = os.path.join(outdir, "summary.csv")
with open(csv, "w") as f:
    f.write("metric,mode," + ",".join(f"ratio_{r}" for r in ratios) + "\n")
    for metric in ("f1", "ttft", "hit"):
        for m in modes:
            f.write(f"{metric},{m}," +
                    ",".join(str(data[metric][m].get(r, "")) for r in ratios) + "\n")
print(f"\nCSV → {csv}")
print("逐 ratio 完整日志 → " + os.path.join(outdir, "ratio_<r>.log"))
PY
