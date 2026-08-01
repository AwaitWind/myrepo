#!/bin/bash
# =========================================================================
# GLM5.2 PIC vs 标准 SGLang Prefill 耗时对比脚本
#
# 用法:
#   bash benchmark_pic.sh
#
# 自动执行:
#   1. 启动标准 SGLang 服务
#   2. 发送测试请求，测量 TTFT
#   3. 停止标准服务
#   4. 启动 PIC SGLang 服务
#   5. 发送测试请求，测量 TTFT
#   6. 停止 PIC 服务
#   7. 输出对比结果
# =========================================================================

set -euo pipefail

# ===================== 配置 =====================
MODEL_PATH="${MODEL_PATH:-/workspace/models/GLM-5.2-FP8}"
TP="${TP:-8}"
STD_PORT=30001
PIC_PORT=30002
NUM_ROUNDS=3
RESULTS_FILE="/tmp/pic_benchmark_results.txt"

# CUDA 环境
CU13_LIB="/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${CU13_LIB}:${LD_LIBRARY_PATH:-}"

# ===================== 工具函数 =====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*"; }

# ===================== HTTP 请求函数 =====================

api_request() {
    # $1: port, $2: prompt, $3: max_tokens (默认 10)
    local port=$1
    local prompt=$2
    local max_tokens=${3:-10}
    curl -s --max-time 120 \
        "http://127.0.0.1:${port}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"default\",
            \"prompt\": $(python3 -c "import json; print(json.dumps('$prompt'))"),
            \"max_tokens\": ${max_tokens},
            \"temperature\": 0
        }"
}

measure_ttft() {
    # $1: port, $2: prompt
    # 返回 TTFT (毫秒)
    local port=$1
    local prompt=$2
    local start_ms end_ms

    start_ms=$(python3 -c "import time; print(int(time.monotonic()*1000))")
    api_request "$port" "$prompt" 5 > /dev/null 2>&1
    end_ms=$(python3 -c "import time; print(int(time.monotonic()*1000))")
    echo $((end_ms - start_ms))
}

wait_for_server() {
    local port=$1
    local timeout=${2:-600}
    local elapsed=0
    log "等待服务端口 ${port}..."
    while [ $elapsed -lt $timeout ]; do
        if curl -s --max-time 3 "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; then
            ok "服务就绪 (端口 ${port})"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
    done
    echo ""
    err "服务启动超时 (端口 ${port})"
    return 1
}

kill_server() {
    local port=$1
    local pid
    pid=$(lsof -ti :"${port}" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        log "关闭端口 ${port} 上的服务 (PID: $pid)"
        kill "$pid" 2>/dev/null || true
        sleep 3
        # 确保清理
        kill -9 "$pid" 2>/dev/null || true
    fi
}

# ===================== 测试 Prompts =====================

SEP="<<PIC_SEP>>"

SHORT_DOC="The capital of France is Paris. France is a country in Western Europe."
MEDIUM_DOC="The transformer architecture was introduced in 2017 by Vaswani et al. It uses self-attention mechanisms to process sequential data without recurrence. The key innovation is multi-head attention."
LONG_DOC="Machine learning is a subset of artificial intelligence that enables systems to learn from experience. Deep learning uses neural networks with many layers. Large language models like GPT and Claude are built on transformer architectures and have demonstrated remarkable capabilities in understanding and generating human language."

# 测试用例: (name, "docs...", "question")
TEST_CASES=(
    "1doc|${SHORT_DOC}|What is the capital of France?"
    "2docs|${SHORT_DOC}|${MEDIUM_DOC}|What is the capital of France? Who introduced the transformer?"
    "3docs|${SHORT_DOC}|${MEDIUM_DOC}|${LONG_DOC}|Summarize all three documents."
)

# ===================== 启动服务 =====================

launch_server() {
    local port=$1
    local pic_enable=$2
    local log_file=$3

    local pic_flag=""
    if [ "$pic_enable" = "true" ]; then
        pic_flag="--pic-enable --pic-mode addition"
    fi

    log "启动服务: port=${port} pic=${pic_enable}"
    python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --tp "${TP}" \
        --trust-remote-code \
        --mem-fraction-static 0.82 \
        --cuda-graph-max-bs 256 \
        --reasoning-parser glm45 \
        --tool-call-parser glm47 \
        --host 0.0.0.0 \
        --port "${port}" \
        ${pic_flag} \
        > "${log_file}" 2>&1 &

    local server_pid=$!
    echo "${server_pid}"
}

# ===================== 运行 Benchmark =====================

run_benchmark() {
    local port=$1
    local mode=$2
    local output_file=$3

    echo "" > "$output_file"
    echo "mode=${mode}" >> "$output_file"
    echo "port=${port}" >> "$output_file"
    echo "rounds=${NUM_ROUNDS}" >> "$output_file"
    echo "------------------------" >> "$output_file"

    log "开始 ${mode} benchmark (${NUM_ROUNDS} 轮)..."

    # 预热
    log "  预热..."
    for i in 1 2 3; do
        measure_ttft "$port" "Hello world $i" > /dev/null
    done
    sleep 2

    local total_ttft=0
    local test_count=0

    for test_case in "${TEST_CASES[@]}"; do
        IFS='|' read -r name docs_str question <<< "$test_case"

        # 构建 prompt
        local prompt=""
        if [ "$mode" = "pic" ]; then
            # PIC 模式: 用 <<PIC_SEP>> 分隔
            local parts=()
            IFS='|' read -ra docs <<< "$docs_str"
            for doc in "${docs[@]}"; do
                parts+=("$doc")
            done
            prompt=$(printf "${SEP}" "${parts[@]}")
            prompt="${prompt}${SEP}${question}"
        else
            # 标准模式: 用换行分隔
            prompt="${docs_str}\n\n${question}"
        fi

        log "  [${name}] 测试中..."
        local ttfts=()
        for r in $(seq 1 ${NUM_ROUNDS}); do
            local ttft
            ttft=$(measure_ttft "$port" "$prompt")
            ttfts+=("$ttft")
            log "    round ${r}: TTFT=${ttft}ms"
        done

        # 计算平均
        local sum=0
        for t in "${ttfts[@]}"; do sum=$((sum + t)); done
        local avg=$((sum / ${#ttfts[@]}))

        echo "${name}|${avg}|${ttfts[*]}" >> "$output_file"
        total_ttft=$((total_ttft + sum))
        test_count=$((test_count + ${#ttfts[@]}))
    done

    local overall_avg=$((total_ttft / test_count))
    echo "OVERALL_AVG|${overall_avg}" >> "$output_file"
    ok "${mode} benchmark 完成, 整体平均 TTFT=${overall_avg}ms"
}

# ===================== 结果对比 =====================

compare_results() {
    local std_file=$1
    local pic_file=$2

    echo ""
    echo "================================================================================"
    echo "  GLM5.2 Prefill 耗时对比: 标准 SGLang vs PIC 模式"
    echo "================================================================================"
    echo ""
    printf "  %-15s %12s %12s %10s %10s\n" "Benchmark" "标准TTFT" "PIC TTFT" "节省" "提升"
    printf "  %-15s %12s %12s %10s %10s\n" "---------------" "------------" "------------" "----------" "----------"

    local total_std=0 total_pic=0 count=0

    while IFS='|' read -r name std_avg rest; do
        if [ "$name" = "OVERALL_AVG" ] || [ "$name" = "mode" ] || [ "$name" = "port" ] || [ "$name" = "rounds" ] || [ "$name" = "------------------------" ] || [ -z "$name" ]; then
            continue
        fi
        local pic_avg
        pic_avg=$(grep "^${name}|" "$pic_file" | cut -d'|' -f2)
        if [ -z "$pic_avg" ]; then
            continue
        fi

        local saving=$((std_avg - pic_avg))
        local ratio=0
        if [ "$std_avg" -gt 0 ]; then
            ratio=$(python3 -c "print(f'{${saving}/${std_avg}*100:.1f}')")
        fi

        printf "  %-15s %9sms %9sms %+8sms %+8s%%\n" \
            "$name" "$std_avg" "$pic_avg" "$saving" "$ratio"

        total_std=$((total_std + std_avg))
        total_pic=$((total_pic + pic_avg))
        count=$((count + 1))
    done < "$std_file"

    if [ "$count" -gt 0 ]; then
        local total_saving=$((total_std - total_pic))
        local total_ratio=0
        if [ "$total_std" -gt 0 ]; then
            total_ratio=$(python3 -c "print(f'{${total_saving}/${total_std}*100:.1f}')")
        fi
        printf "  %-15s %12s %12s %10s %10s\n" "---------------" "------------" "------------" "----------" "----------"
        printf "  %-15s %9sms %9sms %+8sms %+8s%%\n" \
            "总计" "$total_std" "$total_pic" "$total_saving" "$total_ratio"
    fi

    echo ""
    echo "  TTFT 越低 = prefill 越快 = 越好"
    echo ""

    if [ "$total_ratio" != "0" ]; then
        local ratio_num
        ratio_num=$(python3 -c "print(float($total_ratio))")
        if python3 -c "exit(0 if float($ratio_num) > 0 else 1)" 2>/dev/null; then
            echo "  ✅ PIC 模式平均节省 ${total_ratio}%"
        else
            echo "  ⚠️  PIC 模式反而更慢。可能原因: 段间无共享前缀，缓存未命中"
        fi
    fi
}

# ===================== 主流程 =====================

main() {
    local std_log="/tmp/sglang_std_server.log"
    local pic_log="/tmp/sglang_pic_server.log"
    local std_result="/tmp/bench_std.txt"
    local pic_result="/tmp/bench_pic.txt"

    echo ""
    echo "================================================================================"
    echo "  GLM5.2 PIC vs 标准 Prefill 对比测试"
    echo "  Model: ${MODEL_PATH}"
    echo "  TP: ${TP}"
    echo "================================================================================"
    echo ""

    # ── 阶段 1: 标准 SGLang ──
    log "=========================================="
    log "  阶段 1/2: 标准 SGLang (无 PIC)"
    log "=========================================="

    kill_server ${STD_PORT}
    local std_pid
    std_pid=$(launch_server ${STD_PORT} "false" "$std_log")

    if ! wait_for_server ${STD_PORT} 600; then
        err "标准服务启动失败，查看日志: ${std_log}"
        kill "$std_pid" 2>/dev/null || true
        exit 1
    fi

    run_benchmark ${STD_PORT} "standard" "$std_result"

    log "关闭标准服务..."
    kill "$std_pid" 2>/dev/null || true
    sleep 5
    kill_server ${STD_PORT}

    echo ""

    # ── 阶段 2: PIC SGLang ──
    log "=========================================="
    log "  阶段 2/2: PIC SGLang (--pic-enable)"
    log "=========================================="

    kill_server ${PIC_PORT}
    local pic_pid
    pic_pid=$(launch_server ${PIC_PORT} "true" "$pic_log")

    if ! wait_for_server ${PIC_PORT} 600; then
        err "PIC 服务启动失败，查看日志: ${pic_log}"
        kill "$pic_pid" 2>/dev/null || true
        exit 1
    fi

    run_benchmark ${PIC_PORT} "pic" "$pic_result"

    log "关闭 PIC 服务..."
    kill "$pic_pid" 2>/dev/null || true
    sleep 5
    kill_server ${PIC_PORT}

    echo ""

    # ── 对比 ──
    compare_results "$std_result" "$pic_result"

    echo ""
    echo "  详细日志:"
    echo "    标准服务: ${std_log}"
    echo "    PIC 服务: ${pic_log}"
    echo "    原始结果: ${std_result} / ${pic_result}"
    echo ""
}

main