#!/usr/bin/env python3
"""
PIC Online Quick Test — 在线对比测试脚本 (GLM-5.2 版本)

对同一组 prompt 分别测试 full_recompute、prefix_cache、PIC 三种模式，
测量 TTFT（首 token 延迟）、缓存命中率和输出正确性（首个分歧 token FDT）。

测试数据、prompt 构造、测试流程严格遵循 quick_test_online_spec.md。
服务器启动参数使用 GLM-5.2 实际参数（参考 /root/run_prefill_benchmark.sh）。

用法：
    python quick_test_online.py [--port PORT] [--model MODEL_PATH] [--tp TP]

可覆盖的环境变量：
    PIC_MODEL           模型路径（默认 /workspace/models/GLM-5.2-FP8）
    PIC_PORT            端口（默认 30001）
    PIC_READY_TIMEOUT   就绪超时秒数（默认 1500）
    MEM_FRAC            静态内存比例（默认 0.82）
    SGLANG_PY           Python 解释器路径
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

# ============================================================
# 配置常量（规格文档 §2.1 + GLM-5.2 服务器配置）
# ============================================================

MODEL = os.environ.get("PIC_MODEL", "/workspace/models/GLM-5.2-FP8")
PORT: int = int(os.environ.get("PIC_PORT", "30001"))
READY_TIMEOUT: int = int(os.environ.get("PIC_READY_TIMEOUT", "1500"))
SGLANG_PY: str = os.environ.get("SGLANG_PY", sys.executable)
MEM_FRAC: str = os.environ.get("MEM_FRAC", "0.82")
# cuda-graph-max-bs：测试脚本只发单请求（bs=1），不需要大 batch size 的 CUDA graph。
# 设为 4（[1,2,4]）可将 CUDA graph capture 数量最小化。
CUDA_GRAPH_MAX_BS: str = os.environ.get("CUDA_GRAPH_MAX_BS", "4")

# DeepGEMM fast warmup 开关：启用后每个 GEMM shape 的 warmup m_list 只编译 ~3073 个
# （而非 65536 个），启动时间大幅下降。
# 用独立变量名 PIC_FAST_WARMUP 控制，避免读到 shell 中可能残留的
# SGLANG_JIT_DEEPGEMM_FAST_WARMUP=0（那会导致本脚本误以为要禁用 fast warmup）。
FAST_WARMUP: str = os.environ.get("PIC_FAST_WARMUP", "1")

# DeepGEMM fast-warmup stride multiplier：在 FAST_WARMUP 基础上再稀疏化 prefill M 采样。
# pic_a3 / pic_cacheblend 的 A³ imp-clip 路径会为 MHC prenorm gemm 引入额外的
# num_splits shape（每个 num_tokens 变体一次），比 pure pic 多几次每 shape ~3000
# 次的 warmup 迭代，把启动时间往上抬 5-10s。设 stride_mult=4 让每个 shape 的 M
# 采样从 ~3073 降到 ~1281（decode 段 1..1024 保持全覆盖），首次 hit 时若碰到未
# 覆盖的 M 会走 DeepGEMM 单个 JIT compile，代价 ms 级，可接受。
# 设 PIC_FAST_WARMUP_STRIDE_MULT=1 可以关闭该稀疏化。
FAST_WARMUP_STRIDE_MULT: str = os.environ.get("PIC_FAST_WARMUP_STRIDE_MULT", "4")

# 段 64 对齐开关（路径 A 验证）：将每段 token 数 padding 到 64 的倍数。
# GLM-5.2(DSA) 强制 page_size=64，PIC 的段 slot 必须 page 对齐，否则 DSA 的
# paged 寻址(req_to_token[:, ::64] // 64)会越界崩溃。让每段 token 数为 64 的倍数后，
# PIC 现有的 slot 分配会自动 page 对齐，从而在不改内核的前提下验证 PIC 能否跑通。
PIC_PAD_TO_64: str = os.environ.get("PIC_PAD_TO_64", "1")
# 段对齐粒度（与 DSA page_size 一致，固定 64）。
PIC_ALIGN: int = int(os.environ.get("PIC_ALIGN", "64"))

# 分隔符（规格文档 §2.1）
SEP = "<<PIC_SEP>>"
# 系统提示（规格文档 §2.1）—— 直接构造成精确 64 tokens (0 padding),
# 避免 pad 位于段尾污染 attention/generation。见 pic_a3_imp_token_view 诊断:
# 哪怕 Q 尾部有 1 个 pad token, 都会让 full_recompute 生成全套 pad token。
_SYS_UNIT = "You are a helpful AI assistant. "
SYS = _SYS_UNIT * 9  # 精确 64 tokens
# 测试问题（内容型，非“第几个文档”型）。
# ── 旧设计（已弃用）：文档是 "Document B about dogs." × 800 的极端重复 + 问题问
#    "哪个是文档 B"。这对 pic_a3 是病态输入：① 每段只有 ~5 个不同 token，15%
#    稀疏重算分不清 A/B/C；② 位置型问题("第几个文档")需要文档计数，稀疏复用下
#    信号很弱。结果 pic_a3 第一个 token 就发散（FDT=0），常吐 padding 里的 "?"
#    撞 stop → 空输出。
# ── 新设计：三段“内容各异”的地理小百科 + 内容型问题。答案 "Mount Kilimanjaro"
#    是 C2（中间段）里的独特专有名词，query 词 "highest mountain in Africa" 能让
#    A³ 的 imp-selection 精确定位到它 → pic_a3 可正确作答，FDT 才有意义。
Q = (
    "Read the three documents above and answer using only the documents. "
    "Question: What is the highest mountain in Africa? "
    "Reply with only the name, no explanation. Do not repeat.\n\nAnswer:"
)

# ============================================================
# 文档数据：三段内容各异的真实文字（河流 / 山峰 / 沙漠）
# 每段 = 一段独立小百科 × _DOC_REPEAT，用重复达到 ~数千 token（展示 PIC 的 TTFT
# 收益），但每段有 ~150+ 个不同 token（旧设计每段仅 ~5 个），imp-selection 有真实
# 信号可挑。答案落在 C2（中间段），测试中间段的 KV 复用。
# 用 PIC_SYNTH_DOC_REPEAT 调长度（默认 12 → 每段 ~1800 tokens，全 prompt ~5.5k）。
# ============================================================
_DOC_REPEAT = int(os.environ.get("PIC_SYNTH_DOC_REPEAT", "12"))
_C1_BASE = (
    "The Amazon River flows through South America and is widely regarded as "
    "the largest river in the world by the volume of water it discharges. It "
    "rises in the Andes mountains of Peru and runs eastward across Brazil "
    "before emptying into the Atlantic Ocean. Its vast basin holds a rainforest "
    "full of fish, birds, and mammals, and many riverside communities rely on "
    "the water for transport, food, and irrigation. Seasonal floods reshape the "
    "surrounding wetlands every year. "
)
_C2_BASE = (
    "Mount Kilimanjaro is a dormant volcano in Tanzania and the highest "
    "mountain in Africa, rising about 5,895 meters above sea level. Its "
    "snow-capped summit towers over the surrounding savanna and is visible from "
    "a great distance. Climbers from around the world ascend through rainforest, "
    "moorland, and alpine desert to reach the peak. The mountain has three "
    "volcanic cones and has become an enduring symbol of the African continent. "
)
_C3_BASE = (
    "The Sahara is the largest hot desert on Earth, spreading across much of "
    "northern Africa over an area comparable in size to the United States. It is "
    "famous for towering sand dunes, rocky plateaus, and punishing daytime heat "
    "that gives way to cold nights. Despite the harsh climate, hardy plants, "
    "reptiles, and nomadic peoples have lived in and crossed the desert for "
    "centuries, following ancient routes that once carried salt and gold. "
)
C1 = _C1_BASE * _DOC_REPEAT   # 河流（distractor）
C2 = _C2_BASE * _DOC_REPEAT   # 山峰（含答案 Mount Kilimanjaro）
C3 = _C3_BASE * _DOC_REPEAT   # 沙漠（distractor）

# ============================================================
# 多文档扩展（keepalive 多段崩溃调试用）
# ============================================================
# PIC_SYNTH_NUM_DOCS=N (默认 3) 生成 N 段 fixed prompt: 答案段(_C2_BASE
# Kilimanjaro)固定放在中间 index N//2, 其余用互异的 distractor base 填充
# (河流/沙漠/珊瑚礁/尼罗河/太平洋/戈壁/密西西比——均非"非洲最高峰",不与答案
# 竞争)。N=3 时 == 原始 [Amazon, Kilimanjaro, Sahara](零回归)。用于在 fixed
# prompt(可 K-dump)下复现 dataset 的多段 keepalive 崩溃: synthetic 3 段可用,
# 4/5/6/7 段是否崩 → 定位 chunk-count 是否是触发因子。最多 8 段(1 答案+7 distractor)。
_NUM_DOCS = int(os.environ.get("PIC_SYNTH_NUM_DOCS", "3"))
_GBR_BASE = (
    "The Great Barrier Reef lies off the coast of Queensland in Australia and "
    "is the largest coral reef system on the planet. It stretches for over two "
    "thousand kilometers and is made up of thousands of individual reefs and "
    "hundreds of small islands. The reef supports an extraordinary variety of "
    "marine life, including fish, turtles, and mollusks, and can even be seen "
    "from space. Warming seas and pollution threaten its fragile ecosystem. "
)
_NILE_BASE = (
    "The Nile is a major north-flowing river in northeastern Africa and has "
    "long been considered one of the longest rivers on Earth. It runs through "
    "many countries before reaching the Mediterranean Sea and has supported "
    "agriculture along its banks for thousands of years. Ancient civilizations "
    "depended on its predictable floods to grow crops in an otherwise dry "
    "region. Dams built along its course now regulate the seasonal water flow. "
)
_PACIFIC_BASE = (
    "The Pacific Ocean is the largest and deepest of the world's oceans, "
    "covering more area than all the land on Earth combined. It stretches from "
    "the Arctic in the north to the Southern Ocean and is ringed by volcanoes "
    "and earthquake zones known as the Ring of Fire. Countless islands dot its "
    "vast surface, and its waters hold a huge share of the planet's marine "
    "biodiversity. Deep trenches plunge many kilometers below the surface. "
)
_GOBI_BASE = (
    "The Gobi is a large cold desert spanning parts of northern China and "
    "southern Mongolia. Unlike hot sandy deserts it is mostly bare rock and "
    "gravel, and temperatures swing wildly between scorching days and freezing "
    "nights. The region is famous for fossil discoveries, including many "
    "dinosaur eggs found among its rocky basins. Hardy camels and small "
    "communities of herders have adapted to its harsh and windswept plains. "
)
_MISS_BASE = (
    "The Mississippi River is one of the principal rivers of North America and "
    "drains a huge basin stretching across the central United States. It flows "
    "generally southward before emptying into the Gulf of Mexico through a broad "
    "delta. The river has been a vital route for trade and transport since "
    "before the arrival of European settlers. Levees and locks now manage its "
    "flow to protect the many cities that line its winding banks. "
)
_DISTRACTOR_BASES = [
    _C1_BASE, _C3_BASE, _GBR_BASE, _NILE_BASE, _PACIFIC_BASE, _GOBI_BASE, _MISS_BASE,
]


def _build_synth_docs(n):
    """Return (docs, answer_idx): N distinct doc strings, answer at index N//2.

    PIC_SYNTH_VARIED=1 → give each doc a DIFFERENT repeat count (→ varied
    segment token sizes after 64-padding), mimicking real datasets where chunks
    have very different lengths. Used to test whether varied segment sizes (vs
    the uniform ~1152-tok synthetic docs) trigger the keepalive multi-chunk
    collapse (dataset chunks are varied-size; synthetic uniform ones don't
    collapse at N<=6). Answer doc keeps a mid repeat so it stays substantial.
    """
    n = max(1, min(int(n), 1 + len(_DISTRACTOR_BASES)))
    mid = n // 2
    _varied = os.environ.get("PIC_SYNTH_VARIED") == "1"
    # per-doc repeat multipliers when varied (cycled); answer doc uses _DOC_REPEAT.
    _rep_cycle = [3, 10, 5, 16, 7, 13, 9, 20]
    out, di = [], 0
    for i in range(n):
        if i == mid:
            out.append(_C2_BASE * _DOC_REPEAT)     # 含答案 Mount Kilimanjaro
        else:
            _rep = _rep_cycle[di % len(_rep_cycle)] if _varied else _DOC_REPEAT
            out.append(_DISTRACTOR_BASES[di] * _rep)
            di += 1
    return out, mid


DOCS, _ANSWER_IDX = _build_synth_docs(_NUM_DOCS)


# ============================================================
# 非 PIC 模式 prompt（规格文档 §2.3）
# 文档之间无分隔符，直接拼接
# ============================================================
PROMPT = f"{SYS}{C1}{C2}{C3}{Q}"   # 完整测试 prompt
W1     = f"{SYS}{C1}{Q}"           # warmup 1：只含 C1
W3     = f"{SYS}{C3}{Q}"           # warmup 2：只含 C3

# ============================================================
# PIC 模式 prompt（规格文档 §2.4）
# 文档之间用 <<PIC_SEP>> 分隔
# ============================================================
PIC_PROMPT = f"{SYS}{SEP}{C1}{SEP}{C2}{SEP}{C3}{SEP}{Q}"   # 完整测试 prompt
PIC_W1     = f"{SYS}{SEP}{C1}{SEP}{Q}"                      # warmup 1：只含 C1 段
PIC_W3     = f"{SYS}{SEP}{C3}{SEP}{Q}"                      # warmup 2：只含 C3 段


# ============================================================
# 辅助函数：服务器生命周期
# ============================================================

def _wait_port_free(port: int, timeout: float = 30.0) -> None:
    """等待端口释放（规格文档 §6.1）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect(("127.0.0.1", port))
                # 连接成功 → 端口还在使用
            except (ConnectionRefusedError, OSError):
                return  # 连接失败 → 端口已释放
        time.sleep(1.0)
    print(f"  [警告] 等待端口 {port} 释放超时（{timeout}s），继续运行……")


def _wait_server_ready(port: int, proc: subprocess.Popen, timeout: float = READY_TIMEOUT) -> None:
    """轮询 GET /health 直到返回 200 或超时（规格文档 §6.2）。"""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"服务器进程提前退出，returncode={proc.returncode}")
        try:
            resp = requests.get(url, timeout=2.0)
            if resp.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(3.0)
    raise TimeoutError(f"服务器在 {timeout}s 内未就绪（端口 {port}）")


def _shutdown(proc: subprocess.Popen, timeout_sigterm: float = 30.0, timeout_sigkill: float = 10.0) -> None:
    """SIGTERM → 等待 30s → SIGKILL 整个进程组（规格文档 §6.3）。"""
    if proc.poll() is not None:
        # 关闭日志文件（如果存在）
        log_file = getattr(proc, "_log_file", None)
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    else:
        deadline = time.time() + timeout_sigterm
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(1.0)

        if proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

            deadline = time.time() + timeout_sigkill
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                time.sleep(0.5)

    # 关闭日志文件
    log_file = getattr(proc, "_log_file", None)
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass


def _start_server(port: int, model: str, tp: int, extra_args: List[str], mode: Optional[str] = None) -> subprocess.Popen:
    """
    启动 sglang 服务器，使用 GLM-5.2 必要参数。
    公共参数来自 run_prefill_benchmark.sh 的 SGLANG_COMMON_ARGS。
    日志写入 /tmp/sglang_quick_test_<port>[_<mode>].log。加 mode 后缀是为了
    每个模式独立日志，避免后一个模式启动时 open("w") 把前一个崩溃日志清空。
    """
    cmd = [
        SGLANG_PY, "-m", "sglang.launch_server",
        "--model-path", model,
        "--tp", str(tp),
        "--trust-remote-code",
        "--mem-fraction-static", MEM_FRAC,
        "--cuda-graph-max-bs", CUDA_GRAPH_MAX_BS,
        "--reasoning-parser", "glm45",
        "--tool-call-parser", "glm47",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--disable-piecewise-cuda-graph",
        "--log-level", "warning",
        *extra_args,
    ]

    env = os.environ.copy()
    # CUDA 13 兼容库路径（与 run_prefill_benchmark.sh 保持一致）
    cu13_lib = "/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
    ld_path = f"/usr/local/cuda-13.0/compat:{cu13_lib}"
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{ld_path}:{existing}" if existing else ld_path
    # DeepGEMM fast warmup：把每个 GEMM shape 的 warmup m_list 从 65536 降到 ~3073
    # （约 21 倍）。GLM-5.2 (DeepSeek DSA/V3 架构) 有几十个 GEMM shape，每个都要
    # 完整 warmup 一遍 m_list，因此 --chunked-prefill-size=-1 触发的 m_max=65536
    # 会让总编译量 = shape数 × 65536，启动远超 READY_TIMEOUT。
    # 直接赋值（而非 setdefault）以覆盖 shell 中可能存在的空值/旧值。
    env["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = FAST_WARMUP
    # 见文件顶端 FAST_WARMUP_STRIDE_MULT 注释：pic_a3 系列会多引入 shape 变体，
    # 用 stride_mult=4 把 per-shape M 采样从 ~3073 降到 ~1281，缩短启动时间。
    env["SGLANG_JIT_DEEPGEMM_FAST_WARMUP_STRIDE_MULT"] = FAST_WARMUP_STRIDE_MULT
    # A3 调试模式:PIC_A3_DEBUG=1 时,加 CUDA_LAUNCH_BLOCKING=1 让 CUDA 崩溃
    # 时的堆栈是同步的(定位真实失败位置,而非下游 tolist())。
    # 同时开启 A3 clipping 的每层 shape 日志(SGLANG_A3_CLIP_DEBUG=1)。
    if os.environ.get("PIC_A3_DEBUG") == "1":
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        env["TORCH_USE_CUDA_DSA"] = "1"
        env["SGLANG_A3_CLIP_DEBUG"] = "1"

    log_path = (
        f"/tmp/sglang_quick_test_{port}_{mode}.log" if mode
        else f"/tmp/sglang_quick_test_{port}.log"
    )
    print(f"  [启动] {' '.join(cmd[2:6])} ... 端口 {port}")
    print(f"  [日志] {log_path}")
    print(f"  [warmup] SGLANG_JIT_DEEPGEMM_FAST_WARMUP="
          f"{env['SGLANG_JIT_DEEPGEMM_FAST_WARMUP']}"
          f" STRIDE_MULT={env['SGLANG_JIT_DEEPGEMM_FAST_WARMUP_STRIDE_MULT']}"
          f"（生效时日志中 DeepGEMM warmup 进度条 total 应为 ~3073/stride_mult 而非 65536）")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,   # 独立进程组，关闭时 killpg
    )
    # 保存 log_file 引用到 proc 对象，shutdown 时关闭
    proc._log_file = log_file  # type: ignore[attr-defined]
    proc._log_path = log_path  # type: ignore[attr-defined]
    return proc


# ============================================================
# 请求发送（规格文档 §4.1）
# ============================================================

def _post_generate(
    port: int,
    text: str,
    max_new_tokens: int,
    extra_payload: Optional[Dict] = None,
    stop: Optional[List[str]] = None,
) -> Dict:
    """
    向 /generate 发送请求，返回响应 dict（规格文档 §4.1）。

    请求格式：
      POST /generate
      {"text": "<prompt>", "sampling_params": {"temperature": 0, "max_new_tokens": N}}
    temperature=0：贪心解码，结果可复现。

    extra_payload: 额外的顶层字段（合并到 payload 顶层，不进 sampling_params），
                   A3/CacheBlend 用来传 reuse_method、precomputed_kv_path 等。
    stop: 可选的 stop 字符串列表,写入 sampling_params.stop。用于正确性采样
          (max_new_tokens=32) 时强制模型吐完答案就停,避免 greedy 循环
          刷屏(见 Q 定义处的注释)。
    """
    url = f"http://127.0.0.1:{port}/generate"
    sampling: Dict = {
        "temperature": 0,
        "max_new_tokens": max_new_tokens,
    }
    if stop:
        sampling["stop"] = stop
    payload = {
        "text": text,
        "sampling_params": sampling,
    }
    if extra_payload:
        payload.update(extra_payload)
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def first_divergence_token(a: List[int], b: List[int]) -> int:
    """返回 a 和 b 第一个不同的 token 索引，全相同则返回 min(len)（规格文档 §5.2）。"""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _extract_ids32(out: Dict) -> List[int]:
    """从 /generate 响应中提取 output token ids，兼容多种返回格式。"""
    meta = out.get("meta_info", {})
    ids = meta.get("output_token_ids", [])
    if not ids:
        ids = out.get("token_ids", [])
    if not ids:
        ids = out.get("output_ids", [])
    return ids


def _extract_output_text(out: Dict) -> str:
    """从 /generate 响应中提取输出文本(含 CoT reasoning)。

    sglang 的 /generate 返回 {"text": "...", "meta_info": {...}}。
    GLM-5.2 用 `--reasoning-parser glm45` 会把 `<think>...</think>` 之间的内容
    剥到 `meta_info.reasoning_content`,`text` 只留后续输出。

    问题:很多样本模型把**真正的答案**留在 `<think>` 里(常见 pattern:
    `<think>...The answer is X.</think> If you cannot find, say "I don't know"`)。
    只读 `text` 会拿到空串或纯 meta 引导语 → F1=0 的假阴性。

    这里把 reasoning_content 用 `<think>...</think>` 包裹后拼在 text 前面,
    保留原始 CoT 结构,让下游 `_extract_answer_flexible` 能在**整段文本**
    里搜 "Answer: X" 模式(不只是 </think> 之后的部分)。

    Batch 请求会返回 list,这里只取第 0 项。
    """
    if isinstance(out, list):
        out = out[0] if out else {}
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    txt = out.get("text", "") if isinstance(out, dict) else ""
    if not txt:
        # 有些配置下 text 字段可能缺失/为空,退化到 meta_info.output_text
        txt = meta.get("output_text", "") or ""
    reasoning = meta.get("reasoning_content", "") or ""
    if reasoning and "</think>" not in (txt or ""):
        # 重建 <think>...</think>text 结构。extract_final_answer 会切在
        # </think> 上取 post-think,`_extract_answer_flexible` 单独在整段
        # 里 re.findall "Answer: X"(见下面 fix),覆盖 answer-in-think 场景。
        txt = f"<think>{reasoning}</think>{txt}"
    return txt if isinstance(txt, str) else str(txt)


# ============================================================
# 段 64 对齐（路径 A）：把每段文本 padding 到 encode 后 token 数为 64 的倍数
# ============================================================

def _pad_text_to_multiple(text: str, tokenizer, multiple: int = 64, prepend: bool = False):
    """
    在文本追加/前置填充字符，使其 encode(add_special_tokens=False) 后的 token 数
    正好是 multiple 的倍数。返回 (padded_text, orig_len, padded_len)。

    prepend=False（默认）：在末尾追加 padding。
    prepend=True：在开头前置 padding —— 用于 QUERY 段，让真实问题 + "Answer:" 提示
                  落在 prompt 最末尾，避免生成阶段回显 padding 的 `.,!?` 尾巴
                  （这正是 dataset F1 掉分的主因：答案区被 padding 污染）。

    必须与服务器端 split_and_tokenize 使用同一个 tokenizer.encode（且
    add_special_tokens=False），才能保证脚本本地测得的段长与服务器实际分段一致。

    ★ pad 策略: 优先用**多字符轮换**(如 ".,!?" 每次追加一个不同 char),避免
    BPE tokenizer 把 N 个重复 char merge 成一个 dominant token 淹没 padding 段。
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    orig = len(ids)
    if orig % multiple == 0:
        return text, orig, orig
    target = ((orig // multiple) + 1) * multiple

    # 优先级: 多字符轮换 (每次追加序列里下一个 char, 让 padding 区 tokens 分散)
    # → 双字符轮换 → 单字符 (兜底; 单字符必然 merge 成 dominant token)
    # 用不同 tokenizer 时哪个能精确打到 target 不好预测, 挨个试, 命中即返回。
    for pad_seq in [".,!?", ".!,", ".,", " .", ".", " ", "\n", "!", "a", "0"]:
        pad = ""
        cur = orig
        guard = 0
        max_guard = multiple * 8
        idx = 0
        while cur < target and guard < max_guard:
            pad = pad + pad_seq[idx % len(pad_seq)]
            t = (pad + text) if prepend else (text + pad)
            cur = len(tokenizer.encode(t, add_special_tokens=False))
            idx += 1
            guard += 1
        if cur == target:
            return t, orig, cur
    # 兜底：返回最后一次尝试（可能未精确对齐；调用方会打印警告）
    return t, orig, cur


# ============================================================
# PIC 段命中解析（从服务器日志的 [PIC-HIT] 记录）
# ============================================================
# 说明：PIC 段级缓存的命中 **不通过 meta_info.cached_tokens 上报**
# （picache.match_prefix 故意返回空 device_indices 以满足 DSA indexer kernel
#  的 num_tokens==seq_len 要求）。因此 cached_tokens 恒为 0，无法反映 PIC 命中。
# picache.py 在命中时打印 "[PIC-HIT] rid=... hit_segments=x/y hit_tokens=z"，
# 这里通过解析该日志来获得真实的段级命中情况。

_PIC_HIT_RE = re.compile(
    r"\[PIC-HIT\]\s+rid=\S+\s+hit_segments=(\d+)/(\d+)\s+hit_tokens=(\d+)"
)


def _log_offset(proc: subprocess.Popen) -> int:
    """返回服务器日志当前字节大小，用作解析日志增量的起点。"""
    log_path = getattr(proc, "_log_path", None)
    if not log_path:
        return 0
    try:
        return os.path.getsize(log_path)
    except OSError:
        return 0


def _parse_pic_hits(proc: subprocess.Popen, since_offset: int) -> Optional[Dict[str, int]]:
    """
    解析 since_offset 之后服务器日志中的 [PIC-HIT] 记录。

    TP=N 时同一请求每个 rank 各打印一条（hit_tokens 相同），取 hit_tokens
    最大的一条作为该请求的命中结果。
    返回 {"hit_segments", "total_segments", "hit_tokens"}；无命中则返回 None。
    """
    log_path = getattr(proc, "_log_path", None)
    if not log_path:
        return None
    # 给子进程一点时间把日志刷到文件（logger 到 stderr 可能有缓冲）
    time.sleep(0.5)
    try:
        with open(log_path, "r", errors="replace") as f:
            f.seek(since_offset)
            chunk = f.read()
    except OSError:
        return None

    best: Optional[Dict[str, int]] = None
    for m in _PIC_HIT_RE.finditer(chunk):
        hit_seg, total_seg, hit_tok = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if best is None or hit_tok > best["hit_tokens"]:
            best = {
                "hit_segments": hit_seg,
                "total_segments": total_seg,
                "hit_tokens": hit_tok,
            }
    return best


# ============================================================
# 单次模式测试（规格文档 §4 完整流程）
# ============================================================

def run_mode(
    mode: str,
    port: int,
    model: str,
    tp: int,
    prompt: str,
    warmup_prompts: List[str],
    extra_args: List[str],
    extra_payload: Optional[Dict] = None,
) -> Dict:
    """
    执行完整的单次模式测试流程（规格文档 §4）：
      1. 等待端口释放
      2. 启动服务器
      3. 等待服务器就绪（/health，最多 READY_TIMEOUT 秒）
      4. Warmup：发送 2 个 warmup prompt（max_new_tokens=4，temperature=0）
      5. TTFT 测量：发送完整 prompt（max_new_tokens=1），计时
      6. 正确性采样：发送完整 prompt（max_new_tokens=32），保存 output token ids
      7. 关闭服务器（SIGTERM → SIGKILL）
      8. 等待端口释放

    返回：
        {
            "mode": str,
            "ttft": float,          # 单位：秒
            "cached_tokens": int,
            "prompt_tokens": int,
            "hit_rate": float,
            "ids32": List[int],     # 前 32 个 output token id
        }
    """
    print(f"\n{'='*64}")
    print(f"  模式: {mode}")
    print(f"{'='*64}")

    # 步骤 1：确保端口空闲
    print("  [1/8] 等待端口释放……")
    _wait_port_free(port)

    # 步骤 2：启动服务器
    print("  [2/8] 启动服务器……")
    proc = _start_server(port, model, tp, extra_args, mode=mode)

    ttft = 0.0
    cached_tokens = 0
    prompt_tokens = 0
    hit_rate = 0.0
    ids32: List[int] = []
    text32: str = ""
    # PIC 段级命中（从服务器 [PIC-HIT] 日志解析，独立于 cached_tokens）
    pic_hit_segments = 0
    pic_total_segments = 0
    pic_hit_tokens = 0
    pic_hit_rate = 0.0

    try:
        # 步骤 3：等待服务器就绪
        print(f"  [3/8] 等待服务器就绪（最多 {READY_TIMEOUT}s）……")
        _wait_server_ready(port, proc)
        print("  [3/8] 服务器就绪！")

        # 步骤 4：Warmup（规格文档 §4，每个 warmup max_new_tokens=4，temperature=0）
        print(f"  [4/8] Warmup 阶段（{len(warmup_prompts)} 个请求）……")
        for i, wp in enumerate(warmup_prompts, 1):
            print(f"    Warmup {i}/{len(warmup_prompts)} (~{len(wp)} 字符)……")
            _post_generate(port, wp, max_new_tokens=4, extra_payload=extra_payload)
        print("  [4/8] Warmup 完成")

        # 步骤 5：TTFT 测量（规格文档 §4.2）
        # max_new_tokens=1，几乎等于纯 prefill 延迟；使用 perf_counter 高精度计时
        print("  [5/8] 测量 TTFT（max_new_tokens=1）……")
        # 记录发请求前的日志位置，用于解析本次请求产生的 [PIC-HIT] 记录
        _pic_log_offset = _log_offset(proc)
        t0 = time.perf_counter()
        out = _post_generate(port, prompt, max_new_tokens=1, extra_payload=extra_payload)
        ttft = time.perf_counter() - t0

        # 提取标准前缀缓存命中信息（规格文档 §4.3）。
        # 注意：对 PIC 模式，cached_tokens 恒为 0（见 _parse_pic_hits 说明），
        # 真实的段级命中要从服务器 [PIC-HIT] 日志解析。
        meta_info = out.get("meta_info", {})
        cached_tokens = meta_info.get("cached_tokens", 0)
        prompt_tokens = meta_info.get("prompt_tokens", 0)
        hit_rate = cached_tokens / prompt_tokens if prompt_tokens > 0 else 0.0

        # PIC 段级命中（独立于 cached_tokens）
        pic_hit = _parse_pic_hits(proc, _pic_log_offset)
        if pic_hit is not None:
            pic_hit_segments = pic_hit["hit_segments"]
            pic_total_segments = pic_hit["total_segments"]
            pic_hit_tokens = pic_hit["hit_tokens"]
            pic_hit_rate = (
                pic_hit_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
            )

        print(f"  [5/8] TTFT={ttft:.3f}s  "
              f"cached_tokens(标准前缀缓存)={cached_tokens}/{prompt_tokens}  "
              f"命中率={hit_rate:.1%}")
        if pic_hit is not None:
            print(f"        PIC 段命中: {pic_hit_segments}/{pic_total_segments} 段  "
                  f"命中 tokens={pic_hit_tokens}  "
                  f"PIC 命中率={pic_hit_rate:.1%}")
        else:
            print(f"        PIC 段命中: 无（日志未见 [PIC-HIT]，"
                  f"若为 PIC 模式说明段缓存未命中）")

        # 步骤 6：正确性采样（max_new_tokens=32，temperature=0）
        # 加 stop 截掉 Q 段 64 对齐 padding(".,!?" 轮换)被 greedy 回显的尾巴:
        # 一词答案(如 "dog.")后面撞到 ",!?" 立即停,输出干净可读。
        # 仍保留 max_new_tokens=32: 真崩坏(从头乱码)时照样能看出来——乱码里也会
        # 很快撞到这些标点; FDT 判定看第 0 位分歧,不受 stop 影响。
        _clean_stop = ["\n", ",", "!", "?", "，", "！", "？"]
        print("  [6/8] 正确性采样（max_new_tokens=32，stop 截 padding 尾巴）……")
        out32 = _post_generate(
            port, prompt, max_new_tokens=32,
            extra_payload=extra_payload,
            stop=_clean_stop,
        )
        ids32 = _extract_ids32(out32)
        # 除了 token id 也把解码后的完整文本（含 <think> reasoning）留一份,
        # 汇总时打出来。ids32 只能看前 10 个整数,肉眼看不出模型到底吐了什么;
        # 保留文本能一眼分辨"崩到重复"还是"逻辑跑偏"还是"看起来正常"。
        text32 = _extract_output_text(out32)
        print(f"  [6/8] 采样完成，获得 {len(ids32)} 个 output token id"
              f"（文本长度 {len(text32)} char）")

    except Exception as _exc:
        # 打印服务器日志尾部，帮助定位崩溃原因
        log_path = getattr(proc, "_log_path", None)
        if log_path:
            try:
                # flush log file before reading
                log_file = getattr(proc, "_log_file", None)
                if log_file:
                    log_file.flush()
                with open(log_path, "r", errors="replace") as _f:
                    lines = _f.readlines()
                tail = lines[-80:] if len(lines) > 80 else lines
                print(f"\n  ══ 服务器日志尾部（{log_path}，最后 {len(tail)} 行）══")
                print("".join(tail))
                print("  ══ 日志结束 ══\n")
            except Exception as _le:
                print(f"  [警告] 无法读取服务器日志: {_le}")
        raise  # 重新抛出原始异常
    finally:
        # 步骤 7：关闭服务器
        print("  [7/8] 关闭服务器（SIGTERM→SIGKILL）……")
        _shutdown(proc)
        print("  [7/8] 服务器已关闭")

    # 步骤 8：等待端口释放
    print("  [8/8] 等待端口释放……")
    _wait_port_free(port)

    return {
        "mode": mode,
        "ttft": ttft,
        "cached_tokens": cached_tokens,
        "prompt_tokens": prompt_tokens,
        "hit_rate": hit_rate,
        "ids32": ids32,
        "text32": text32,
        "pic_hit_segments": pic_hit_segments,
        "pic_total_segments": pic_total_segments,
        "pic_hit_tokens": pic_hit_tokens,
        "pic_hit_rate": pic_hit_rate,
    }


# ============================================================
# 数据集加载 + F1 打分（pic_bench_lite Phase 1）
# ============================================================
# 现有 run_mode 只跑一个 hardcoded prompt,输出 FDT(首个分歧 token)——粗到无法
# 反映 pic_a3 / pic_cacheblend 修复对真实任务准确度的影响。这里引入
# pic_bench_lite:载入真实数据集(hotpotqa / longbench_*),按每个 mode 跑 N 个
# 样本,用 F1(HotpotQA/SQuAD 风格)+ ROUGE-L(可选)打分。
#
# 设计权衡:
# 1) 每个 mode 只启动一次服务器,内层循环 N 个样本 → 避免每样本重启开销。
# 2) 每个样本前做 prime_sample(warmup),让 PIC 段级缓存/Radix 前缀缓存
#    在 measure 请求前就已注入,与 hardcoded 场景的 W1/W3 warmup 语义对齐。
# 3) 每个样本先测 TTFT(max_new_tokens=1),再测正式输出(max_new_tokens=N)。
# 4) 输出文本用 extract_answer_only 剥离 <think> 后再打分,避免 CoT reasoning
#    文本污染 F1。
# 5) PIC-HIT 日志按样本级切片解析(每个样本前记录日志偏移量)。


def _make_send_fn(
    port: int, extra_payload: Optional[Dict] = None
) -> Callable[[str, int], bool]:
    """把 _post_generate 包装成 pic_bench_lite.warmup 需要的 send_fn 签名。"""

    def _send(prompt: str, max_new_tokens: int) -> bool:
        try:
            _post_generate(
                port, prompt, max_new_tokens=max_new_tokens, extra_payload=extra_payload
            )
            return True
        except Exception:
            return False

    return _send


def _pad_chunk_if_needed(text: str, tokenizer, pad_to_64: bool, prepend: bool = False) -> str:
    """对每段(chunk/sys/query)按需 pad 到 64 的倍数(PIC + DSA 场景必需)。
    prepend=True 用于 QUERY 段：padding 前置，真实问题落在段尾，避免答案区被污染。"""
    if not pad_to_64 or tokenizer is None:
        return text
    padded, _o, _p = _pad_text_to_multiple(text, tokenizer, PIC_ALIGN, prepend=prepend)
    return padded


def _pct(vals: List[float], q: float) -> float:
    """简易百分位:vals 非空时取排序后 q(0..1) 分位;空时返回 0.0。"""
    if not vals:
        return 0.0
    xs = sorted(vals)
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    idx = int(round(q * (len(xs) - 1)))
    return xs[idx]


def run_mode_dataset(
    mode: str,
    port: int,
    model: str,
    tp: int,
    samples: List,
    extra_args: List[str],
    separator: str,
    extra_payload: Optional[Dict] = None,
    query_suffix: Optional[str] = None,
    max_new_tokens: int = 256,
    pad_to_64: bool = False,
    do_warmup: bool = True,
    ttft_probe: bool = True,
    seed: int = 0,
    answer_extractor: str = "flexible",
) -> Dict:
    """对 N 个真实样本跑一次 mode 测试,返回聚合指标(mean F1、mean TTFT 等)。

    与 run_mode 的关系:共用 _start_server / _shutdown / _post_generate /
    _parse_pic_hits;差异在于内层循环 N 个样本,并按样本 extract_answer_only
    + f1_score 打分。

    参数:
      samples          : list[ProcessedSample],pic_bench_lite.datasets 返回的对象
      separator        : PIC 模式用 SEP,非 PIC 模式用 ""
      query_suffix     : 追加到 query 后的引导语,例如 "Answer with only the answer"
      max_new_tokens   : measure 请求生成上限;CoT 输出常要 256+ token
      pad_to_64        : True 时每段(sys/chunk/query)pad 到 64 的倍数
                         (DSA + PIC 必需,与 hardcoded 路径一致)
      do_warmup        : True 时每样本先 prime_sample(chunks) 打命中
      ttft_probe       : True 时每样本额外发一次 max_new_tokens=1 请求测 TTFT
    """
    # 惰性导入:非 dataset 模式不需要 pic_bench_lite / transformers 依赖
    from sglang.test.pic_bench_lite.prompt import build_prompt
    from sglang.test.pic_bench_lite.scorers.extract import (
        extract_answer_only,
        extract_final_answer,
        trim_to_short_answer,
    )
    from sglang.test.pic_bench_lite.scorers.f1 import f1_score
    from sglang.test.pic_bench_lite.warmup import prime_sample

    import random as _random

    tokenizer = None
    warmup_query_padded: Optional[str] = None
    if pad_to_64:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        # ★ warmup query 也必须 pad 到 64,否则 pic_alloc_for_extend 会因为
        # 尾段(warmup 段)长度不是 64 倍数而 crash:
        #   RuntimeError: The expanded size of the tensor (73) must match
        #   the existing size (64) at non-singleton dimension 0.
        # 见 python/sglang/srt/pic/pic_alloc.py:334。
        warmup_query_padded = _pad_chunk_if_needed("warmup", tokenizer, True)

    def _extract_answer_flexible(text: str, gt) -> str:
        """CoT-aware answer extractor.

        GLM-5.2 + sglang `--reasoning-parser glm45` 会把 <think>...</think> 剥到
        `meta_info.reasoning_content`,`text` 只留后续输出。`_extract_output_text`
        已经把 reasoning 用 `<think>...</think>` 重新拼回来,所以这里的 `text`
        可能包含完整的 <think>reasoning</think>tail 结构。常见 4 种 pattern:

          A) `<引导语> Answer: <真答案>`               (同一行末尾)
          B) `<引导语>\\n\\n<真答案>`                  (空行后另起一段)
          C) 只有 `<引导语>`                            (max_new_tokens 太小截断/EOS)
          D) 答案在 <think> 里,</think> 之后只剩 meta  (答案在 reasoning 里)

        抽取顺序:
        0) **在整段 text(含 reasoning)里搜 `Answer: X`,取最后一个非占位符**
           —— 处理 D:答案在 <think>...The answer is X.</think> 里
        1) 按 `\\n\\n` 切 post-think body,取最后一非空块(处理 B)
        2) 在候选块里 `re.findall("Answer:")` 取最后一个非占位符 val(处理 A)
        3) 若无 "Answer:" 标签,块内按行取第一非引导语行
        4) 全被过滤则退化到 extract_answer_only(处理 C —— 至少给个东西打分)
        """
        # 0) 整段文本(含 reasoning)里搜 "Answer: X" —— 答案在 <think> 里时救回来。
        #    只匹配简短 val(<80 char)避免抓到 "The answer to this is a long
        #    explanation..." 之类的长解释。GLM 一般用 "Answer: X" 或 "答案: X"
        #    做终结标记,其他推理句子不会用这个格式。
        _all_matches = re.findall(
            r'(?i)(?:answer|答案)\s*[::]\s*([^\n)]+)', text or "",
        )
        _all_valid = [
            m.strip() for m in _all_matches
            if m.strip()
            and not m.strip().startswith("<")
            and len(m.strip()) < 80
        ]
        if _all_valid:
            return _all_valid[-1]

        body, _had_close = extract_final_answer(text)
        if not body:
            return ""

        # 1) `\n\n` 分块,取最后一块。GLM 常用 "<引导语>\n\n<真答案>" 布局。
        _blocks = [b.strip() for b in body.split("\n\n") if b.strip()]
        candidate = _blocks[-1] if _blocks else body

        # 2) 在候选块里找所有 "Answer: <val>",取最后一个非占位符
        matches = re.findall(r'(?i)(?:answer|答案)\s*[::]\s*([^\n)]+)', candidate)
        valid = [m.strip() for m in matches
                 if m.strip() and not m.strip().startswith("<")]
        if valid:
            return valid[-1]

        # 3) 无 "Answer:" 标签 —— 按行找第一非引导语行
        _meta_kws = (
            "use the format", "please format", "if the answer",
            "answer with only", "no explanation", "respond with",
            "if you cannot", "i don't know", "i cannot",
            "provide your final", "final answer",
        )
        for line in candidate.splitlines():
            line = line.strip()
            if not line:
                continue
            _low = line.lower()
            _is_bracket_meta = line.startswith("(") and line.endswith(")")
            _has_meta_kw = any(kw in _low for kw in _meta_kws)
            if _is_bracket_meta or _has_meta_kw:
                continue
            return line

        # 4) 全被过滤 → 回退第一非空行(让 caller 知道模型只吐了引导语)
        ans, _ = extract_answer_only(text)
        return ans

    print(f"\n{'=' * 64}")
    print(f"  模式: {mode}  (dataset 模式, N={len(samples)} 样本)")
    print(f"{'=' * 64}")

    print("  [1/8] 等待端口释放……")
    _wait_port_free(port)

    print("  [2/8] 启动服务器……")
    proc = _start_server(port, model, tp, extra_args, mode=mode)

    per_sample_results: List[Dict] = []

    try:
        print(f"  [3/8] 等待服务器就绪(最多 {READY_TIMEOUT}s)……")
        _wait_server_ready(port, proc)
        print("  [3/8] 服务器就绪!")

        send_fn = _make_send_fn(port, extra_payload=extra_payload)

        # SYS 段(所有样本相同)—— 提到循环外,pre-warm 与逐样本 measure 共用。
        _sys_padded = _pad_chunk_if_needed(
            "Read the following documents and answer the question that follows.",
            tokenizer, pad_to_64,
        )

        # pic_a3_oracle / PIC 家族:先单独 pre-warm SYS(sys<sep>warmup),把 SYS 段
        # 缓存上。这样后续逐段 warmup(sys<sep>chunk<sep>warmup)的第一段就已经是
        # "SYS 命中 + chunk miss" → 触发 A³ keepalive/oracle 钩子 → 捕获该 chunk 的
        # 隔离 hidden。否则第一个 chunk 的 warmup 全冷(all-miss)→ 钩子不触发 → 该
        # chunk 永远抓不到 oracle capture(与合成路径的 pic_sys_prewarm 同理)。
        # 仅 PIC 家族(separator 非空)需要;非 PIC 模式此预热无害但也无用,跳过。
        if do_warmup and separator:
            _sysp = build_prompt(
                [], warmup_query_padded or "warmup",
                separator=separator, system_prompt=_sys_padded,
            )
            _ok = send_fn(_sysp, 1)
            print(f"  [prewarm] SYS 段预缓存 {'✓' if _ok else '✗(失败)'} "
                  f"(oracle keepalive 首段钩子触发用)")

        # pic_bench 一致的 distractor 池:每个样本的 measure prompt 会被其他样本
        # 的 chunk 填充(spec.meta["_pool_chunks"])。这些 distractor 必须在
        # measure 前已预热;本循环是顺序执行的,所以在这里先把"每个样本自己的
        # chunk"全部 prime 一遍(覆盖所有可能的 distractor)。
        _pool_active = any(
            "_pool_chunks" in getattr(s, "meta", {}) for s in samples
        )
        _prewarmed = False
        if do_warmup and _pool_active:
            print(f"  [prewarm] pool 已启用,预热 {len(samples)} 个样本的自有 chunk …")
            _pf = _pn = 0
            for s in samples:
                _own = [_pad_chunk_if_needed(c, tokenizer, pad_to_64) for c in s.chunks]
                if not _own:
                    continue
                _wr = prime_sample(
                    send_fn, _own, separator=separator,
                    system_prompt=_sys_padded, warmup_query=warmup_query_padded,
                )
                _pf += _wr.n_failed
                _pn += _wr.n_requests
            print(f"  [prewarm] 完成 (请求 {_pn}, 失败 {_pf})")
            _prewarmed = True

        for i, spec in enumerate(samples, 1):
            sid = getattr(spec, "sample_id", str(i))
            print(f"\n  ── 样本 {i}/{len(samples)}  id={sid} ──")

            # 段对齐(可选)—— query pad 前先拼 suffix(否则末段非 64 倍数 →
            # pic_alloc_for_extend 崩)。sys_padded 已提到循环外(_sys_padded)。
            sys_padded = _sys_padded
            # 自有 chunk(用于非 pool 路径的逐样本 warmup)。
            chunks_padded = [_pad_chunk_if_needed(c, tokenizer, pad_to_64)
                             for c in spec.chunks]
            # pic_bench 一致:measure prompt = pool distractor + 自有 chunk,各自
            # pad(确定性 → 与 prime 时同一 padded 文本 → 段 hash 命中),再按
            # 每样本确定性种子 shuffle(sys/query 仍在两端)。
            _pool_chunks = (
                spec.meta.get("_pool_chunks", [])
                if isinstance(getattr(spec, "meta", None), dict) else []
            )
            all_chunks_padded = [
                _pad_chunk_if_needed(c, tokenizer, pad_to_64)
                for c in (list(_pool_chunks) + list(spec.chunks))
            ]
            # request_id = f"{ds}-{sample_id}" 与 pic_bench 的 spec.request_id
            # (qa_f1.py:58)对齐 → shuffle/pool 的 RNG 抽样与 pic_bench 一致。
            _rid = (
                f'{spec.meta.get("dataset", "")}-{sid}'
                if isinstance(getattr(spec, "meta", None), dict) else sid
            )
            _random.Random(f"{seed}:{_rid}:chunk_order").shuffle(all_chunks_padded)
            _q_with_suffix = spec.query
            if query_suffix:
                _q_with_suffix = spec.query + "\n\n" + query_suffix
            # Query padding is PREPEND: the 64-align pad chars (".,!?") go to the
            # FRONT of the query segment so the prompt still ends with the real
            # question + the "Answer:" cue. Appending padding made the model
            # parrot ".,!?" after the answer (and sometimes emit only "?." with
            # NO answer). The query is the fresh (recomputed) segment, so moving
            # the pad to the front does not affect the cached document-chunk hits.
            query_padded = _pad_chunk_if_needed(
                _q_with_suffix, tokenizer, pad_to_64, prepend=True
            )

            prompt = build_prompt(
                all_chunks_padded,
                query_padded,
                separator=separator,
                query_suffix=None,   # ← suffix 已并入 query_padded,不再重复
                system_prompt=sys_padded,
            )
            _n_pool_chunks = len(_pool_chunks)

            # 逐样本 warmup:仅在未做 pool 预热时(pool 路径已在循环外全量预热)。
            if do_warmup and not _prewarmed and chunks_padded:
                print(f"    [warmup] prime_sample: {len(chunks_padded)} chunks, "
                      f"separator={'<SEP>' if separator else '<empty>'}")
                # warmup 与 measure 用同一 system_prompt+sep 布局;PIC + DSA
                # (page_size=64)要求 warmup query 也 pad 到 64。
                _wr = prime_sample(
                    send_fn, chunks_padded, separator=separator,
                    system_prompt=sys_padded,
                    warmup_query=warmup_query_padded,
                )
                if _wr.n_failed:
                    print(f"    [warmup] 警告: {_wr.n_failed}/{_wr.n_requests} 失败")

            ttft_val: Optional[float] = None
            cached_tokens = 0
            prompt_tokens = 0
            pic_hit_segments = 0
            pic_total_segments = 0
            pic_hit_tokens = 0

            log_offset = _log_offset(proc)

            if ttft_probe:
                t0 = time.perf_counter()
                out_ttft = _post_generate(
                    port, prompt, max_new_tokens=1, extra_payload=extra_payload
                )
                ttft_val = time.perf_counter() - t0
                meta_info = out_ttft.get("meta_info", {})
                cached_tokens = meta_info.get("cached_tokens", 0)
                prompt_tokens = meta_info.get("prompt_tokens", 0)

                pic_hit = _parse_pic_hits(proc, log_offset)
                if pic_hit is not None:
                    pic_hit_segments = pic_hit["hit_segments"]
                    pic_total_segments = pic_hit["total_segments"]
                    pic_hit_tokens = pic_hit["hit_tokens"]

            # 正式采样:max_new_tokens=max_new_tokens
            # stop=["\n"]:query 以 "Answer:" 收尾,raw-completion 模式下模型补出
            # 答案后会继续换行乱写(伪造新 Q&A / 解释)。在第一个换行处截断得到
            # 干净的单行答案,extractor 直接可用。短答 QA(hotpotqa/qasper/
            # narrativeqa)答案都是单行;若跑摘要类(gov_report 需多行)请去掉此 stop。
            out_full = _post_generate(
                port, prompt, max_new_tokens=max_new_tokens,
                extra_payload=extra_payload,
                stop=["\n"],
            )
            output_text = _extract_output_text(out_full)
            # 答案抽取:
            #   pic_bench(默认,对齐): extract_answer_only —— 取 </think> 后第一
            #     非空行(与 pic_bench runner.py:176 一致)。
            #   flexible: 本仓库增强版,优先抓 "Answer:" 后内容、跳过 GLM-5.2 常见的
            #     "(Use the format: ...)" 引导语(extract_answer_only 会把引导语当第一
            #     行抓走 → 抓不到答案)。更鲁棒但与 pic_bench 不一致。
            if answer_extractor == "flexible":
                answer = _extract_answer_flexible(output_text, spec.ground_truth)
                # 收尾巴:raw-completion 模型答完常在同一行继续解释
                # (stop=["\n"] 切不掉),取首句 + 去尾部括注。仅 flexible 路径做,
                # pic_bench 路径保持与参考实现字节对齐。
                answer = trim_to_short_answer(answer)
            else:
                answer, _ = extract_answer_only(output_text)

            gt = spec.ground_truth
            f1 = f1_score(answer, gt) if gt is not None else 0.0

            hit_rate = cached_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
            pic_hit_rate = (
                pic_hit_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
            )

            per_sample_results.append({
                "sample_id": sid,
                "ttft": ttft_val,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "hit_rate": hit_rate,
                "pic_hit_segments": pic_hit_segments,
                "pic_total_segments": pic_total_segments,
                "pic_hit_tokens": pic_hit_tokens,
                "pic_hit_rate": pic_hit_rate,
                "answer": answer,
                "ground_truth": gt,
                "f1": f1,
                "n_pool_chunks": _n_pool_chunks,
            })

            gt_str = gt if isinstance(gt, str) else (
                gt[0] if isinstance(gt, list) and gt else "")
            gt_preview = (gt_str[:60] + "…") if len(gt_str) > 60 else gt_str
            ans_preview = (answer[:120] + "…") if len(answer) > 120 else answer
            # 输出原始 text 头,方便日后诊断 extract 是否抓对了
            raw_head = output_text.replace("\n", "\\n")[:180]
            print(f"    F1={f1:.3f}  TTFT={ttft_val or 0:.3f}s  "
                  f"hit={hit_rate:.1%}  pic={pic_hit_segments}/{pic_total_segments} "
                  f"({pic_hit_rate:.1%})  ptok={prompt_tokens} pool={_n_pool_chunks}")
            print(f"    GT   = {gt_preview!r}")
            print(f"    Pred = {ans_preview!r}")
            print(f"    Raw  = {raw_head!r}")

    except Exception as _exc:
        log_path = getattr(proc, "_log_path", None)
        if log_path:
            try:
                log_file = getattr(proc, "_log_file", None)
                if log_file:
                    log_file.flush()
                with open(log_path, "r", errors="replace") as _f:
                    lines = _f.readlines()
                tail = lines[-80:] if len(lines) > 80 else lines
                print(f"\n  ══ 服务器日志尾部({log_path},最后 {len(tail)} 行)══")
                print("".join(tail))
                print("  ══ 日志结束 ══\n")
            except Exception as _le:
                print(f"  [警告] 无法读取服务器日志: {_le}")
        raise
    finally:
        print("  [7/8] 关闭服务器(SIGTERM→SIGKILL)……")
        _shutdown(proc)
        print("  [7/8] 服务器已关闭")

    print("  [8/8] 等待端口释放……")
    _wait_port_free(port)

    # 聚合
    f1s = [r["f1"] for r in per_sample_results]
    ttfts = [r["ttft"] for r in per_sample_results if r["ttft"] is not None]
    hits = [r["hit_rate"] for r in per_sample_results]
    pic_hits = [r["pic_hit_rate"] for r in per_sample_results]

    return {
        "mode": mode,
        "n_samples": len(per_sample_results),
        "f1_mean": statistics.mean(f1s) if f1s else 0.0,
        "f1_median": statistics.median(f1s) if f1s else 0.0,
        "ttft_mean": statistics.mean(ttfts) if ttfts else 0.0,
        "ttft_p50": _pct(ttfts, 0.5),
        "ttft_p95": _pct(ttfts, 0.95),
        "hit_rate_mean": statistics.mean(hits) if hits else 0.0,
        "pic_hit_rate_mean": statistics.mean(pic_hits) if pic_hits else 0.0,
        "per_sample": per_sample_results,
    }


def _load_dataset_samples(
    dataset_name: str,
    raw_dir: Path,
    cache_path: Path,
    n_samples: int,
) -> List:
    """载入前 N 个 ProcessedSample。用 load_or_build_processed 落盘 cache。

    缓存陷阱:load_or_build_processed 一旦落盘就"缓存里有多少给多少",不管调用
    方要多少。如果本次要的比缓存里存的多,自动 unlink 重建(否则要用户手动 rm
    才能扩容,很容易漏掉→静默用少样本跑测)。
    """
    from sglang.test.pic_bench_lite.datasets import (
        get_dataset,
        load_or_build_processed,
    )

    ds = get_dataset(dataset_name)

    def _pull(iterator, target):
        _out: List = []
        for _s in iterator:
            _out.append(_s)
            if len(_out) >= target:
                break
        return _out

    it = load_or_build_processed(
        cache_path, lambda: ds.preprocess_for_pic(raw_dir),
    )
    samples: List = _pull(it, n_samples)

    # 缓存不够:删掉重建。builder 是全量迭代器,rebuild 时 break 在 n_samples
    # 会把 n_samples 个写进新缓存。
    if len(samples) < n_samples and cache_path.exists():
        cached_n = len(samples)
        print(
            f"  [dataset] 缓存 {cache_path} 只有 {cached_n} 个样本 < 请求的 "
            f"{n_samples},删除并重建……"
        )
        cache_path.unlink()
        it = load_or_build_processed(
            cache_path, lambda: ds.preprocess_for_pic(raw_dir),
        )
        samples = _pull(it, n_samples)

    if not samples:
        raise RuntimeError(
            f"未从 {raw_dir} / cache {cache_path} 载入任何样本(dataset={dataset_name})"
        )
    if len(samples) < n_samples:
        print(
            f"  [dataset] 警告:数据集只提供了 {len(samples)} 个样本 < 请求的 "
            f"{n_samples}(源已耗尽,不是缓存问题)"
        )
    return samples


# ============================================================
# PNG 汇总图(--plot-output 指定时,hardcoded 模式结束后调用)
# ============================================================

def _render_summary_plot(
    results: Dict[str, Dict],
    modes: List[str],
    output_path: str,
    baseline_mode: str = "full_recompute",
) -> None:
    """把 hardcoded 模式的 TTFT / 命中率 / 段命中 / 加速比画成一张 2 面板 PNG。

    Panel A: TTFT bar chart,每根 bar 顶上标注相对 baseline 的加速比。
    Panel B: PIC 命中率 bar chart,PIC 系列 bar 顶上标注 "hit/total 段"。

    出错(matplotlib 不可用 / 无数据)时打印警告但不抛,避免污染主流程结果。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无 GUI 环境
        import matplotlib.pyplot as plt
    except ImportError as _e:
        print(f"  [plot] matplotlib 不可用,跳过 PNG 渲染: {_e}")
        return

    valid = [m for m in modes if m in results and "error" not in results[m]]
    if not valid:
        print("  [plot] 无可用 mode 结果,跳过渲染")
        return

    ttfts = [results[m].get("ttft", 0.0) for m in valid]
    hit_rates = [
        (results[m].get("pic_hit_rate", 0.0)
         if m in ("pic", "pic_a3", "pic_cacheblend", "pic_a3_oracle")
         else results[m].get("hit_rate", 0.0))
        for m in valid
    ]
    seg_labels = [
        (f"{results[m].get('pic_hit_segments', 0)}/"
         f"{results[m].get('pic_total_segments', 0)}"
         if m in ("pic", "pic_a3", "pic_cacheblend", "pic_a3_oracle") else "-")
        for m in valid
    ]
    prompt_tokens = [results[m].get("prompt_tokens", 0) for m in valid]

    baseline_ttft = results.get(baseline_mode, {}).get("ttft") or 0.0
    speedup_labels = [
        (f"{baseline_ttft / t:.2f}×" if (t and baseline_ttft) else "-")
        for t in ttfts
    ]

    _PALETTE = {
        "full_recompute": "#888888",
        "prefix_cache":   "#4C9AFF",
        "pic":            "#36B37E",
        "pic_a3":         "#FF8B00",
        "pic_cacheblend": "#DE350B",
        "a3":             "#6554C0",
        "cacheblend":     "#00B8D9",
        "pic_a3_oracle":  "#8777D9",
    }
    colors = [_PALETTE.get(m, "#666") for m in valid]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: TTFT
    x = list(range(len(valid)))
    barsA = axA.bar(x, ttfts, color=colors, edgecolor="black", linewidth=0.5)
    axA.set_xticks(x)
    axA.set_xticklabels(valid, rotation=15, ha="right")
    axA.set_ylabel("TTFT (seconds)")
    axA.set_title(f"TTFT per mode  (baseline={baseline_mode})")
    axA.grid(axis="y", linestyle="--", alpha=0.4)
    _ymaxA = max(ttfts) if ttfts else 1.0
    axA.set_ylim(0, _ymaxA * 1.25)
    for bar, t, spd in zip(barsA, ttfts, speedup_labels):
        axA.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + _ymaxA * 0.02,
            f"{t:.3f}s\n{spd}",
            ha="center", va="bottom", fontsize=9,
        )

    # Panel B: hit rate + segments
    hit_pct = [h * 100.0 for h in hit_rates]
    barsB = axB.bar(x, hit_pct, color=colors, edgecolor="black", linewidth=0.5)
    axB.set_xticks(x)
    axB.set_xticklabels(valid, rotation=15, ha="right")
    axB.set_ylabel("Hit rate (%)")
    axB.set_title("Cache hit rate & segments  (PIC family: pic_hit_rate)")
    axB.grid(axis="y", linestyle="--", alpha=0.4)
    axB.set_ylim(0, max(100.0, (max(hit_pct) if hit_pct else 0) * 1.25))
    for bar, pct, seg, ptok in zip(barsB, hit_pct, seg_labels, prompt_tokens):
        axB.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{pct:.1f}%\nseg {seg}\n{ptok} tok",
            ha="center", va="bottom", fontsize=9,
        )

    fig.suptitle(
        f"PIC benchmark  |  modes={','.join(valid)}",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] 已写入 PNG: {os.path.abspath(output_path)}")


# ============================================================
# pic_a3_oracle：全 in-process（方案①）
# ============================================================
# 不再需要单独的 Phase 1 capture server / 文件流程。捕获与注入都在测试 server
# 内部完成：warmup(pic_w1/w2/w3) 时首次算到某文档段就按 PIC 段 hash 存下它各
# oracle 层的隔离输入 hidden；measure 时该段命中就把 hidden 注入回去，让该层重
# 选 imp 并从准确(未漂移)输入重算 imp KV。实现见:
#   model_runner._pic_a3_oracle_capture_or_inject  （捕获/注入）
#   deepseek_v2 主循环 check 层边界的调用           （注入点）
#   env SGLANG_PIC_A3_ORACLE + SGLANG_PIC_A3_KEEP_ALIVE（开关）


# ============================================================
# 主测试入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIC Online Quick Test — GLM-5.2，按 quick_test_online_spec.md 规格"
    )
    parser.add_argument("--port", type=int, default=PORT, help="服务器端口")
    parser.add_argument("--model", type=str, default=MODEL, help="模型路径")
    parser.add_argument("--tp", type=int, default=8, help="张量并行数（默认 8）")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["full_recompute", "prefix_cache", "pic"],
        choices=["full_recompute", "prefix_cache", "pic", "a3", "cacheblend",
                 "pic_a3", "pic_cacheblend", "pic_a3_oracle"],
        help="要测试的模式列表",
    )
    parser.add_argument(
        "--a3-precomputed-kv",
        type=str,
        default="/tmp/precomputed_kv/rank{rank}.pt",
        help="A3/CacheBlend 用的预计算 KV .pt 路径模板。使用 {rank} 占位符会在"
             "每个 TP rank 展开为该 rank 的分片文件 (由 scripts/precompute_kv_mla.py 生成)",
    )
    parser.add_argument(
        "--a3-recomp-ratio",
        type=float,
        default=0.15,
        help="A3/CacheBlend 的 recomp_ratio (context 重算比例, 建议 0.05~0.30)",
    )
    parser.add_argument(
        "--a3-check-layers",
        type=int,
        nargs="+",
        default=None,
        help="A³ 多层重选研究探针：在这些层(re)选 imp token。省略或单值(如 1)=当前"
             "单层 pic_a3(零回归)；多值(如 1 10 20 30 40 50 60)开启多层重选。"
             "透传给 server 的 --a3-check-layers。Phase B(keep-all-alive 动态增删)"
             "需再设 env SGLANG_PIC_A3_KEEP_ALIVE=1，否则走 Phase A(单层应用+探针日志)。",
    )
    parser.add_argument(
        "--oracle-layers",
        type=int,
        nargs="+",
        default=[1, 20, 40, 60],
        help="pic_a3_oracle 模式:在这些层做窗口化重选(Phase B keepalive),并用"
             "in-process oracle 的隔离 hidden 在这些层重选 imp + 重算 imp KV。"
             "每次运行可自由指定(如 --oracle-layers 1 30 60 或 1 10 20 30 40 50 60)。"
             "透传给 server 的 --a3-check-layers;总会强制含 layer 1。默认 1 20 40 60。",
    )
    parser.add_argument(
        "--oracle-warmup-docs",
        type=int,
        default=0,
        help="pic_a3_oracle 诊断:warmup 只覆盖前 K 个文档段(C1..CK)。0(默认)=覆盖"
             "全部 N 段(N 由 PIC_SYNTH_NUM_DOCS 决定,N>3 时必须全覆盖否则未 warmup "
             "段命中却无 oracle capture → reselect_full BAIL → 退回全长 keepalive)。"
             "K>=1 时截断为前 K 段(测 isolated-inject 从第几段开始崩,答案在中间段)。"
             "pic_sys_prewarm 始终包含以触发 keepalive 钩子。measure 恒为 SYS+全 N 段+Q。",
    )
    parser.add_argument(
        "--pic-sink-len",
        type=int,
        default=0,
        help="给每个 PIC 段前置 N 个 sink token（吸收段隔离 attention sink，"
             "让真实内容首部 K/V 不被污染）。默认 0=关，透传给 server 的 "
             "--pic-sink-len。★ 本脚本 page_size=64 + PIC_PAD_TO_64 下 N 必须是 "
             "64 的倍数（如 64/128），否则每段变 64k+N 破坏 page 对齐会崩。",
    )
    parser.add_argument(
        "--pic-sink-token-id",
        type=int,
        default=None,
        help="PIC sink 前缀用的 token id。默认 None：server 端自动取 "
             "tokenizer.pad_token_id → bos_token_id → 0。",
    )
    parser.add_argument(
        "--enable-clip",
        action="store_true",
        default=False,
        help="[DEPRECATED / NO-OP] 历史 flag: 给 server 加 --enable-pic-a3-clip。"
             "现在 pic_a3 / pic_cacheblend 默认就是 ragkv-style clip (layer 1 "
             "Q·K 挑 + 收窄到 miss+imp), server_args 已移除 --enable-pic-a3-clip, "
             "传了会崩。此 flag 保留仅为兼容旧脚本, 实际不再传参给 server。",
    )
    parser.add_argument(
        "--gen-prompt-file",
        type=str,
        default=None,
        metavar="PATH",
        help="导出当前测试用的完整 PROMPT 到指定文件后立即退出,不跑测试。"
             "供 scripts/precompute_kv_mla.py --prompt-file 直接使用,保证"
             "precompute 的字符串与本脚本 a3/cacheblend 模式实际发送的完全一致"
             "(包括 --pic-pad-to-64 的 token 对齐 padding)。",
    )
    # ── pic_bench_lite dataset 模式(Phase 1)────────────────────────────
    # --dataset 未指定时走 legacy hardcoded 3-doc + FDT 判定;指定时切换到
    # ProcessedSample 循环 + F1 打分。两条路径共享 mode_configs 里的
    # extra_args / extra_payload,只是 prompt / warmup / scoring 不同。
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="启用 pic_bench_lite 数据集模式(例如 hotpotqa / longbench_qasper"
             " / longbench_narrativeqa / longbench_gov_report)。未指定则跑"
             " legacy hardcoded 3-doc 测试。",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="数据集模式下每个 mode 跑多少个样本(默认 10)",
    )
    # ── pic_bench 一致性:Zipf 共享文档池(distractor pool)────────────────
    # 默认开启,把每个 pool-eligible 样本(hotpotqa 等)填充成 8K-16K tokens 的
    # 多文档 prompt(distractor 来自其他样本、Zipf 加权),与参考实现 pic_bench
    # 的 _fill_distractor_pool 完全一致,让 F1 / 命中率可直接对比。
    # ★ 池从"已载入的样本"构建,所以要复现 pic_bench 的 200-pool 规模,
    #   --n-samples 建议 >= 50(样本越多池越丰富)。
    parser.add_argument(
        "--dataset-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="dataset 模式下启用 pic_bench 一致的 Zipf distractor 文档池"
             "(默认开启;--no-dataset-pool 关闭,退回单文档行为)。",
    )
    parser.add_argument(
        "--pool-min-tokens",
        type=int,
        default=None,
        help="distractor 池填充目标 token 下限(默认按数据集取 pic_bench 值,"
             "hotpotqa=8192)。",
    )
    parser.add_argument(
        "--pool-max-tokens",
        type=int,
        default=None,
        help="distractor 池填充 token 上限(默认按数据集取 pic_bench 值,"
             "hotpotqa=16384)。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="distractor 池构建 + 每请求 chunk shuffle 的随机种子(默认 0,"
             "保证跨 mode / 跨运行确定性)。",
    )
    parser.add_argument(
        "--answer-extractor",
        choices=["pic_bench", "flexible"],
        default="flexible",
        help="答案抽取器。flexible(默认,本仓库增强版: 优先抓 'Answer:' 后内容并"
             "跳过 GLM-5.2 的 '(Use the format: ...)' 引导语,对 GLM CoT 更鲁棒);"
             "pic_bench(与 pic_bench 对齐: extract_answer_only,取 </think> 后第一"
             "非空行,但对 GLM 常抓到引导语 → F1 偏低)。",
    )
    parser.add_argument(
        "--dataset-raw-dir",
        type=str,
        default=None,
        help="数据集原始文件(*.parquet / *.jsonl)所在目录。缺省用"
             " data/raw/<dataset_name>/",
    )
    parser.add_argument(
        "--dataset-cache",
        type=str,
        default=None,
        help="预处理后的 ProcessedSample jsonl 缓存路径。缺省用"
             " data/processed/<dataset_name>/processed.jsonl",
    )
    parser.add_argument(
        "--dataset-max-new-tokens",
        type=int,
        default=64,
        help="数据集模式下正式采样的 max_new_tokens(默认 64)。HotpotQA 类短答"
             "案数据集,GT 大多 1-6 token,64 已足够。原来的 512 会让模型生成完"
             "答案后继续吐 `!!!` 填充攻击到 512 token,平均每样本多耗 30-60s "
             "(H20 上),对 F1 无收益。真需要长生成(比如 gov_report 摘要)时"
             "再单独提高。",
    )
    parser.add_argument(
        "--query-suffix",
        type=str,
        default=(
            "Answer with only the answer word or short phrase, no explanation, "
            "and do not repeat.\n\nAnswer:"
        ),
        help="追加到 query 末尾的引导语。默认以 'Answer:' 收尾作为 completion "
             "cue,逼 GLM-5.2 直接补出答案而不是续写指令(raw /generate 不走 chat "
             "template,模型会顺着指令续写)。设为空字符串以禁用。",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        metavar="PATH",
        help="hardcoded 模式测试结束后将各 mode 的 TTFT / 命中率 / 段命中 / 加速比"
             "渲染为一张 PNG 到指定路径(matplotlib)。dataset 模式忽略此选项。",
    )
    args = parser.parse_args()

    port = args.port
    model = args.model
    tp = args.tp
    modes = args.modes

    # PIC_SYNTH_REAL_SAMPLE=<idx>: 决定性验证 —— 把一个真实 hotpotqa 样本的段落
    # 当 synthetic 固定文档、它的多跳问题当 Q,走 SYNTHETIC(run_mode,不传
    # --dataset)这条已确认忠实的路径。若 pic_a3_oracle 在这里也崩 → 铁证是
    # "隔离捕获设计对多跳内容失效",而非数据集 pool/prime_sample 路径的 bug。
    # 每段 = 该样本 context 的一个段落(title\nbody),即真实"多段+多跳"内容,
    # 但无 distractor pool、无 shuffle → 干净的 fixed prompt。
    _real_sample = os.environ.get("PIC_SYNTH_REAL_SAMPLE")
    if _real_sample is not None and not args.dataset:
        global SYS, Q, DOCS, _ANSWER_IDX
        from sglang.test.pic_bench_lite.datasets import (
            get_dataset,
            load_or_build_processed,
        )

        _ridx = int(_real_sample)
        _ndist = int(os.environ.get("PIC_SYNTH_REAL_DISTRACTORS", "0"))
        _ds = get_dataset("hotpotqa")
        _raw = Path(args.dataset_raw_dir or "data/raw/hotpotqa")
        _cache = Path(
            args.dataset_cache or "data/processed/hotpotqa/processed.jsonl"
        )
        # load enough samples to also supply distractors (pool isolation test)
        _need = _ridx + 1 + max(0, _ndist) + 3
        _samps = []
        for _s in load_or_build_processed(
            _cache, lambda: _ds.preprocess_for_pic(_raw)
        ):
            _samps.append(_s)
            if len(_samps) >= _need:
                break
        _samp = _samps[_ridx]
        # 还原成 per-paragraph 文档(hotpotqa chunks = ["\n\n".join(all_docs)])。
        # PIC_SYNTH_REAL_MERGED=1: 不拆段,整个 context 当 1 个 merged chunk(与
        # dataset 路径一致——dataset 的 own chunk 就是 merged)。用于忠实复现
        # dataset 的分段(答案在 1 个整块里,不被 shuffle 打散)。
        if os.environ.get("PIC_SYNTH_REAL_MERGED") == "1":
            _paras = [_samp.chunks[0]]
        else:
            _paras = [p.strip() for p in _samp.chunks[0].split("\n\n") if p.strip()]
        DOCS = list(_paras)
        # PIC_SYNTH_REAL_DISTRACTORS=N: 在 run_mode(忠实)路径里给目标样本的段落
        # 前后各塞 N/2 个"其它样本的整块 context"当 distractor(模拟 dataset pool)。
        # 若加了 distractor 就崩 → 铁证 pool/distractor 是 dataset 路径崩的元凶
        # (且与 prime_sample warmup 无关,因为这里仍走 run_mode)。答案段留在中间。
        if _ndist > 0:
            _dist = [
                _os.chunks[0] for _j, _os in enumerate(_samps) if _j != _ridx
            ][:_ndist]
            _h = len(_dist) // 2
            DOCS = _dist[:_h] + list(_paras) + _dist[_h:]
        _ANSWER_IDX = 0  # 真实答案位置未知(仅用于打印);机制不依赖它
        Q = (
            "Read the documents above and answer using only the documents. "
            f"Question: {_samp.query} "
            "Reply with only the answer, no explanation. Do not repeat.\n\nAnswer:"
        )
        print(f"  [REAL-SAMPLE] hotpotqa#{_ridx} id={_samp.sample_id}: "
              f"{len(_paras)} 段真实段落 + {_ndist} distractor块 = {len(DOCS)} 段, "
              f"GT={_samp.ground_truth}, Q={_samp.query!r}")

    print(f"\n{'#'*64}")
    print(f"  PIC Online Quick Test — GLM-5.2")
    print(f"  模型路径 : {model}")
    print(f"  TP={tp}  端口={port}  mem_frac={MEM_FRAC}")
    print(f"  测试模式 : {modes}")
    print(f"{'#'*64}")
    print(f"\n  prompt 数据（规格文档 §2.2）:")
    print(f"    SYS = {SYS!r}")
    print(f"    Q   = {Q!r}")
    print(f"    文档段数 N={len(DOCS)} (答案 'Mount Kilimanjaro' 在 index {_ANSWER_IDX});"
          f" 每段 × {_DOC_REPEAT}")
    print(f"    N 段合计 ≈ {sum(len(d) for d in DOCS):,} 字符")

    # ── PIC 启动参数（规格文档 §3，必须严格按此配置）────────────────────
    #
    # page_size=1  ← 关键！PIC 按 token 粒度分配/释放 KV slots，
    #                默认 page_size=64 会导致 invariant_checker 检测到
    #                "pool memory leak"（evictable 计数包含 PIC 私有 slots），
    #                服务器在 on_idle() 时 crash。
    #
    # chunked-prefill-size=-1  ← 禁用分块预填充，保证 PIC 段级缓存原子性。
    #                            如果分块，long prompt 会被拆成多块，破坏段边界对齐。
    #
    # pic-separator-str  ← 明确指定分隔符，与 prompt 中的 <<PIC_SEP>> 对应。
    #
    # 注意：--disable-piecewise-cuda-graph 已在公共参数（_start_server）中，
    # SGLANG_EAGER_INPUT_NO_COPY=1 通过 os.environ 注入到子进程环境。
    pic_extra_args = [
        "--pic-enable",
        "--page-size", "64",
        "--chunked-prefill-size", "-1",
        "--pic-separator-str", SEP,
    ]
    # sink 前缀：--pic-sink-len N 透传给 server，给每段前置 N 个 sink token。
    # 默认 0=关。★ page_size=64 场景下 N 必须是 64 的倍数，否则段长 64k+N 不
    # 再 page 对齐，pic_alloc_for_extend 会崩（见文件顶部 _pad_text_to_multiple
    # 注释与 pic_alloc.py:334）。
    if getattr(args, "pic_sink_len", 0):
        if args.pic_sink_len % PIC_ALIGN != 0:
            print(f"  [warn] --pic-sink-len={args.pic_sink_len} 不是 {PIC_ALIGN} 的"
                  f"倍数，page_size=64 下大概率会崩；建议用 {PIC_ALIGN} 的倍数。")
        pic_extra_args += ["--pic-sink-len", str(args.pic_sink_len)]
        if args.pic_sink_token_id is not None:
            pic_extra_args += ["--pic-sink-token-id", str(args.pic_sink_token_id)]
        print(f"  [pic-sink] 已启用：每段前置 {args.pic_sink_len} 个 sink token"
              f"（token_id={args.pic_sink_token_id if args.pic_sink_token_id is not None else 'auto'}）")
    # --a3-check-layers 透传给 server（多层重选研究探针）。仅在提供多值时透传；
    # 单值 [1] / 省略 == 当前单层 pic_a3。只并入 pic_a3 / pic_cacheblend 的 extra_args。
    a3_check_layers_args = []
    if getattr(args, "a3_check_layers", None):
        a3_check_layers_args = ["--a3-check-layers", *[str(x) for x in args.a3_check_layers]]
        _n_cl = len(args.a3_check_layers)
        print(f"  [a3-check-layers] {args.a3_check_layers} "
              f"（{'多层重选' if _n_cl > 1 else '单层'}；Phase B keep-alive 需 "
              f"env SGLANG_PIC_A3_KEEP_ALIVE=1，否则 Phase A 单层应用+探针日志）")
    # pic_a3_oracle 的重选/重算层:用 --oracle-layers(默认 1 20 40 60,每次运行可指定),
    # 透传成 server 的 --a3-check-layers。强制含 layer 1。
    _oracle_layers = [str(int(x)) for x in (args.oracle_layers or [1, 20, 40, 60])]
    if "1" not in _oracle_layers:
        _oracle_layers = ["1"] + _oracle_layers
    print(f"  [oracle-layers] {_oracle_layers}（Phase B 窗口化重选:每层用 in-process "
          f"oracle 的隔离 hidden 重选 imp + 从准确输入重算 imp KV）")
    # --enable-pic-a3-clip has been removed from server_args (pic_a3 / pic_cacheblend
    # now do clip natively — see comment on mode_configs["pic_a3"] below).
    # Passing --enable-clip is a no-op; kept for CLI compat with old scripts.
    clip_extra_args: List[str] = []
    if getattr(args, "enable_clip", False):
        print("  [warn] --enable-clip is deprecated / no-op: pic_a3 clips natively "
              "and server_args no longer accepts --enable-pic-a3-clip.")

    # ── 段 64 对齐（路径 A 验证）─────────────────────────────────────────────
    # 把每段(SYS/C1/C2/C3/Q)的 token 数 padding 到 64 的倍数，使 PIC 的段 slot
    # 自动 page 对齐(满足 DSA 的 page_size=64 寻址)，从而在不改内核的前提下验证
    # PIC 能否在 GLM-5.2 上跑通。非 PIC 模式用同样 padding 的段(仅去掉 SEP)，保证
    # 与 PIC 的实际 token 序列一致，FDT 对比才公平(SEP 在 split_and_tokenize 中不产生 token)。
    sys_seg, q_seg = SYS, Q
    doc_segs = list(DOCS)   # N 段(可能未对齐);下面按需 pad
    if PIC_PAD_TO_64 == "1":
        print(f"\n  [段对齐] 加载 tokenizer，将每段 padding 到 {PIC_ALIGN} 的倍数……")
        from transformers import AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        def _align(name, text, prepend=False):
            padded, o, p = _pad_text_to_multiple(
                text, _tok, PIC_ALIGN, prepend=prepend
            )
            flag = "对齐✓" if p % PIC_ALIGN == 0 else "未对齐✗(填充失败)"
            print(f"    {name:<4}: {o:>5} → {p:>5} tokens  [{flag}]")
            return padded

        sys_seg = _align("SYS", SYS)
        doc_segs = [_align(f"C{i+1}", d) for i, d in enumerate(DOCS)]
        # Q padding is PREPEND (same fix as the dataset path): the ".,!?" pad
        # chars go to the FRONT so the prompt ends with the real question's
        # "Answer:" cue. With APPEND the prompt ended "...Answer:.,!?" and
        # pic_a3 parroted the padding — its first token was "?" (id 30), which
        # is in _clean_stop, so generation halted immediately → empty output.
        q_seg = _align("Q", Q, prepend=True)

    # 用(可能已对齐的)段重新拼接各 prompt。列表化以支持 N 段(N=3 时 == 原始 3 段)。
    # 非 PIC：无 SEP 直接拼接；PIC：段间插入 SEP(SEP 不产生 token，故两者实际 token 序列一致)。
    # PIC_SYNTH_SHUFFLE=1: 打乱 MEASURE prompt 里的段顺序(warmup 仍按原 doc 隔离
    # 捕获,内容寻址与顺序无关)。用于隔离 dataset 崩溃的触发因子:dataset 的 pool
    # 把段 shuffle 了(段 cache 在 warmup pos~64,measure 却在乱序大 pos → delta_pos
    # 大且各段迥异)。synthetic 连续顺序(N<=6)不崩;若 shuffle 后崩 → shuffle/乱序
    # 大 delta_pos 是触发因子(且此时仍是 fixed prompt,可 K-dump)。
    _measure_docs = list(doc_segs)
    if os.environ.get("PIC_SYNTH_SHUFFLE") == "1":
        import random as _synr
        _synr.Random(1234).shuffle(_measure_docs)
        _ans_new = _measure_docs.index(doc_segs[_ANSWER_IDX])
        print(f"  [synth-shuffle] measure 段顺序已打乱(seed=1234);答案段现在 measure "
              f"index={_ans_new}(warmup 仍按原 doc 隔离捕获)")
    prompt_full = sys_seg + "".join(_measure_docs) + q_seg
    # 非 PIC warmup:首段 + 末段(full_recompute / prefix_cache 用,保持部分命中对照)。
    w1_full = f"{sys_seg}{doc_segs[0]}{q_seg}"
    w3_full = f"{sys_seg}{doc_segs[-1]}{q_seg}"
    # PIC：sys SEP doc1 SEP ... SEP docN SEP q(measure 段顺序可能已 shuffle)
    pic_prompt = sys_seg + SEP + SEP.join(_measure_docs) + SEP + q_seg
    # 每段一个 PIC warmup(sys<SEP>doc_i<SEP>q):让 pic/pic_a3/pic_cacheblend 命中全部
    # N 段(measure 里 SYS + N 段都 hit,只有 Q 走 fresh)。答案段在中间 index N//2。
    # warmup 用原始 doc_segs 顺序(内容寻址,与 measure 顺序无关)。
    pic_doc_warmups = [f"{sys_seg}{SEP}{d}{SEP}{q_seg}" for d in doc_segs]
    pic_all_warmups = pic_doc_warmups   # pic/pic_a3/pic_cacheblend 全段覆盖
    # 向后兼容名(N>=3 时存在;仅历史代码/打印引用)。
    pic_w1 = pic_doc_warmups[0]
    pic_w2 = pic_doc_warmups[1] if len(pic_doc_warmups) > 1 else pic_doc_warmups[0]
    pic_w3 = pic_doc_warmups[-1]
    # pic_a3_oracle 专用 pre-warm：先把 SYS 段缓存上（SYS+Q，SYS 非末段→会被缓存）。
    # 否则第一个 doc warmup 是冷启全 miss，keepalive/oracle 钩子不触发 → 首段抓不到。
    # 有了这个 pre-warm，所有 doc warmup 都变成"SYS 命中 + 文档 miss"→ 钩子触发 → 全段 capture。
    pic_sys_prewarm = f"{sys_seg}{SEP}{q_seg}"

    # pic_a3_oracle warmup: SYS pre-warm + 全部 N 段 warmup。
    # --oracle-warmup-docs 仍可截断(诊断:只 warmup 前 K 段),但默认覆盖全部 N 段
    # (N>3 时必须全覆盖,否则未 warmup 段在 measure 命中却无 oracle capture →
    #  reselect_full BAIL → 退回全长 keepalive)。K<=0 或 >N 时取全部。
    _oracle_wu_docs = int(getattr(args, "oracle_warmup_docs", 3))
    if _oracle_wu_docs <= 0 or _oracle_wu_docs > len(pic_doc_warmups):
        _oracle_wu_docs = len(pic_doc_warmups)
    _oracle_warmup_prompts = [pic_sys_prewarm] + pic_doc_warmups[:_oracle_wu_docs]
    print(f"  [synth-docs] N={len(doc_segs)} 段 (答案段 index={_ANSWER_IDX}, "
          f"'Mount Kilimanjaro'); oracle warmup 覆盖前 {_oracle_wu_docs} 段 "
          f"({len(_oracle_warmup_prompts)} 个 warmup: SYS-prewarm + {_oracle_wu_docs} 段); "
          f"measure = SYS + {len(doc_segs)} 段 + Q")


    # ── --gen-prompt-file: 导出并立即退出(不启动服务器)──
    # 把 a3/cacheblend 模式实际会发送的 prompt_full(已经过 PIC_PAD_TO_64 对齐)
    # 写入文件,供 scripts/precompute_kv_mla.py --prompt-file 直接消费。
    # 这样 precompute 生成的 KV 与本脚本 a3 请求的 tokenized 序列 byte-for-byte
    # 一致,先决条件对齐,不会出现 total_len 不匹配的 shape 错误。
    if args.gen_prompt_file:
        out_path = os.path.abspath(args.gen_prompt_file)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(prompt_full)
        print(f"\n  [gen-prompt-file] 已写入 {len(prompt_full)} 字符 → {out_path}")
        print(f"  下一步:\n"
              f"    python scripts/precompute_kv_mla.py \\\n"
              f"        --model {model} \\\n"
              f"        --prompt-file {out_path} \\\n"
              f"        --output-dir /tmp/precomputed_kv/ \\\n"
              f"        --tp {tp}\n"
              f"  然后:\n"
              f"    python quick_test_online.py --modes full_recompute a3 --tp {tp}")
        sys.exit(0)

    # ── 各模式配置（prompt、warmup、启动参数）──────────────────────────────
    # A3/CacheBlend 共用启动参数:
    #   --enable-a3 / --enable-cacheblend  开启选择性重算分支
    #   --recomp-ratio                      重算比例
    #   --disable-cuda-graph                imp_len 每次不同,禁 graph 捕获
    #   --chunked-prefill-size -1           关闭分块 prefill (baseline 约束)
    #   --disable-overlap-schedule          关闭 overlap (baseline 约束)
    # 请求携带 5 个字段: reuse_method / recomp_ratio / reuse_last_len /
    # reuse_prefix_len / precomputed_kv_path (通过 extra_payload 顶层传输)。
    a3_common_args = [
        "--recomp-ratio", str(args.a3_recomp_ratio),
        "--disable-cuda-graph",
        "--chunked-prefill-size", "-1",
        "--disable-overlap-schedule",
        # ★ A3 必须禁用 radix cache: A3 假设 precompute 位置 == 推理位置,
        # 但 radix cache 命中会让第 2 次请求只跑 tail token 的 extend,导致
        # a3_hit_mask (当前 batch 长度) 与 a3_precomputed_latent (完整 prompt 长度)
        # 形状不匹配,DSA indexer 里 `key[_a3_hit] = _cur_idx_k[_a3_hit]` 崩溃。
        # 详见 A3andcacheblend.md §9.1。
        "--disable-radix-cache",
    ]
    a3_common_payload = {
        "recomp_ratio": args.a3_recomp_ratio,
        "precomputed_kv_path": args.a3_precomputed_kv,
    }

    mode_configs = {
        # full_recompute:与 A3 用相同 flags(--disable-cuda-graph 等)做公平对比。
        # 之前 baseline 只加 --disable-radix-cache,享受 CUDA graph + overlap +
        # chunked prefill 全套优化,与关掉所有优化的 A3 比不公平。
        "full_recompute": {
            "desc": "全量重计算(与 A3 相同 flags,fair baseline)",
            "extra_args": [
                "--disable-radix-cache",
                "--disable-cuda-graph",
                "--chunked-prefill-size", "-1",
                "--disable-overlap-schedule",
            ],
            "prompt": prompt_full,
            "warmup_prompts": [w1_full, w3_full],
        },
        # prefix_cache：RadixCache 开启，其余 flags 与 full_recompute 对齐
        # (关 cuda-graph / overlap / chunked-prefill),这样加速比里只反映
        # "RadixCache 命中省下的 forward",跟 pic 加速比可以直接比较。
        # FDT 结果反映 "RadixCache 是否引入精度漂移" (预期 FDT=32)。
        "prefix_cache": {
            "desc": "前缀缓存（RadixCache，与 full_recompute 相同 flags）",
            "extra_args": [
                "--disable-cuda-graph",
                "--chunked-prefill-size", "-1",
                "--disable-overlap-schedule",
                # 注意:不加 --disable-radix-cache,让 RadixCache 保持开启
            ],
            "prompt": prompt_full,
            "warmup_prompts": [w1_full, w3_full],
        },
        # pic：段级缓存。warmup 覆盖 C1/C2/C3 全部 3 段(与 pic_a3 / pic_cacheblend
        # 一致),test 时 SYS+C1+C2+C3 全命中、只有 Q 走 fresh。答案在 C2(中间段),
        # 三个 PIC 模式都缓存并复用 C2,对比才公平。
        "pic": {
            "desc": "PIC 段级缓存（--pic-enable --disable-piecewise-cuda-graph）",
            "extra_args": [
                *pic_extra_args,
                "--disable-cuda-graph",
                "--disable-overlap-schedule",
                "--disable-radix-cache",
            ],
            "prompt": pic_prompt,
            "warmup_prompts": pic_all_warmups,
        },
        # a3: A³ 选择性重算 (Q-K attention top-k)
        "a3": {
            "desc": f"A³ 选择性重算 (--enable-a3 --recomp-ratio {args.a3_recomp_ratio})",
            "extra_args": ["--enable-a3", *a3_common_args],
            "prompt": prompt_full,
            "warmup_prompts": [],   # A3 不需要 warmup 制造命中
            "extra_payload": {"reuse_method": "debug", **a3_common_payload},
        },
        # cacheblend: CacheBlend 选择性重算 (V L2-diff top-k)
        "cacheblend": {
            "desc": f"CacheBlend 选择性重算 (--enable-cacheblend --recomp-ratio {args.a3_recomp_ratio})",
            "extra_args": ["--enable-cacheblend", *a3_common_args],
            "prompt": prompt_full,
            "warmup_prompts": [],
            "extra_payload": {"reuse_method": "blend", **a3_common_payload},
        },
        # pic_a3: PIC 段级缓存 + A³ 选择性重算(新路径,无 .pt 文件)。
        # latent_old 从 kv_pool 读(PIC 已 pre-populate),不需要 precomputed_kv。
        # pic_a3: PIC 段级缓存 + A³ 选择性重算(新路径,无 .pt 文件)。
        # latent_old 从 kv_pool 读(PIC 已 pre-populate),不需要 precomputed_kv。
        # ★ warmup 用 3 段全覆盖(C1/C2/C3 都 warmup),让 test 时 SYS+C1+C2+C3 全命中,
        #   只有 Q 段走 fresh。中间层挑 imp_indices 重算 15%。
        "pic_a3": {
            "desc": f"PIC + A³ (--pic-enable --enable-a3 --recomp-ratio {args.a3_recomp_ratio})",
            "extra_args": [
                *pic_extra_args,
                *clip_extra_args,
                *a3_check_layers_args,
                "--enable-a3",
                "--recomp-ratio", str(args.a3_recomp_ratio),
                "--disable-cuda-graph",
                "--disable-overlap-schedule",
                "--disable-radix-cache",
            ],
            "prompt": pic_prompt,
            "warmup_prompts": pic_all_warmups,
            "extra_payload": {},   # reuse_method 由 server-side enable_a3 fallback 到 "debug"
        },
        # pic_cacheblend: PIC 段级缓存 + CacheBlend 选择性重算(新路径,无 .pt 文件)。
        "pic_cacheblend": {
            "desc": f"PIC + CacheBlend (--pic-enable --enable-cacheblend --recomp-ratio {args.a3_recomp_ratio})",
            "extra_args": [
                *pic_extra_args,
                *clip_extra_args,
                *a3_check_layers_args,
                "--enable-cacheblend",
                "--recomp-ratio", str(args.a3_recomp_ratio),
                "--disable-cuda-graph",
                "--disable-overlap-schedule",
                "--disable-radix-cache",
            ],
            "prompt": pic_prompt,
            # warmup 覆盖 C1/C2/C3 全部 3 段(与 pic_a3 一致),让答案所在的 C2
            # 也被缓存+选择性重算,三个 PIC 模式对比一致。
            "warmup_prompts": pic_all_warmups,
            "extra_payload": {},   # reuse_method 由 server-side enable_cacheblend fallback 到 "blend"
        },
        # pic_a3_oracle (方案①, 全 in-process): PIC + A³, Phase B keepalive
        # 窗口化重选 + in-process oracle。server 自己 warmup 时按 PIC 段 hash 存下
        # C1/C2/C3 各 oracle 层的隔离输入 hidden；measure 时命中段注入回去，让每个
        # --a3-check-layers 层用准确(未漂移)输入重选 imp + 重算 imp KV(做法1)。
        # 开关: SGLANG_PIC_A3_KEEP_ALIVE=1 + SGLANG_PIC_A3_ORACLE=1(mode 循环里设)。
        "pic_a3_oracle": {
            "desc": (
                f"PIC + A³ oracle 窗口化重选 (Phase B keepalive + in-process oracle, "
                f"oracle-layers {' '.join(_oracle_layers)}, "
                f"--recomp-ratio {args.a3_recomp_ratio})"
            ),
            "extra_args": [
                *pic_extra_args,
                *clip_extra_args,
                # 多层 → server_args.pic_a3_multiselect=True → 触发 Phase B。
                "--a3-check-layers", *_oracle_layers,
                "--enable-a3",
                "--recomp-ratio", str(args.a3_recomp_ratio),
                "--disable-cuda-graph",
                "--disable-overlap-schedule",
                "--disable-radix-cache",
            ],
            "prompt": pic_prompt,
            # pre-warm 先缓存 SYS，之后 pic_w1/w2/w3 就都是"SYS 命中 + 文档 miss"→
            # keepalive 触发 → C1/C2/C3 各段的隔离 hidden 全被 capture（否则冷启的
            # pic_w1 全 miss 不触发钩子，C1 抓不到）。SYS 段本身不需要 oracle（位置 0、
            # 因果下只看自己、不漂移），故不 capture 它无妨。
            "warmup_prompts": _oracle_warmup_prompts,
            "extra_payload": {},   # reuse_method 由 server-side enable_a3 fallback 到 "debug"
        },
        # pic_a3_reselect / pic_cacheblend_reselect were removed (ragkv-style
        # pic_a3/pic_cacheblend now do the layer-1 Q·K pick + clip natively,
        # obsoleting the reselect variant). See plan
        # polished-inventing-squirrel.md and git log for the rollback.
    }

    # ── 执行各模式测试 ──────────────────────────────────────────────────────
    # 若测试 a3 / cacheblend,先检查预计算 KV 是否存在;不存在则报错并给出生成命令。
    # 支持 {rank} 模板 (多 TP 时每 rank 一个 .pt),检查 rank0 存在即可。
    if any(m in modes for m in ("a3", "cacheblend")):
        _probe_path = args.a3_precomputed_kv.format(rank=0) if "{rank}" in args.a3_precomputed_kv else args.a3_precomputed_kv
        if not os.path.exists(_probe_path):
            print(
                f"\n  [错误] A3/CacheBlend 模式需要预计算 KV 文件,未找到:\n"
                f"    {_probe_path}\n"
                f"  请先运行 (tp 需与本次一致,当前 tp={tp}):\n"
                f"    python scripts/precompute_kv_mla.py \\\n"
                f"        --model {model} \\\n"
                f"        --prompt-file <your_prompt.txt> \\\n"
                f"        --output-dir {os.path.dirname(_probe_path) or '/tmp/precomputed_kv'} \\\n"
                f"        --tp {tp}\n"
                f"  或者从 --modes 中移除 a3 / cacheblend。",
                file=sys.stderr,
            )
            sys.exit(1)
        _tot = os.path.getsize(_probe_path) / 1e6
        print(f"\n  [A3] 预计算 KV: {args.a3_precomputed_kv}  "
              f"(rank0 分片 {_tot:.1f} MB)")

    # ── pic_bench_lite dataset 模式:载入 N 个 ProcessedSample ──────────────
    # dataset_samples 非空时,后面的 mode 循环走 run_mode_dataset 分支。
    dataset_samples: List = []
    if args.dataset:
        # a3 / cacheblend 需要预计算 KV(基于固定 prompt)——不兼容 dataset 模式
        # (每个样本 prompt 都不同)。这里直接过滤掉,防止使用体验灾难。
        if any(m in modes for m in ("a3", "cacheblend")):
            print(
                "\n  [错误] --dataset 与 a3 / cacheblend 模式不兼容:后者需要"
                "对某个固定 prompt precompute 的 KV,而 dataset 中每个样本"
                "prompt 都不同。请从 --modes 中移除 a3 / cacheblend,或改用"
                "pic_a3 / pic_cacheblend(它们不需要 .pt 文件)。",
                file=sys.stderr,
            )
            sys.exit(1)
        raw_dir = Path(args.dataset_raw_dir or f"data/raw/{args.dataset}")
        cache_path = Path(
            args.dataset_cache
            or f"data/processed/{args.dataset}/processed.jsonl"
        )
        print(f"\n  [dataset] 加载 {args.dataset}: raw_dir={raw_dir}  "
              f"cache={cache_path}  n_samples={args.n_samples}")
        try:
            dataset_samples = _load_dataset_samples(
                args.dataset, raw_dir, cache_path, args.n_samples,
            )
        except Exception as _e:
            print(f"  [错误] 数据集加载失败: {_e}", file=sys.stderr)
            print(
                f"  提示:如果尚未下载数据,可先执行 (hotpotqa 示例):\n"
                f"    mkdir -p {raw_dir}\n"
                f"    curl -L -o {raw_dir}/validation.parquet \\\n"
                f"        \"https://huggingface.co/api/datasets/hotpot_qa/parquet/distractor/validation/0.parquet\"\n"
                f"  或(推荐)用 huggingface-cli / datasets 库下载 THUDM/LongBench 等。",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  [dataset] 载入 {len(dataset_samples)} 个样本 "
              f"(平均 chunk 数={sum(len(s.chunks) for s in dataset_samples) / len(dataset_samples):.1f})")

        # pic_bench 一致性:构建 Zipf distractor 文档池(一次性、mode 无关,挂到
        # 每个 sample.meta["_pool_chunks"],让所有 mode 用同一份 pooled prompt →
        # 公平对比,且与 pic_bench 的 _fill_distractor_pool 完全一致)。
        from sglang.test.pic_bench_lite.pool import (
            POOL_DATASETS,
            fill_distractor_pool,
        )
        if args.dataset_pool and args.dataset in POOL_DATASETS:
            from transformers import AutoTokenizer

            _pool_tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            fill_distractor_pool(
                dataset_samples, _pool_tok, sep=SEP, seed=args.seed,
                min_tokens_override=args.pool_min_tokens,
                max_tokens_override=args.pool_max_tokens,
            )
            _npc = [len(s.meta.get("_pool_chunks", [])) for s in dataset_samples]
            _padded = sum(1 for n in _npc if n > 0)
            print(f"  [dataset] distractor pool 已启用: {_padded}/{len(dataset_samples)} "
                  f"个样本被填充, 平均 pool chunks={sum(_npc) / max(1, len(_npc)):.1f}"
                  f"  (--no-dataset-pool 可关闭; 池从已载入样本构建, N 越大越贴近 "
                  f"pic_bench 200-pool)")
        elif args.dataset_pool:
            print(f"  [dataset] {args.dataset} 不在 pool-eligible 集合内, "
                  f"跳过 distractor 池 (pool-eligible: {sorted(POOL_DATASETS)})")

    results: Dict[str, Dict] = {}
    for mode in modes:
        cfg = mode_configs[mode]
        print(f"\n>>> 开始 [{mode}] — {cfg['desc']}")
        try:
            # DeepGEMM fast warmup：必须通过 os.environ 注入（与 SGLANG_EAGER_INPUT_NO_COPY
            # 相同机制），才能让 TP worker 子进程继承。实测仅在 _start_server 的 env
            # 字典里设置时，只有 launch_server 主进程拿到，真正执行 warmup 的 TP worker
            # 不会继承该变量 → _FAST_WARMUP=False → 走 m_max=65536 分支导致超时。
            os.environ["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = FAST_WARMUP
            # STRIDE_MULT 同理：必须 os.environ 注入，否则 TP worker 拿不到。
            os.environ["SGLANG_JIT_DEEPGEMM_FAST_WARMUP_STRIDE_MULT"] = FAST_WARMUP_STRIDE_MULT

            # oracle / keepalive envs：默认清掉，避免泄漏到其它 mode；只有
            # pic_a3_oracle 下面重设。这些 env 必须在 launch 前设(TP worker 继承)。
            os.environ.pop("SGLANG_PIC_A3_KEEP_ALIVE", None)
            os.environ.pop("SGLANG_PIC_A3_ORACLE", None)
            os.environ.pop("SGLANG_PIC_A3_CLIP_CAPTURE", None)

            # K-dump 观测台: 每个 mode dump 到自己的 tag，否则 full_recompute / pic /
            # pic_cacheblend 会都落到 auto-tag "ref" 互相覆盖。仅在开了 dump 时设，
            # env 经 _start_server 的 os.environ.copy() 传到 TP worker。
            if os.environ.get("SGLANG_PIC_KDUMP_DIR"):
                os.environ["SGLANG_PIC_KDUMP_TAG"] = mode

            # pic_a3_oracle：全 in-process（方案①）。开 Phase B keepalive（在每个
            # --a3-check-layers 层做窗口化重选）+ in-process oracle。server 自己
            # warmup(pic_w1/w2/w3) 时首次算到某文档段就按 PIC 段 hash 存下它各
            # oracle 层的隔离输入 hidden；measure 时该段命中就注入回去，让该层重
            # 选 imp 并从准确(未漂移)输入重算 imp KV。无单独 capture server / 文件。
            # dataset 模式每样本 prompt 不同 → oracle 命中有限，keepalive 仍启用
            # （趋近普通 Phase B）。
            if mode == "pic_a3_oracle":
                os.environ["SGLANG_PIC_A3_KEEP_ALIVE"] = "1"
                # 诊断开关:PIC_A3_NO_INJECT=1 → 只开 Phase-B keepalive,不开 oracle
                # inject(隔离 keepalive vs inject:若此时 pic_a3_oracle 仍崩,说明
                # 根因是 keepalive 机制本身,与 oracle inject / C1 无关)。
                if os.environ.get("PIC_A3_NO_INJECT") == "1":
                    print("  [diag] PIC_A3_NO_INJECT=1 → plain Phase-B keepalive "
                          "(不设 SGLANG_PIC_A3_ORACLE,零 inject)")
                else:
                    os.environ["SGLANG_PIC_A3_ORACLE"] = "1"
                # Route-A clip (窗口化重选 + clip 收窄到 miss∪imp) = 已验证的加速
                # 路径(合成测 3.5x + FDT5-6)。默认开启,让 pic_a3_oracle 自包含
                # (不再依赖 driver 脚本外部 export SGLANG_PIC_A3_CLIP_CAPTURE=1)。
                # PIC_A3_NO_CLIP=1 → 退回 full-length keepalive(正确但零加速),
                # 用于隔离诊断 capture/inject 本身是否正确(不引入 clip reselect 变量)。
                if os.environ.get("PIC_A3_NO_CLIP") == "1":
                    os.environ.pop("SGLANG_PIC_A3_CLIP_CAPTURE", None)
                    print("  [diag] PIC_A3_NO_CLIP=1 → full-length keepalive "
                          "(不 clip,零加速;仅验证 capture/inject 正确性)")
                else:
                    os.environ["SGLANG_PIC_A3_CLIP_CAPTURE"] = "1"
                if dataset_samples:
                    print("  [oracle] dataset 模式:oracle 存储按 PIC 段 hash 内容寻址,"
                          "warmup 时逐段(sys<sep>chunk<sep>warmup)捕获隔离 hidden,"
                          "measure 命中即注入 → 跨样本 / distractor 池均可命中"
                          "(前提:该段已在 warmup 覆盖,见 run_mode_dataset 的 SYS 预热"
                          "+ 全样本 chunk 预热)。")

            # PIC 模式需要额外环境变量（在 _start_server 内部处理，这里通过
            # 全局环境变量提前设置，让子进程继承）
            if mode in ("pic", "pic_a3", "pic_cacheblend", "pic_a3_oracle"):
                os.environ["SGLANG_EAGER_INPUT_NO_COPY"] = "1"
                # GLM-5.2 (DeepSeek DSA) 在 CUDA 上强制 page_size=64,覆盖了 PIC 需要
                # 的 page_size=1。这会让 invariant_checker 把 PIC 段级缓存私有持有的
                # KV slot 误判为 "pool memory leak",在 on_idle() 时抛错使所有 TP
                # scheduler 崩溃(表现为客户端 RemoteDisconnected)。
                # 将 IDLE 严格内存检查降级为仅警告以避免崩溃。
                # 注意:这是绕过误报;page_size=64 下 PIC 段缓存的命中率/正确性仍需另行核对。
                os.environ["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"
            else:
                os.environ.pop("SGLANG_EAGER_INPUT_NO_COPY", None)
                os.environ.pop("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE", None)

            if dataset_samples:
                # 每 mode 用相同 samples,只有 separator / extra_args /
                # extra_payload 变。pad_to_64 与 legacy 路径保持一致(PIC 系列
                # 需要,其他模式无害)。
                _pic_family = mode in ("pic", "pic_a3", "pic_cacheblend", "pic_a3_oracle")
                _separator = SEP if _pic_family else ""
                # PIC_FORCE_PAD_ALL=1: apply pad_to_64 to ALL modes (incl.
                # full_recompute) — isolating experiment to test whether the
                # .,!? 64-align padding (required by DSA for pic) is what tanks
                # F1, independent of KV reuse. Default off (only pic family pads).
                _pad_all = os.environ.get("PIC_FORCE_PAD_ALL", "0") == "1"
                result = run_mode_dataset(
                    mode=mode,
                    port=port,
                    model=model,
                    tp=tp,
                    samples=dataset_samples,
                    extra_args=cfg["extra_args"],
                    separator=_separator,
                    extra_payload=cfg.get("extra_payload"),
                    query_suffix=args.query_suffix or None,
                    max_new_tokens=args.dataset_max_new_tokens,
                    pad_to_64=(PIC_PAD_TO_64 == "1" and (_pic_family or _pad_all)),
                    # non-PIC 模式做 warmup 也无害(只是让 prefix cache 预热),
                    # 但 a3 / cacheblend 系列的 payload 假设"每次请求同一 prompt",
                    # warmup 会污染判定;这里 dataset 模式已禁用 a3/cacheblend,
                    # 剩下的 full_recompute / prefix_cache / pic 系列 warmup 都合理。
                    do_warmup=True,
                    ttft_probe=True,
                    seed=args.seed,
                    answer_extractor=args.answer_extractor,
                )
                results[mode] = result
                _done = (f">>> [{mode}] 完成: F1_mean={result['f1_mean']:.3f}  "
                         f"TTFT_p50={result['ttft_p50']:.3f}s  "
                         f"n_samples={result['n_samples']}")
                if _pic_family:
                    _done += (f"  PIC命中率(mean)="
                              f"{result.get('pic_hit_rate_mean', 0):.1%}")
                else:
                    _done += (f"  命中率(mean)="
                              f"{result.get('hit_rate_mean', 0):.1%}")
                print(_done)
            else:
                result = run_mode(
                    mode=mode,
                    port=port,
                    model=model,
                    tp=tp,
                    prompt=cfg["prompt"],
                    warmup_prompts=cfg["warmup_prompts"],
                    extra_args=cfg["extra_args"],
                    extra_payload=cfg.get("extra_payload"),
                )
                results[mode] = result
                _done = (f">>> [{mode}] 完成: TTFT={result['ttft']:.3f}s  "
                         f"prompt_tokens={result['prompt_tokens']}")
                if mode in ("pic", "pic_a3", "pic_cacheblend", "pic_a3_oracle"):
                    _done += (f"  PIC段命中={result.get('pic_hit_segments', 0)}"
                              f"/{result.get('pic_total_segments', 0)}"
                              f"  PIC命中率={result.get('pic_hit_rate', 0):.1%}")
                else:
                    _done += f"  命中率={result['hit_rate']:.1%}"
                print(_done)
        except Exception as e:
            import traceback
            print(f">>> [{mode}] 失败: {e}")
            traceback.print_exc()
            results[mode] = {"mode": mode, "error": str(e)}

    # ── 汇总报告 ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*64}")
    print("  测试结果汇总")
    print(f"{'='*64}")

    if dataset_samples:
        # dataset 模式:F1 mean / TTFT p50/p95 / 命中率 mean
        print(f"\n  数据集: {args.dataset}  样本数: {len(dataset_samples)}")
        header = (f"  {'模式':<16} {'F1_mean':<10} {'F1_median':<10} "
                  f"{'TTFT_p50 (s)':<14} {'TTFT_p95 (s)':<14} "
                  f"{'命中率_mean':<12} {'PIC段命中(mean)'}")
        print(header)
        print(f"  {'-'*88}")
        for mode in modes:
            r = results.get(mode, {})
            if "error" in r:
                print(f"  {mode:<16} ERROR: {r['error']}")
                continue
            _pic_family = mode in ("pic", "pic_a3", "pic_cacheblend", "pic_a3_oracle")
            hit_col = (f"{r.get('pic_hit_rate_mean', 0):.1%}"
                       if _pic_family
                       else f"{r.get('hit_rate_mean', 0):.1%}")
            pic_col = "-"
            if _pic_family:
                per = r.get("per_sample", [])
                if per:
                    seg_mean = statistics.mean(
                        p.get("pic_hit_segments", 0) for p in per)
                    tot_mean = statistics.mean(
                        p.get("pic_total_segments", 0) for p in per)
                    pic_col = f"{seg_mean:.1f}/{tot_mean:.1f}"
            print(f"  {mode:<16} "
                  f"{r.get('f1_mean', 0):<10.3f} "
                  f"{r.get('f1_median', 0):<10.3f} "
                  f"{r.get('ttft_p50', 0):<14.3f} "
                  f"{r.get('ttft_p95', 0):<14.3f} "
                  f"{hit_col:<12} {pic_col}")

        # TTFT 加速比 vs full_recompute
        baseline_ttft = results.get("full_recompute", {}).get("ttft_p50")
        if baseline_ttft:
            print(f"\n  TTFT_p50 加速比(相对 full_recompute="
                  f"{baseline_ttft:.3f}s):")
            for mode in modes:
                if mode == "full_recompute":
                    continue
                r = results.get(mode, {})
                t = r.get("ttft_p50")
                if t:
                    speedup = baseline_ttft / t if t > 0 else float("inf")
                    print(f"    {mode:<20}: {speedup:.2f}x  ({t:.3f}s)")

        # F1 相对 baseline 的绝对差
        baseline_f1 = results.get("full_recompute", {}).get("f1_mean")
        if baseline_f1 is not None:
            print(f"\n  F1 vs full_recompute baseline (F1_mean="
                  f"{baseline_f1:.3f}):")
            for mode in modes:
                if mode == "full_recompute":
                    continue
                r = results.get(mode, {})
                f = r.get("f1_mean")
                if f is not None:
                    delta = f - baseline_f1
                    delta_pct = (delta / baseline_f1 * 100.0) if baseline_f1 > 0 else 0.0
                    sign = "+" if delta >= 0 else ""
                    print(f"    {mode:<20}: F1={f:.3f}  Δ={sign}{delta:+.3f} "
                          f"({sign}{delta_pct:+.1f}%)")

        print(f"\n{'='*64}")
        print("  测试完成!")
        print(f"{'='*64}\n")
        return

    print(f"\n  {'模式':<16} {'TTFT (s)':<10} {'命中率':<10} "
          f"{'PIC段命中':<12} {'prompt_tokens'}")
    print(f"  {'-'*64}")
    for mode in modes:
        r = results.get(mode, {})
        if "error" in r:
            print(f"  {mode:<16} ERROR: {r['error']}")
        else:
            # 命中率列：PIC 模式用 PIC 命中率（cached_tokens 对 PIC 无意义），
            # 其他模式用标准前缀缓存命中率。
            hit_col = (
                f"{r.get('pic_hit_rate', 0):.1%}"
                if mode in ("pic", "pic_a3", "pic_cacheblend", "pic_a3_oracle")
                else f"{r.get('hit_rate', 0):.1%}"
            )
            pic_col = (
                f"{r.get('pic_hit_segments', 0)}/{r.get('pic_total_segments', 0)}"
                if mode in ("pic", "pic_a3", "pic_cacheblend", "pic_a3_oracle")
                else "-"
            )
            print(f"  {mode:<16} {r.get('ttft', 0):<10.3f} "
                  f"{hit_col:<10} {pic_col:<12} {r.get('prompt_tokens', 0)}")

    # ── 正确性对比：FDT（首个分歧 token，规格文档 §5.2）────────────────────
    print(f"\n  正确性分析（FDT = 首个分歧 token 位置，值越大越好，最大=32）:")
    baseline_ids = results.get("full_recompute", {}).get("ids32", [])

    if baseline_ids:
        prefix_ids = results.get("prefix_cache", {}).get("ids32", [])
        pic_ids    = results.get("pic", {}).get("ids32", [])

        # prefix_cache vs full_recompute → 噪声底噪（规格文档 §5.2）
        noise_fdt = first_divergence_token(baseline_ids, prefix_ids) if prefix_ids else None
        pic_fdt   = first_divergence_token(baseline_ids, pic_ids)   if pic_ids   else None

        print(f"  基准: full_recompute  ids32(前10) = {baseline_ids[:10]}")
        if noise_fdt is not None:
            same = noise_fdt == min(len(baseline_ids), len(prefix_ids))
            print(f"  prefix_cache vs baseline: FDT={noise_fdt}"
                  f"  {'(全相同，噪声底噪=32)' if same else f'(在位置 {noise_fdt} 出现分歧，用作噪声底噪)'}")
            print(f"                ids32(前10) = {prefix_ids[:10]}")

        if pic_fdt is not None:
            if noise_fdt is not None:
                # PASS：FDT >= 噪声底噪（规格文档 §5.2 判定）
                pass_fail = "PASS ✓" if pic_fdt >= noise_fdt else "FAIL ✗"
                print(f"  pic vs baseline:          FDT={pic_fdt}  "
                      f"噪声底噪={noise_fdt}  [{pass_fail}]")
            else:
                print(f"  pic vs baseline:          FDT={pic_fdt}")
            print(f"                ids32(前10) = {pic_ids[:10]}")

        # a3 / cacheblend / pic_a3 / pic_cacheblend / pic_a3_oracle FDT
        for _extra_mode in ("a3", "cacheblend", "pic_a3",
                            "pic_cacheblend", "pic_a3_oracle"):
            _extra_ids = results.get(_extra_mode, {}).get("ids32", [])
            if _extra_ids:
                _extra_fdt = first_divergence_token(baseline_ids, _extra_ids)
                _label = f"{_extra_mode} vs baseline:"
                if noise_fdt is not None:
                    _pf = "PASS ✓" if _extra_fdt >= noise_fdt else "FAIL ✗"
                    print(f"  {_label:<28} FDT={_extra_fdt}  "
                          f"噪声底噪={noise_fdt}  [{_pf}]")
                else:
                    print(f"  {_label:<28} FDT={_extra_fdt}")
                print(f"                ids32(前10) = {_extra_ids[:10]}")
    else:
        print("  [跳过] full_recompute 未运行或无 ids32，无法计算 FDT")

    # ── TTFT 加速比 ────────────────────────────────────────────────────────
    baseline_ttft = results.get("full_recompute", {}).get("ttft")
    if baseline_ttft:
        print(f"\n  TTFT 加速比（相对 full_recompute = {baseline_ttft:.3f}s）:")
        for mode in modes:
            if mode == "full_recompute":
                continue
            r = results.get(mode, {})
            t = r.get("ttft")
            if t:
                speedup = baseline_ttft / t if t > 0 else float("inf")
                print(f"    {mode:<20}: {speedup:.2f}x  ({t:.3f}s)")

    # ── 生成文本 ────────────────────────────────────────────────────────────
    # 打印每个 mode 的 max_new_tokens=32 完整解码文本(含 <think>...</think>
    # reasoning,若有)。ids32 只能看出前 10 个 token 是否重复;文本能一眼分辨
    # "崩到重复""逻辑跑偏""看起来正常"三种失败/成功模式。
    # 转义换行,截断到 400 char/行 * 6 行,再截 60 行防日志爆炸。
    def _preview_text(t: str, max_lines: int = 8, max_line: int = 400) -> str:
        if not t:
            return "(empty)"
        # 保留 \n 语义但用可见转义,避免多行 print 破坏对齐
        lines = t.split("\n")
        out_lines: List[str] = []
        for ln in lines[:max_lines]:
            if len(ln) > max_line:
                ln = ln[:max_line] + f"…(+{len(ln) - max_line} char)"
            out_lines.append(ln)
        if len(lines) > max_lines:
            out_lines.append(f"…(+{len(lines) - max_lines} more lines)")
        return "\n".join(out_lines)

    print(f"\n  生成文本（max_new_tokens=32，含 <think> reasoning 若有）:")
    for mode in modes:
        r = results.get(mode, {})
        if "error" in r:
            print(f"\n  ── {mode} ── ERROR: {r['error']}")
            continue
        _text = r.get("text32", "") or ""
        _ids_head = r.get("ids32", [])[:10]
        print(f"\n  ── {mode} ──  (ids[:10]={_ids_head}, text_len={len(_text)})")
        # 缩进 4 空格,每行前置 "    | " 让文本块与其它行有明显边界
        _preview = _preview_text(_text)
        for _ln in _preview.split("\n"):
            print(f"    | {_ln}")

    print(f"\n{'='*64}")
    print("  测试完成！")
    print(f"{'='*64}\n")

    if args.plot_output:
        _render_summary_plot(results, modes, args.plot_output)


if __name__ == "__main__":
    main()
