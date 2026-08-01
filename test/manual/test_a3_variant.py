#!/usr/bin/env python3
"""test_a3_variant.py — A³ / CacheBlend 真实价值测试(prompt 变体)

与 quick_test_online.py 的 sanity 测试(precompute 用的 prompt 与请求 prompt
完全一致)不同,本脚本测的是 A³ 设计初衷:

    precompute KV 是从 prompt A 算出来的,但请求发送的是 prompt B
    (B 与 A 只差中间一段)。A³ 依靠 Q-K attention 分数挑出"重要位置"
    重新计算 K/V,让最终输出接近 baseline(prompt B 全量计算)的结果。

流程:
    1. 构造 prompt A / prompt B —— 两者仅 C2 中的动物名不同(dogs vs fish),
       其他 SYS/C1/C3/Q 完全一致,长度也强制相同(单 token 词替换,借用
       PIC 段对齐机制保证 tokenized 后位置精确对齐)。
    2. (前置)已用 prompt A 跑 scripts/precompute_kv_mla.py 生成 kv_A。
    3. 起 --enable-a3 --disable-radix-cache 服务器,发三种请求(每次 32 tok 采样):
         (a) full_recompute prompt B  →  baseline_ids_B(真答:fish)
         (b) full_recompute prompt A  →  baseline_ids_A(真答:dogs;做参照)
         (c) A³ prompt B + kv_A       →  a3_ids
    4. 对比:
         - FDT(baseline_ids_B, a3_ids)  ↑越高越好: A³ 恢复了 B 的语义
         - FDT(baseline_ids_A, a3_ids)  ↓越低越好: A³ 没有单纯"复读"A 的输出
       诊断打印:三个模式的前 32 个 token,人肉看输出中"dogs"/"fish"出现情况

前置准备(必须先跑):

    # 1. 从本脚本导出 prompt A 到文件
    python test/manual/test_a3_variant.py --gen-prompt-a /tmp/prompt_A.txt

    # 2. 用 prompt A 生成 kv
    python scripts/precompute_kv_mla.py \\
        --model /workspace/models/GLM-5.2-FP8 \\
        --prompt-file /tmp/prompt_A.txt \\
        --output-dir /tmp/precomputed_kv_variantA/ \\
        --tp 8

    # 3. 跑本测试
    python test/manual/test_a3_variant.py --tp 8 \\
        --a3-precomputed-kv /tmp/precomputed_kv_variantA/rank{rank}.pt

约束(与 A³/CacheBlend baseline 一致):
    --enable-a3 --disable-cuda-graph --chunked-prefill-size -1
    --disable-overlap-schedule  自动打开
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Dict, List, Optional

import requests

# ── 与 quick_test_online.py 保持一致的常量 ──
MODEL = os.environ.get("PIC_MODEL", "/workspace/models/GLM-5.2-FP8")
PORT = int(os.environ.get("PIC_PORT", "30001"))
READY_TIMEOUT = int(os.environ.get("PIC_READY_TIMEOUT", "1500"))
SGLANG_PY = os.environ.get("SGLANG_PY", sys.executable)
MEM_FRAC = os.environ.get("MEM_FRAC", "0.82")
CUDA_GRAPH_MAX_BS = os.environ.get("CUDA_GRAPH_MAX_BS", "4")
FAST_WARMUP = os.environ.get("PIC_FAST_WARMUP", "1")
PIC_ALIGN = int(os.environ.get("PIC_ALIGN", "64"))

# ── prompt 构造 ──
# 关键设计:
#   - SYS / C1 / C3 / Q 完全一致(hit 位置 KV 与真实值 byte-一致 → 完美 reuse)
#   - 仅 C2 中的动物名不同,且用单 token 词替换以保持 tokenized 长度一致
#   - 长度对齐后每段 padding 到 64 的倍数(与 quick_test_online 保持一致)
SYS = "You are a helpful assistant."
C1 = "Document A about cats. " * 800
C2_A_TEMPLATE = "Document B about {animal}. " * 800
C3 = "Document C about birds. " * 800
Q = "Question: which animal is in document B?"

# 动物名对(选单 token 词,减少 tokenized 长度差异的概率)
# GLM-5.2 tokenizer 通常把 "dogs"/"fish"/"cats" 都编成单 token
ANIMAL_A = "dogs"     # prompt A: baseline 答 dogs
ANIMAL_B = "fish"     # prompt B: baseline 答 fish


# ─────────────────────────────────────────────────────────────────
# 服务器生命周期(与 quick_test_online 复用一样的模式)
# ─────────────────────────────────────────────────────────────────

def _wait_port_free(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect(("127.0.0.1", port))
            except (ConnectionRefusedError, OSError):
                return
        time.sleep(1.0)


def _wait_server_ready(port: int, proc: subprocess.Popen, timeout: float) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"服务器退出,returncode={proc.returncode}")
        try:
            if requests.get(url, timeout=2.0).status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(3.0)
    raise TimeoutError(f"服务器在 {timeout}s 内未就绪(端口 {port})")


def _shutdown(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1.0)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _start_server(port: int, model: str, tp: int, extra_args: List[str]) -> subprocess.Popen:
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
    cu13_lib = "/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
    ld_path = f"/usr/local/cuda-13.0/compat:{cu13_lib}"
    env["LD_LIBRARY_PATH"] = f"{ld_path}:{env.get('LD_LIBRARY_PATH', '')}"
    env["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = FAST_WARMUP
    log_path = f"/tmp/sglang_a3_variant_{port}.log"
    print(f"  [启动] port={port}  日志={log_path}")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=log_file, start_new_session=True
    )
    proc._log_file = log_file  # type: ignore[attr-defined]
    proc._log_path = log_path  # type: ignore[attr-defined]
    return proc


# ─────────────────────────────────────────────────────────────────
# 请求发送 + 输出提取
# ─────────────────────────────────────────────────────────────────

def _post_generate(
    port: int, text: str, max_new_tokens: int,
    extra_payload: Optional[Dict] = None,
) -> Dict:
    payload = {
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if extra_payload:
        payload.update(extra_payload)
    resp = requests.post(f"http://127.0.0.1:{port}/generate",
                          json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def _extract_ids32(out: Dict) -> List[int]:
    meta = out.get("meta_info", {})
    ids = meta.get("output_token_ids", []) or out.get("token_ids", []) or out.get("output_ids", [])
    return ids


def first_divergence_token(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# ─────────────────────────────────────────────────────────────────
# prompt 构造(含 padding 对齐)
# ─────────────────────────────────────────────────────────────────

def _pad_to_multiple(text: str, tokenizer, multiple: int = PIC_ALIGN):
    """Padding 到 multiple 的倍数,与 quick_test_online.py 使用相同 tokenizer 保持一致。"""
    ids = tokenizer.encode(text, add_special_tokens=False)
    orig = len(ids)
    if orig % multiple == 0:
        return text, orig, orig
    target = ((orig // multiple) + 1) * multiple
    for pad_char in [" ", "\n", ".", "!", "a", "0"]:
        t = text
        cur = orig
        guard = 0
        while cur < target and guard < multiple * 8:
            t = t + pad_char
            cur = len(tokenizer.encode(t, add_special_tokens=False))
            guard += 1
        if cur == target:
            return t, orig, cur
    return t, orig, cur


def build_prompts(model: str, align: bool = True):
    """构造 (prompt_A, prompt_B) —— 仅 C2 不同,其他完全一致。"""
    c2_A = C2_A_TEMPLATE.format(animal=ANIMAL_A)
    c2_B = C2_A_TEMPLATE.format(animal=ANIMAL_B)

    if align:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        # 各段单独对齐;C2_A 与 C2_B 独立对齐,tokenized 长度应严格一致
        # (dogs/fish 都是单 token,差异不影响)
        sys_seg, _, _ = _pad_to_multiple(SYS, tok)
        c1_seg, _, _ = _pad_to_multiple(C1, tok)
        c2A_seg, _, _ = _pad_to_multiple(c2_A, tok)
        c2B_seg, _, _ = _pad_to_multiple(c2_B, tok)
        c3_seg, _, _ = _pad_to_multiple(C3, tok)
        q_seg, _, _ = _pad_to_multiple(Q, tok)

        # 验证 c2A_seg 与 c2B_seg tokenized 长度一致(A³ 需要 kv 位置对齐)
        lenA = len(tok.encode(c2A_seg, add_special_tokens=False))
        lenB = len(tok.encode(c2B_seg, add_special_tokens=False))
        if lenA != lenB:
            print(f"  [警告] C2 token 长度不一致: A={lenA} vs B={lenB}. "
                  f"A³ reuse 可能行为异常。考虑替换动物名。", file=sys.stderr)
    else:
        sys_seg, c1_seg, c2A_seg, c2B_seg, c3_seg, q_seg = SYS, C1, c2_A, c2_B, C3, Q

    prompt_A = f"{sys_seg}{c1_seg}{c2A_seg}{c3_seg}{q_seg}"
    prompt_B = f"{sys_seg}{c1_seg}{c2B_seg}{c3_seg}{q_seg}"
    return prompt_A, prompt_B


# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="A³ variant test")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--a3-recomp-ratio", type=float, default=0.15)
    parser.add_argument(
        "--a3-precomputed-kv",
        type=str,
        default="/tmp/precomputed_kv_variantA/rank{rank}.pt",
        help="prompt A 生成的 KV .pt 路径模板,{rank} 会被 per-TP 替换",
    )
    parser.add_argument(
        "--gen-prompt-a", type=str, default=None, metavar="PATH",
        help="导出 prompt A 到文件后退出,用于喂给 precompute_kv_mla.py",
    )
    parser.add_argument(
        "--no-align", action="store_true",
        help="跳过 PIC-64 对齐(测试用,一般不用)",
    )
    args = parser.parse_args()

    print(f"\n{'#'*64}")
    print(f"  A³ Variant Test — 真实价值测试")
    print(f"  模型: {args.model}   TP={args.tp}   port={args.port}")
    print(f"  变体设计: C2 中的动物名 {ANIMAL_A!r} → {ANIMAL_B!r}")
    print(f"{'#'*64}")

    print(f"\n  [1/x] 构造 prompt A/B (align={not args.no_align})...")
    prompt_A, prompt_B = build_prompts(args.model, align=not args.no_align)
    print(f"    prompt_A: {len(prompt_A):,} 字符")
    print(f"    prompt_B: {len(prompt_B):,} 字符")
    if len(prompt_A) != len(prompt_B):
        print(f"    [注意] 字符长度不同(可能 tokenized 长度也不同)")

    # ── 分支:仅导出 prompt A ──
    if args.gen_prompt_a:
        out = os.path.abspath(args.gen_prompt_a)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            f.write(prompt_A)
        print(f"\n  [gen-prompt-a] 写入 → {out}")
        print(f"  下一步 (用同 tp 值):")
        print(f"    python scripts/precompute_kv_mla.py \\")
        print(f"        --model {args.model} \\")
        print(f"        --prompt-file {out} \\")
        print(f"        --output-dir {os.path.dirname(args.a3_precomputed_kv) or '/tmp/precomputed_kv_variantA'} \\")
        print(f"        --tp {args.tp}")
        sys.exit(0)

    # ── 前置检查:precompute .pt 存在? ──
    _probe = args.a3_precomputed_kv.format(rank=0) if "{rank}" in args.a3_precomputed_kv else args.a3_precomputed_kv
    if not os.path.exists(_probe):
        print(f"\n  [错误] 需要先生成 prompt A 的 precomputed KV:\n"
              f"    1) python test/manual/test_a3_variant.py --gen-prompt-a /tmp/prompt_A.txt --tp {args.tp}\n"
              f"    2) python scripts/precompute_kv_mla.py --model {args.model} "
              f"--prompt-file /tmp/prompt_A.txt --output-dir "
              f"{os.path.dirname(_probe) or '/tmp'} --tp {args.tp}\n"
              f"  期望找到: {_probe}", file=sys.stderr)
        sys.exit(1)
    print(f"  [precompute] 找到 KV: {args.a3_precomputed_kv}")

    # ── 三次请求:baseline_B / baseline_A / a3_B ──
    a3_common = [
        "--enable-a3",
        "--recomp-ratio", str(args.a3_recomp_ratio),
        "--disable-cuda-graph",
        "--chunked-prefill-size", "-1",
        "--disable-overlap-schedule",
        "--disable-radix-cache",   # 关掉 prefix cache 排除干扰
    ]
    baseline_extra_args = ["--disable-radix-cache"]

    results: Dict[str, List[int]] = {}
    tests = [
        ("baseline_B", prompt_B, baseline_extra_args, None,
         "full_recompute prompt B (真答: fish)"),
        ("baseline_A", prompt_A, baseline_extra_args, None,
         "full_recompute prompt A (对比: dogs)"),
        ("a3_B",       prompt_B, a3_common, {
            "reuse_method": "debug",
            "recomp_ratio": args.a3_recomp_ratio,
            "precomputed_kv_path": args.a3_precomputed_kv,
        }, f"A³ prompt B + kv_A (recomp={args.a3_recomp_ratio})"),
    ]

    for name, prompt, extra_args, extra_payload, desc in tests:
        print(f"\n  ═══ [{name}] {desc} ═══")
        _wait_port_free(args.port)
        proc = _start_server(args.port, args.model, args.tp, extra_args)
        try:
            _wait_server_ready(args.port, proc, READY_TIMEOUT)
            print(f"  [ok] server ready")
            t0 = time.perf_counter()
            out = _post_generate(args.port, prompt, 32, extra_payload)
            print(f"  [ok] generate done in {time.perf_counter()-t0:.2f}s")
            ids = _extract_ids32(out)
            results[name] = ids
            # 打印文本(前 100 字)看语义
            text = out.get("text", "") or ""
            print(f"  文本预览: {text[:150]!r}")
            print(f"  前 10 tokens: {ids[:10]}")
            for keyword in [ANIMAL_A, ANIMAL_B]:
                if keyword in text.lower():
                    print(f"  [!] 输出包含 {keyword!r}")
        except Exception as e:
            print(f"  [失败] {e}", file=sys.stderr)
            log_path = getattr(proc, "_log_path", None)
            if log_path and os.path.exists(log_path):
                with open(log_path) as f:
                    tail = f.readlines()[-50:]
                print("".join(tail))
            results[name] = []
        finally:
            _shutdown(proc)
        _wait_port_free(args.port)

    # ── 结果分析 ──
    print(f"\n\n{'='*64}")
    print(f"  A³ Variant Test 结果分析")
    print(f"{'='*64}")

    b_B = results.get("baseline_B", [])
    b_A = results.get("baseline_A", [])
    a3  = results.get("a3_B", [])

    if not (b_B and b_A and a3):
        print("  [跳过] 有测试失败,无法完成对比")
        sys.exit(2)

    fdt_a3_vs_baselineB = first_divergence_token(b_B, a3)
    fdt_a3_vs_baselineA = first_divergence_token(b_A, a3)
    fdt_baselineA_vs_baselineB = first_divergence_token(b_A, b_B)

    print(f"\n  FDT(A vs B baseline) = {fdt_baselineA_vs_baselineB}  "
          f"(两 baseline 应该在 C2 变化后 quickly 分歧;越小说明 A/B 语义差别越大)")
    print(f"  FDT(a3 vs baseline B) = {fdt_a3_vs_baselineB}  "
          f"(A³ 越接近 B → 越大越好)")
    print(f"  FDT(a3 vs baseline A) = {fdt_a3_vs_baselineA}  "
          f"(A³ 若只是复读 A → 会等于 32)")

    print(f"\n  判断:")
    if fdt_a3_vs_baselineB > fdt_a3_vs_baselineA:
        print(f"    ✓ A³ 输出更接近 baseline_B (FDT {fdt_a3_vs_baselineB} > {fdt_a3_vs_baselineA})")
        print(f"      → A³ 成功利用 imp_indices 恢复了 B 的语义,不是简单复读 A")
    elif fdt_a3_vs_baselineB == fdt_a3_vs_baselineA:
        print(f"    ~ A³ 与两个 baseline 距离相等 (FDT={fdt_a3_vs_baselineB})")
        print(f"      → 可能 A/B 的差异位置尚未在采样的 32 tokens 内体现;试增大 max_new_tokens")
    else:
        print(f"    ✗ A³ 输出反而更接近 baseline_A (FDT {fdt_a3_vs_baselineA} > {fdt_a3_vs_baselineB})")
        print(f"      → A³ 的 imp_indices 没挑对关键位置,或 recomp_ratio={args.a3_recomp_ratio} 太小")

    print(f"\n  ids 详情:")
    print(f"    baseline_A(dogs)  ids[:10] = {b_A[:10]}")
    print(f"    baseline_B(fish)  ids[:10] = {b_B[:10]}")
    print(f"    a3_B (kv_A + B)   ids[:10] = {a3[:10]}")

    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()
