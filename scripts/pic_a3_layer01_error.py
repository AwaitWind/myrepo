#!/usr/bin/env python3
"""pic_a3_layer01_error.py — 测量 pic_a3 vs full_recompute 在 layer 0 / layer 1
的 K 缓存误差 (按段分组打印 mean/max/p50/p95/p99)。

原理:
    复用 model_runner.py::_pic_writeback_mla_kv 里的 SGLANG_PIC_POOL_DUMP_DIR
    钩子 (在 forward + writeback 结束后, 按 req_to_token 把每层 K 落到
    rank{r}_layer{l}_latent.pt)。

    对同一 test prompt (SYS+C1+C2+C3+Q) 分别以两种模式跑一次 Engine:
      1. full_recompute — baseline, 所有位置 K 都是本次 forward 现算
      2. pic_a3         — warmup 后 SYS/C1/C3 段命中; hit-非-imp 位置的 K
                          最终来自 PICache public slot (warmup 存的 K),
                          miss + imp 位置的 K 是本次 forward 现算

    对 layer 0 和 layer 1 分别做逐 token L2 误差:
        err[i] = ||K_full[layer, i] - K_a3[layer, i]||_2

    按段 (SYS/C1/C2/C3/Q) 分组汇总, 快速判断:
      * 命中段是否有 delta-RoPE / numerical drift
      * miss 段是否近乎 0 (fresh-vs-fresh 差, 应该 ~0)
      * layer 0 vs layer 1 的误差量级差 (layer 1 累积了 layer 0 attention 的差)

用法:
    # 完整跑一次 (两个 Engine 各起一次, ~5-10 分钟)
    python scripts/pic_a3_layer01_error.py \\
        --model /workspace/models/GLM-5.2-FP8 --tp 8

    # 复用已有 dump (秒级):
    python scripts/pic_a3_layer01_error.py --skip-capture

    # 出 CSV + 曲线图:
    python scripts/pic_a3_layer01_error.py --output-csv err.csv --output-plot err.png

跑前需要:
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTHONPATH=/root/sglang/python:$PYTHONPATH
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

# ── 与 quick_test_online.py / pic_mode_kv_error_diag.py 对齐的常量 ──
SEP = "<<PIC_SEP>>"
SYS_TXT = "You are a helpful assistant."
C1_TXT = "Document A about cats. " * 800
C2_TXT = "Document B about dogs. " * 800
C3_TXT = "Document C about birds. " * 800
Q_TXT = "Question: which animal is in document B?"
PIC_ALIGN = int(os.environ.get("PIC_ALIGN", "64"))

_SHARD_RE = re.compile(r"rank(?P<rank>\d+)_layer(?P<layer>\d+)_latent\.pt")


# ──────────────────────────────────────────────────────────────────────────
# Padding (与 quick_test_online.py 相同)
# ──────────────────────────────────────────────────────────────────────────

def _pad_to_multiple(text: str, tokenizer, multiple: int = PIC_ALIGN) -> Tuple[str, int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    orig = len(ids)
    if orig % multiple == 0:
        return text, orig
    target = ((orig // multiple) + 1) * multiple
    for pad_char in [" ", "\n", ".", "!", "a", "0"]:
        t, cur, guard = text, orig, 0
        while cur < target and guard < multiple * 8:
            t = t + pad_char
            cur = len(tokenizer.encode(t, add_special_tokens=False))
            guard += 1
        if cur == target:
            return t, cur
    return t, cur


def build_prompts(model: str) -> Dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    print(f"  [pad] tokenize + 段对齐到 {PIC_ALIGN}...")
    sys_seg, sys_len = _pad_to_multiple(SYS_TXT, tok)
    c1_seg, c1_len = _pad_to_multiple(C1_TXT, tok)
    c2_seg, c2_len = _pad_to_multiple(C2_TXT, tok)
    c3_seg, c3_len = _pad_to_multiple(C3_TXT, tok)
    q_seg, q_len = _pad_to_multiple(Q_TXT, tok)
    total = sys_len + c1_len + c2_len + c3_len + q_len
    print(f"    SYS={sys_len} C1={c1_len} C2={c2_len} C3={c3_len} "
          f"Q={q_len} total={total}")

    seg_ranges: Dict[str, Tuple[int, int]] = {
        "SYS": (0, sys_len),
        "C1":  (sys_len, sys_len + c1_len),
        "C2":  (sys_len + c1_len, sys_len + c1_len + c2_len),
        "C3":  (sys_len + c1_len + c2_len, sys_len + c1_len + c2_len + c3_len),
        "Q":   (total - q_len, total),
    }
    # 命中标记 (与 quick_test_online.py 的 warmup 一致: pic_w1=SYS+C1, pic_w3=SYS+C3)
    seg_hit_status: Dict[str, str] = {
        "SYS": "hit",      # 两次 warmup 都覆盖
        "C1":  "hit",      # 出现在 pic_w1
        "C2":  "miss",     # 从未 warmup
        "C3":  "hit",      # 出现在 pic_w3
        "Q":   "miss",     # PIC 不缓存最后一段
    }

    return {
        "full_test": f"{sys_seg}{c1_seg}{c2_seg}{c3_seg}{q_seg}",
        "pic_test":  f"{sys_seg}{SEP}{c1_seg}{SEP}{c2_seg}{SEP}{c3_seg}{SEP}{q_seg}",
        "pic_w1":    f"{sys_seg}{SEP}{c1_seg}{SEP}{q_seg}",
        "pic_w3":    f"{sys_seg}{SEP}{c3_seg}{SEP}{q_seg}",
        "seg_ranges": seg_ranges,
        "seg_hit_status": seg_hit_status,
        "total": total,
    }


# ──────────────────────────────────────────────────────────────────────────
# Engine 启动 + KV dump
#
# 采用 subprocess + HTTP 而非 sglang.Engine in-process:
#   sglang.Engine 在 in-process 起 TP worker 时会命中 CVD race
#   ("TPn: CUDA error: invalid device ordinal"), 复现于 quick_test_online.py。
#   subprocess 路径下 env 在 Popen 前 snapshot 好, TP worker execve 时环境
#   一致, 该 race 不会发生, 与 quick_test_online.py 的成功姿势对齐。
# ──────────────────────────────────────────────────────────────────────────

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
    print(f"  [warn] 等待端口 {port} 释放超时 ({timeout}s), 继续……")


def _wait_server_ready(port: int, proc: subprocess.Popen,
                       timeout: float = 1500.0) -> None:
    import requests  # 惰性导入
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server 进程提前退出, returncode={proc.returncode}"
            )
        try:
            r = requests.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(3.0)
    raise TimeoutError(f"server 在 {timeout}s 内未就绪 (端口 {port})")


def _shutdown(proc: subprocess.Popen,
              timeout_term: float = 30.0, timeout_kill: float = 10.0) -> None:
    if proc.poll() is not None:
        _lf = getattr(proc, "_log_file", None)
        if _lf:
            try: _lf.close()
            except Exception: pass
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    else:
        deadline = time.time() + timeout_term
        while time.time() < deadline and proc.poll() is None:
            time.sleep(1.0)
        if proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.time() + timeout_kill
            while time.time() < deadline and proc.poll() is None:
                time.sleep(0.5)
    _lf = getattr(proc, "_log_file", None)
    if _lf:
        try: _lf.close()
        except Exception: pass


def _post_generate(port: int, text: str, max_new_tokens: int,
                   timeout: float = 600.0) -> Dict:
    import requests
    url = f"http://127.0.0.1:{port}/generate"
    payload = {
        "text": text,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _mode_launch_args(mode: str) -> List[str]:
    """与 quick_test_online.py 完全一致的 flags。"""
    common = [
        "--reasoning-parser", "glm45",
        "--tool-call-parser", "glm47",
        "--disable-piecewise-cuda-graph",
        "--log-level", "warning",
    ]
    if mode == "full_recompute":
        return common + [
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--chunked-prefill-size", "-1",
            "--disable-overlap-schedule",
        ]
    if mode == "pic_a3":
        return common + [
            "--pic-enable",
            "--page-size", "64",
            "--chunked-prefill-size", "-1",
            "--pic-separator-str", SEP,
            "--enable-a3",
            "--recomp-ratio", "0.15",
            "--disable-cuda-graph",
            "--disable-overlap-schedule",
            "--disable-radix-cache",
        ]
    raise ValueError(f"unknown mode: {mode}")


def _mode_prompts(mode: str, prompts: Dict) -> Tuple[str, List[str]]:
    if mode == "full_recompute":
        return prompts["full_test"], []
    if mode == "pic_a3":
        return prompts["pic_test"], [prompts["pic_w1"], prompts["pic_w3"]]
    raise ValueError(f"unknown mode: {mode}")


def _start_server(mode: str, model: str, tp: int, port: int,
                  mem_frac: float, dump_dir: str) -> subprocess.Popen:
    """启一个 sglang.launch_server 子进程 (与 quick_test_online.py 对齐)。"""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--tp", str(tp),
        "--trust-remote-code",
        "--mem-fraction-static", str(mem_frac),
        "--cuda-graph-max-bs", "4",
        "--host", "0.0.0.0",
        "--port", str(port),
        *_mode_launch_args(mode),
    ]

    env = os.environ.copy()
    # CUDA 13 兼容库 (与 quick_test_online.py 一致)
    cu13 = "/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
    ld = f"/usr/local/cuda-13.0/compat:{cu13}"
    env["LD_LIBRARY_PATH"] = (
        f"{ld}:{env.get('LD_LIBRARY_PATH', '')}"
        if env.get("LD_LIBRARY_PATH") else ld
    )
    # DeepGEMM fast warmup (加速启动)
    env.setdefault("SGLANG_JIT_DEEPGEMM_FAST_WARMUP", "1")
    env.setdefault("SGLANG_JIT_DEEPGEMM_FAST_WARMUP_STRIDE_MULT", "4")
    # PIC 系模式需要的额外 env
    if mode == "pic_a3":
        env["SGLANG_EAGER_INPUT_NO_COPY"] = "1"
        env["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"
    # 关键: 把 dump dir 注入 subprocess env — 只有这样 TP worker 才能读到
    env["SGLANG_PIC_POOL_DUMP_DIR"] = dump_dir
    # ── 强制覆盖 CVD (不用 setdefault, 因为 os.environ 可能被
    # 调用方 shell 预先污染成 CVD='0' 单卡, setdefault 无法覆盖污染值)。
    # 用户想要保留自己的设置就在环境里显式设 SGLANG_L01_ERR_KEEP_ENV=1。
    if os.environ.get("SGLANG_L01_ERR_KEEP_ENV") != "1":
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tp))
    # 打印实际要传给 subprocess 的 CVD, 方便定位 env 传递问题
    print(f"  [env] passing CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')!r}")

    log_path = f"/tmp/pic_a3_l01_err_{mode}.log"
    print(f"  [launch] {' '.join(cmd[2:6])} ... port={port}  log={log_path}")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=log_file, start_new_session=True,
    )
    proc._log_file = log_file  # type: ignore[attr-defined]
    proc._log_path = log_path  # type: ignore[attr-defined]
    return proc


def run_capture(
    mode: str, model: str, tp: int, prompts: Dict,
    dump_dir: str, mem_frac: float, port: int,
) -> None:
    """启一次 sglang server (subprocess), warmup + test 请求, 每层 K dump 到 dump_dir。"""
    os.makedirs(dump_dir, exist_ok=True)
    for f in os.listdir(dump_dir):
        if _SHARD_RE.match(f):
            os.remove(os.path.join(dump_dir, f))

    test_prompt, warmup_prompts = _mode_prompts(mode, prompts)

    print(f"\n{'─'*64}")
    print(f"  mode={mode}  tp={tp}  port={port}  dump_dir={dump_dir}")
    print(f"{'─'*64}")

    _wait_port_free(port)
    proc = _start_server(mode, model, tp, port, mem_frac, dump_dir)

    try:
        print(f"  [wait-ready] 最多 1500s ...")
        _wait_server_ready(port, proc)
        print(f"  [ready]")

        for i, wp in enumerate(warmup_prompts, 1):
            print(f"  [warmup {i}/{len(warmup_prompts)}] ({len(wp)} chars)")
            _post_generate(port, wp, max_new_tokens=4)

        print(f"  [test] {len(test_prompt)} chars, max_new_tokens=1")
        t0 = time.perf_counter()
        _post_generate(port, test_prompt, max_new_tokens=1)
        print(f"  [test] done in {time.perf_counter()-t0:.2f}s")
    except Exception:
        # 打出 server 日志尾, 方便定位
        _lp = getattr(proc, "_log_path", None)
        if _lp:
            try:
                _lf = getattr(proc, "_log_file", None)
                if _lf: _lf.flush()
                with open(_lp) as f:
                    lines = f.readlines()
                tail = lines[-60:] if len(lines) > 60 else lines
                print(f"\n  ── server log tail ({_lp}) ──")
                print("".join(tail))
                print(f"  ── end log ──\n")
            except Exception as _le:
                print(f"  [warn] 读日志失败: {_le}")
        raise
    finally:
        print(f"  [shutdown]")
        _shutdown(proc)
        _wait_port_free(port)

    time.sleep(1.5)
    gc.collect()
    torch.cuda.empty_cache()

    shards = [f for f in os.listdir(dump_dir) if _SHARD_RE.match(f)]
    if not shards:
        raise RuntimeError(
            f"[{mode}] server 跑完但 {dump_dir} 里没有 dump 文件。\n"
            f"        可能原因: SGLANG_PIC_POOL_DUMP_DIR 没传到 TP worker,\n"
            f"        或 _pic_writeback_mla_kv 的钩子被异常吞了 (见 {getattr(proc, '_log_path', '?')})。"
        )
    print(f"  [dump-check] {len(shards)} 个 .pt")


# ──────────────────────────────────────────────────────────────────────────
# 读取 + 对比
# ──────────────────────────────────────────────────────────────────────────

def load_layer(dump_dir: str, layer_id: int, expected_T: int) -> torch.Tensor:
    """加载 rank0 layer_id 的 K, 返回 (T, D) float32 tensor。

    MLA MQA (num_kv_heads=1) → KV 在 rank 间是复制, 只读 rank0 即可。
    dump 出来一般是 (T, 1, D), 这里 squeeze 到 (T, D)。
    """
    path = os.path.join(dump_dir, f"rank0_layer{layer_id}_latent.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing dump: {path}")
    t = torch.load(path, map_location="cpu")
    while t.ndim > 2:
        squeezed = False
        for d in range(t.ndim):
            if t.shape[d] == 1:
                t = t.squeeze(d)
                squeezed = True
                break
        if not squeezed:
            raise ValueError(
                f"cannot squeeze layer {layer_id} tensor to 2D: shape={tuple(t.shape)}"
            )
    if t.shape[0] != expected_T:
        raise ValueError(
            f"layer {layer_id}: dump has T={t.shape[0]}, expected {expected_T}"
        )
    return t.float()


def compute_error(baseline_k: torch.Tensor, other_k: torch.Tensor) -> torch.Tensor:
    """|| baseline - other ||_2 per-token, shape (T,)"""
    return (baseline_k - other_k).norm(dim=-1)


def _summarize(err: torch.Tensor, name: str) -> Dict[str, float]:
    a = err.numpy()
    if a.size == 0:
        print(f"    {name}: <empty>")
        return {}
    stats = {
        "n": int(a.size),
        "mean": float(a.mean()),
        "max":  float(a.max()),
        "min":  float(a.min()),
        "p50":  float(np.percentile(a, 50)),
        "p95":  float(np.percentile(a, 95)),
        "p99":  float(np.percentile(a, 99)),
        "argmax": int(a.argmax()),
    }
    print(
        f"    {name:>16s}  n={stats['n']:<5d}  "
        f"mean={stats['mean']:.4e}  max={stats['max']:.4e}  "
        f"p50={stats['p50']:.4e}  p95={stats['p95']:.4e}  "
        f"p99={stats['p99']:.4e}  argmax@{stats['argmax']}"
    )
    return stats


def analyze_layer(
    layer_id: int, err: torch.Tensor,
    seg_ranges: Dict[str, Tuple[int, int]],
    seg_hit_status: Dict[str, str],
) -> Dict:
    print(f"\n  ══ LAYER {layer_id} ══")
    all_stats = {"overall": _summarize(err, "overall")}
    for name, (s, e) in seg_ranges.items():
        tag = seg_hit_status.get(name, "?")
        label = f"{name}[{s}:{e}]({tag})"
        all_stats[name] = _summarize(err[s:e], label)
    return all_stats


# ──────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="pic_a3 vs full_recompute — 逐层 K 缓存 L2 误差诊断 (默认 layer 0/1)"
    )
    p.add_argument("--model", default="/workspace/models/GLM-5.2-FP8",
                   help="模型路径")
    p.add_argument("--tp", type=int, default=8, help="tensor parallel")
    p.add_argument("--mem-fraction", type=float, default=0.82,
                   help="mem_fraction_static")
    p.add_argument("--dump-root", default="/tmp/pic_a3_l01_err",
                   help="每模式的 dump 目录前缀 (完整为 <root>_<mode>)")
    p.add_argument("--port", type=int, default=30001,
                   help="sglang server 端口 (默认 30001)")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 1],
                   help="要对比的 layer id 列表 (默认 0 1)")
    p.add_argument("--skip-capture", action="store_true",
                   help="跳过 Engine 启动, 复用已有 dump 目录")
    p.add_argument("--output-csv", default=None,
                   help="导出逐 token 原始 L2 误差 CSV (layer,token,err)")
    p.add_argument("--output-plot", default=None,
                   help="导出 layer×token 误差曲线 PNG")
    args = p.parse_args()

    prompts = build_prompts(args.model)
    T = prompts["total"]
    seg_ranges = prompts["seg_ranges"]
    seg_hit_status = prompts["seg_hit_status"]

    dump_full = f"{args.dump_root}_full_recompute"
    dump_a3 = f"{args.dump_root}_pic_a3"

    if not args.skip_capture:
        run_capture("full_recompute", args.model, args.tp, prompts,
                    dump_full, args.mem_fraction, args.port)
        run_capture("pic_a3", args.model, args.tp, prompts,
                    dump_a3, args.mem_fraction, args.port)
    else:
        print(f"  [skip-capture] 复用 {dump_full} / {dump_a3}")

    print(f"\n{'='*72}")
    print(f"  pic_a3 vs full_recompute — layer K 误差 (dump 时机: forward + writeback 之后)")
    print(f"  T = {T} tokens")
    print(f"  段布局: " + "  ".join(
        f"{n}={s}..{e}({seg_hit_status[n]})" for n, (s, e) in seg_ranges.items()
    ))
    print(f"{'='*72}")
    print(
        "  说明: pic_a3 在 hit-非-imp 位置的 K 最终来自 PICache public slot\n"
        "       (warmup 存的 K), 与 full_recompute 现算 K 的差反映的是\n"
        "       'PICache 里的历史 K vs 当前上下文语境的 fresh K' 的语境错位。\n"
        "       miss 段 (C2, Q) 的误差应接近 0 (两边都是 fresh 现算)。"
    )

    all_errors: Dict[int, torch.Tensor] = {}
    all_stats: Dict[int, Dict] = {}
    for layer_id in args.layers:
        k_full = load_layer(dump_full, layer_id, T)
        k_a3 = load_layer(dump_a3, layer_id, T)
        # 顺带打一下 baseline / a3 各自的 K L2 幅值, 让读者能把 err 量级放到
        # 上下文里评估 (相对误差)
        print(f"\n  [layer {layer_id}] |K_full|_2 mean = "
              f"{k_full.norm(dim=-1).mean().item():.4e}  "
              f"|K_a3|_2 mean = {k_a3.norm(dim=-1).mean().item():.4e}")
        err = compute_error(k_full, k_a3)
        all_errors[layer_id] = err
        all_stats[layer_id] = analyze_layer(
            layer_id, err, seg_ranges, seg_hit_status,
        )

    # ── CSV 导出 ──
    if args.output_csv:
        with open(args.output_csv, "w") as f:
            f.write("layer,token,err_l2,segment,hit\n")
            for layer_id, err in all_errors.items():
                arr = err.tolist()
                for i, v in enumerate(arr):
                    seg = "?"
                    for n, (s, e) in seg_ranges.items():
                        if s <= i < e:
                            seg = n
                            break
                    f.write(f"{layer_id},{i},{v},{seg},{seg_hit_status.get(seg,'?')}\n")
        print(f"\n  [csv] {args.output_csv}")

    # ── Plot 导出 ──
    if args.output_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [plot] matplotlib 未安装, 跳过绘图 (pip install matplotlib)")
        else:
            n = len(args.layers)
            fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
            if n == 1:
                axes = [axes]
            for ax, layer_id in zip(axes, args.layers):
                arr = all_errors[layer_id].numpy()
                ax.plot(arr, linewidth=0.4, color="tab:blue")
                ax.set_yscale("log")
                ax.set_ylabel(f"layer {layer_id}\nK L2 err")
                # 段带 + 段标签
                _ymax = ax.get_ylim()[1]
                for name, (s, e) in seg_ranges.items():
                    color = {"hit": "tab:green", "miss": "tab:red"}.get(
                        seg_hit_status[name], "gray"
                    )
                    ax.axvspan(s, e, alpha=0.08, color=color)
                    ax.text(
                        (s + e) / 2, _ymax * 0.4,
                        f"{name}\n({seg_hit_status[name]})",
                        ha="center", va="center", fontsize=8,
                        color=color, alpha=0.9,
                    )
            axes[-1].set_xlabel("token position (0..T)")
            fig.suptitle("pic_a3 vs full_recompute — per-token K L2 error at layer 0/1")
            plt.tight_layout()
            plt.savefig(args.output_plot, dpi=150)
            print(f"  [plot] {args.output_plot}")

    print(f"\n{'='*72}")
    print(f"  DONE.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
