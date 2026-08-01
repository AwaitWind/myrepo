#!/usr/bin/env python3
"""pic_a3_imp_token_view.py — 可视化 pic_a3 / pic_cacheblend 挑了哪些 token 去重算

问题:
    在 quick_test_online.py 的 3 文档场景里 (SYS+C1(cats)+C2(dogs)+C3(birds)+Q),
    pic_a3 / pic_cacheblend 有时输出乱码/文档串味。眼看输出无法判断是"重要 token
    选错了"还是"选对了但融合逻辑错了"。这个脚本回答第一个问题:
        pic_a3 / pic_cacheblend 到底把哪些位置标记为 imp 送去重算?

原理:
    model_runner._pic_a3_pick_imp 在决策完 imp_indices 后打印:
        [PIC-A3-IMP-LIST] rid=<8char> reuse=<method> N=<full_len> \
            last_len=<L> prefix_len=<P> n_imp=<K> positions=<comma-ints>
    本脚本:
      1. 起一次 sglang 服务器, 走跟 quick_test_online.py 完全一致的 warmup + test
         请求 (段 pad_to_64 后 SYS+C1+C2+C3+Q, warmup 命中 C1 / C3)。
      2. 从服务器日志里 regex 出 [PIC-A3-IMP-LIST] 行, 解析 positions 列表。
      3. 用同一份 tokenizer 把每段边界算出来, 把位置归到 SYS / C1 / C2 / C3 / Q 五段。
      4. 每段打印:
           - 命中数量与比例 (imp_len_in_seg / seg_len)
           - 抽样若干位置(前几 + 末几 + 若干中段), 解码周围 5 个 token 的文本片段,
             让人一眼看出算法选到的到底是 "Document A about cats" 还是 "dogs" 还是别的。

    只解析日志 (跟 quick_test_online 的 _parse_pic_hits 一样), 不落盘中间产物, 不改
    源码之外的一行。

用法:
    python test/manual/pic_a3_imp_token_view.py \\
        --model /workspace/models/GLM-5.2-FP8 --tp 8 \\
        --modes pic_a3 pic_cacheblend

    # 只跑 pic_a3
    python test/manual/pic_a3_imp_token_view.py --modes pic_a3

    # 抽样密度: 每段最多打印 30 个位置的邻域文本 (默认 15)
    python test/manual/pic_a3_imp_token_view.py --per-seg-samples 30

依赖:
    * model_runner.py::_pic_a3_pick_imp 末尾必须打印 [PIC-A3-IMP-LIST] 行
      (由本 commit 引入)。旧版本 sglang 上跑这个脚本会解析不到位置。
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests

# ─────────────────────────────────────────────────────────────────
# 与 quick_test_online.py 对齐: 段 padding + prompt 构造
# ─────────────────────────────────────────────────────────────────

SEP = "<<PIC_SEP>>"
# SYS/Q 直接构造成正好 64 tokens (0 padding) 而不是"扩展后 pad 一点"。
# 原因: 之前 Q 尾部有 1 个 pad token (57003=`!?`) 就让 full_recompute 也生成
# 全套 `.,!?` 序列 —— 模型自回归时最紧邻 context (Q 尾部) 决定 next token,
# 哪怕 1 个 pad token 也会毒害生成。彻底消除 Q 尾部 pad, 让模型看到真实问题
# 内容后再自然生成答案。
# SYS: "You are a helpful AI assistant. " × 9 = 64 tokens 精确
# Q:   "Which animal is in document B? " × 8   = 64 tokens 精确
_SYS_UNIT = "You are a helpful AI assistant. "
_Q_UNIT = "Which animal is in document B? "
_Q_TAIL = " Answer with just the animal name:"
SYS = _SYS_UNIT * 9
C1 = "Document A about cats. " * 800
C2 = "Document B about dogs. " * 800
C3 = "Document C about birds. " * 800
# Q 结尾必须有明确的答复 cue, 否则模型看完直接吐 EOS (154827) — 见诊断:
# 尾部只有 "in document B?" 时 full_recompute 输出为空 (token 154827 EOS)。
# body × 8 + " Answer with just the animal name:" = 精确 64 tokens, 0 padding。
Q = _Q_UNIT * 8 + _Q_TAIL

PIC_ALIGN = int(os.environ.get("PIC_ALIGN", "64"))
MEM_FRAC = os.environ.get("MEM_FRAC", "0.82")
READY_TIMEOUT = int(os.environ.get("PIC_READY_TIMEOUT", "1500"))
FAST_WARMUP = os.environ.get("PIC_FAST_WARMUP", "1")
FAST_WARMUP_STRIDE_MULT = os.environ.get("PIC_FAST_WARMUP_STRIDE_MULT", "4")


def _pad_text_to_multiple(text: str, tokenizer, multiple: int = PIC_ALIGN):
    """把文本 pad 到 encode 后 token 数为 multiple 的倍数; 返回 (padded, orig_len, padded_len)。

    与 quick_test_online._pad_text_to_multiple 逻辑一致 (为避免耦合, 这里复制而非 import)。

    ★ pad 策略: 优先用多字符轮换 (如 ".,!?" 每次追加一个不同 char), 避免 BPE
    tokenizer 把 N 个重复 char merge 成一个 dominant token 淹没 padding 段
    (会毁掉 layer-2+ attention, 详见本脚本 diagnosis: unique=1 → 4 是关键)。
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    orig = len(ids)
    if orig % multiple == 0:
        return text, orig, orig
    target = ((orig // multiple) + 1) * multiple
    for pad_seq in [".,!?", ".!,", ".,", " .", ".", " ", "\n", "!", "a", "0"]:
        t = text
        cur = orig
        guard = 0
        idx = 0
        while cur < target and guard < multiple * 8:
            t = t + pad_seq[idx % len(pad_seq)]
            cur = len(tokenizer.encode(t, add_special_tokens=False))
            idx += 1
            guard += 1
        if cur == target:
            return t, orig, cur
    return t, orig, cur


# ─────────────────────────────────────────────────────────────────
# 服务器生命周期 (与 quick_test_online.py 完全一致的逻辑, 精简版)
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
    print(f"  [警告] 等待端口 {port} 释放超时 ({timeout}s), 继续运行……")


def _wait_server_ready(port: int, proc: subprocess.Popen, timeout: float = READY_TIMEOUT) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"服务器进程提前退出, returncode={proc.returncode}")
        try:
            resp = requests.get(url, timeout=2.0)
            if resp.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(3.0)
    raise TimeoutError(f"服务器在 {timeout}s 内未就绪 (端口 {port})")


def _shutdown(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        _lf = getattr(proc, "_log_file", None)
        if _lf is not None:
            try:
                _lf.close()
            except Exception:
                pass
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    else:
        deadline = time.time() + 30.0
        while time.time() < deadline and proc.poll() is None:
            time.sleep(1.0)
        if proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.time() + 10.0
            while time.time() < deadline and proc.poll() is None:
                time.sleep(0.5)
    _lf = getattr(proc, "_log_file", None)
    if _lf is not None:
        try:
            _lf.close()
        except Exception:
            pass


def _start_server(port: int, model: str, tp: int, extra_args: List[str], mode: str) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--tp", str(tp),
        "--trust-remote-code",
        "--mem-fraction-static", MEM_FRAC,
        "--cuda-graph-max-bs", "4",
        "--reasoning-parser", "glm45",
        "--tool-call-parser", "glm47",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--disable-piecewise-cuda-graph",
        "--log-level", "warning",
        *extra_args,
    ]
    env = os.environ.copy()
    # ── GPU 可见性检查 ──
    # sglang 会为 TP 每个 rank 起独立子进程, 用 rank 号 set_device。如果父 shell 里
    # CUDA_VISIBLE_DEVICES 被限制到少于 tp 张 GPU, TP{tp-1} worker 会崩:
    #   torch.AcceleratorError: CUDA error: invalid device ordinal
    # (曾经踩过, 表现为 launch_server returncode=-9 / SIGKILL, 日志里能看到
    #  "Context: self.gpu_id=<tp-1> CUDA_VISIBLE_DEVICES='0'")。
    # 与其继承一个坏值让子进程崩 30s 后我们才知道, 不如启动前先自检。
    _parent_cvd = env.get("CUDA_VISIBLE_DEVICES")
    if _parent_cvd is not None and _parent_cvd.strip():
        _n_visible = len([x for x in _parent_cvd.split(",") if x.strip()])
        if _n_visible < tp:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES='{_parent_cvd}' 只暴露 {_n_visible} 张 GPU, "
                f"但 --tp={tp} 需要 {tp} 张。请在 shell 里执行:\n"
                f"    unset CUDA_VISIBLE_DEVICES     # 恢复看到所有卡\n"
                f"  或者\n"
                f"    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n"
                f"  再重跑本脚本。"
            )
    cu13_lib = "/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
    ld_path = f"/usr/local/cuda-13.0/compat:{cu13_lib}"
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{ld_path}:{existing}" if existing else ld_path
    # pip 里 sglang 是 editable install 指向 /root/RedKnot/python/sglang (无 PIC/A3 代码)。
    # 本仓库 (/root/sglang/) 才有 --pic-enable / --enable-a3 支持 (以及本 commit 加的
    # [PIC-A3-IMP-LIST] 日志)。跟 scripts/run_pic_a3_layer01_error.sh:42 一样, 在
    # 子进程 env 里把 /root/sglang/python 加到 PYTHONPATH 前面, 让 launch_server
    # 装的 sglang 被 /root/sglang 版本覆盖。REPO_ROOT 从本脚本位置反推
    # (test/manual/pic_a3_imp_token_view.py → /root/sglang)。
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _repo_py = os.path.join(_repo_root, "python")
    _existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_repo_py}:{_existing_pp}" if _existing_pp else _repo_py
    env["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = FAST_WARMUP
    env["SGLANG_JIT_DEEPGEMM_FAST_WARMUP_STRIDE_MULT"] = FAST_WARMUP_STRIDE_MULT
    env["SGLANG_EAGER_INPUT_NO_COPY"] = "1"
    env["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"

    log_path = f"/tmp/sglang_imp_view_{port}_{mode}.log"
    print(f"  [启动] mode={mode} tp={tp} port={port}")
    print(f"  [日志] {log_path}")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=log_file, start_new_session=True,
    )
    proc._log_file = log_file  # type: ignore[attr-defined]
    proc._log_path = log_path  # type: ignore[attr-defined]
    return proc


def _post_generate(port: int, text: str, max_new_tokens: int) -> Dict:
    resp = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={
            "text": text,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────
# 日志解析: 抓取 [PIC-A3-IMP-LIST]
# ─────────────────────────────────────────────────────────────────

_IMP_LIST_RE = re.compile(
    r"\[PIC-A3-IMP-LIST\]\s+"
    r"rid=(?P<rid>\S+)\s+"
    r"reuse=(?P<reuse>\S+)\s+"
    r"N=(?P<n>\d+)\s+"
    r"last_len=(?P<last_len>\d+)\s+"
    r"prefix_len=(?P<prefix_len>\d+)\s+"
    r"n_imp=(?P<n_imp>\d+)\s+"
    r"positions=(?P<positions>[\d,]+)"
)

_IMP_DIST_RE = re.compile(
    r"\[PIC-A3-IMP-DIST\]\s+total imp=(?P<total>\d+)\s+segs:\s+(?P<segs>.*)"
)


def _log_offset(proc: subprocess.Popen) -> int:
    log_path = getattr(proc, "_log_path", None)
    if not log_path:
        return 0
    try:
        return os.path.getsize(log_path)
    except OSError:
        return 0


def _parse_imp_list(proc: subprocess.Popen, since_offset: int) -> List[Dict]:
    """从服务器日志的 since_offset 之后, 抓所有 [PIC-A3-IMP-LIST] 行。

    TP=N 时同一请求每个 rank 各打一条 (positions 应该相同), 这里返回全部 rank 的
    解析结果 —— 由调用方去重 (通常拿第一条就行, 但保留所有便于诊断跨 rank 是否一致)。
    """
    log_path = getattr(proc, "_log_path", None)
    if not log_path:
        return []
    # 服务器 stderr 到文件有 buffer; 给一点时间落盘
    time.sleep(0.5)
    try:
        with open(log_path, "r", errors="replace") as f:
            f.seek(since_offset)
            chunk = f.read()
    except OSError:
        return []

    out: List[Dict] = []
    for m in _IMP_LIST_RE.finditer(chunk):
        positions = [int(x) for x in m.group("positions").split(",") if x]
        out.append({
            "rid": m.group("rid"),
            "reuse": m.group("reuse"),
            "N": int(m.group("n")),
            "last_len": int(m.group("last_len")),
            "prefix_len": int(m.group("prefix_len")),
            "n_imp": int(m.group("n_imp")),
            "positions": positions,
        })
    return out


def _parse_imp_dist(proc: subprocess.Popen, since_offset: int) -> Optional[Dict]:
    """抓一条 [PIC-A3-IMP-DIST] (segs=... 那种), 用于跟本脚本自己算的分段核对。"""
    log_path = getattr(proc, "_log_path", None)
    if not log_path:
        return None
    try:
        with open(log_path, "r", errors="replace") as f:
            f.seek(since_offset)
            chunk = f.read()
    except OSError:
        return None
    for m in _IMP_DIST_RE.finditer(chunk):
        return {"total": int(m.group("total")), "segs_raw": m.group("segs").strip()}
    return None


# ─────────────────────────────────────────────────────────────────
# 分段汇总 + 抽样文本
# ─────────────────────────────────────────────────────────────────

def _segments_from_lens(sys_len: int, c1_len: int, c2_len: int, c3_len: int, q_len: int
                        ) -> List[Tuple[str, int, int]]:
    """把段名 → (start, end) 半开区间, 与服务器端 pic_segments 计算方式一致。"""
    off = 0
    out: List[Tuple[str, int, int]] = []
    for name, ln in (("SYS", sys_len), ("C1", c1_len), ("C2", c2_len),
                     ("C3", c3_len), ("Q", q_len)):
        out.append((name, off, off + ln))
        off += ln
    return out


def _classify(pos: int, segs: List[Tuple[str, int, int]]) -> str:
    for name, s, e in segs:
        if s <= pos < e:
            return name
    return "?"


def _preview_around(
    all_ids: List[int],
    pos: int,
    tokenizer,
    window: int = 5,
    seg_bounds: Optional[Tuple[int, int]] = None,
) -> str:
    """打印 pos 前 window 个 + 后 window 个 token decode 出来的短文本片段。

    seg_bounds: 若给 (seg_start, seg_end), 窗口被 **卡在段内** — 避免 C2 尾部
    抽样时把 C3 头部的内容误当成 C2 的内容展示 (段边界溢出)。
    """
    seg_lo, seg_hi = seg_bounds if seg_bounds else (0, len(all_ids))
    lo = max(seg_lo, pos - window)
    hi = min(seg_hi, pos + window + 1)
    ids = all_ids[lo:hi]
    txt = tokenizer.decode(ids, skip_special_tokens=False)
    _rel = pos - lo
    # 标一下"pos 落在段头/段尾"的情况(窗口是不是被截了)
    _clip_lo = "^" if lo == seg_lo and pos - window < seg_lo else " "
    _clip_hi = "$" if hi == seg_hi and pos + window + 1 > seg_hi else " "
    return (
        f"pos={pos:>5} [{lo}:{hi}){_clip_lo}{_clip_hi} "
        f"tok_at_pos={all_ids[pos]:>6}  ctx={txt!r} (^ token idx {_rel})"
    )


def _sample_positions(positions: List[int], k: int) -> List[int]:
    """在 positions 里抽 k 个: 前 k/3 + 中间 k/3 + 末尾 k/3, 去重后有序返回。

    这样既能看到 top-attention 的分布, 也能看到"边缘"是不是选进去了。
    """
    if not positions:
        return []
    if len(positions) <= k:
        return list(positions)
    n = len(positions)
    k3 = max(1, k // 3)
    idx = list(range(k3))  # 前 k3
    idx += [n // 2 + i - k3 // 2 for i in range(k3)]  # 中间 k3
    idx += [n - k3 + i for i in range(k3)]  # 末尾 k3
    idx = [i for i in idx if 0 <= i < n]
    picks = sorted(set(positions[i] for i in idx))
    return picks


def summarize_imp(
    positions: List[int],
    seg_ranges: List[Tuple[str, int, int]],
    input_ids: List[int],
    tokenizer,
    per_seg_samples: int,
    hit_seg_names: Optional[set] = None,
    miss_seg_names: Optional[set] = None,
) -> str:
    """把 positions 按段归类并抽样, 拼成一大段可读文本。返回打印用字符串。

    hit_seg_names / miss_seg_names: 段名集合, 用于在段汇总里标注每段的角色
    (hit → 走 top-k 挑; miss → 全体强制 imp; 其他 → SYS/Q 特殊段)。
    """
    hit_seg_names = hit_seg_names or set()
    miss_seg_names = miss_seg_names or set()
    lines: List[str] = []
    total = len(positions)
    lines.append(f"  imp 定义: pic_a3 / pic_cacheblend 在 layer 1 决策后送去 layer 2+ 重算的 token")
    lines.append(f"           (非 imp 的 hit 位置复用 PIC 段缓存里的 stale KV)")
    lines.append(f"  imp = miss段(全体) ∪ hit段(top-k 按 Q·K/V-diff 挑, 比例=recomp_ratio) ∪ Q段(全体)")
    lines.append("")
    lines.append(f"  imp 总数: {total}  (全序列长度 N={len(input_ids)})")
    lines.append("")
    lines.append(f"  {'段':<4} {'角色':<8} {'范围':<20} {'段长':>7} {'imp数':>7} {'占段':>8} {'占总imp':>8}")
    lines.append(f"  {'-'*4} {'-'*8} {'-'*20} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")

    # 分段统计
    seg_imps: Dict[str, List[int]] = {name: [] for name, _, _ in seg_ranges}
    for p in positions:
        cls = _classify(p, seg_ranges)
        seg_imps.setdefault(cls, []).append(p)

    for name, s, e in seg_ranges:
        seg_len = e - s
        imp_in = seg_imps.get(name, [])
        n_imp = len(imp_in)
        frac_seg = f"{n_imp / seg_len:.1%}" if seg_len > 0 else "-"
        frac_tot = f"{n_imp / total:.1%}" if total > 0 else "-"
        if name in hit_seg_names:
            role = "HIT"
        elif name in miss_seg_names:
            role = "MISS"
        elif name in ("SYS", "Q"):
            role = name  # SYS 通常是 prefix, Q 是 last_len 强制 imp
        else:
            role = "?"
        lines.append(
            f"  {name:<4} {role:<8} [{s:>6},{e:>6}) {seg_len:>7} {n_imp:>7} "
            f"{frac_seg:>8} {frac_tot:>8}"
        )
    # 其他 (理论上不应该有, 但防御性打印)
    _other = seg_imps.get("?", [])
    if _other:
        lines.append(f"  {'?':<4} {'-':<8} (out-of-range)      {'-':>7} {len(_other):>7}"
                     f"      -        -")

    # 抽样文本
    lines.append("")
    lines.append(f"  ── 每段抽样 imp 位置 + 邻域文本 (每段最多 {per_seg_samples} 个) ──")
    lines.append(f"     窗口卡在段内; `^`/`$` 标记窗口被段头/段尾截断")
    for name, s, e in seg_ranges:
        imp_in = seg_imps.get(name, [])
        if not imp_in:
            lines.append(f"\n  [{name}] 段内无 imp 位置")
            continue
        picks = _sample_positions(imp_in, per_seg_samples)
        lines.append(f"\n  [{name}]  段范围=[{s},{e})  imp={len(imp_in)}  抽样 {len(picks)} 个:")
        for p in picks:
            lines.append(f"    {_preview_around(input_ids, p, tokenizer, seg_bounds=(s, e))}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# 单模式跑流程
# ─────────────────────────────────────────────────────────────────

def _mode_extra_args(mode: str, recomp_ratio: float) -> List[str]:
    pic_common = [
        "--pic-enable", "--page-size", "64",
        "--chunked-prefill-size", "-1",
        "--pic-separator-str", SEP,
    ]
    if mode == "pic_a3":
        return [*pic_common, "--enable-a3", "--recomp-ratio", str(recomp_ratio),
                "--disable-cuda-graph", "--disable-overlap-schedule",
                "--disable-radix-cache"]
    if mode == "pic_cacheblend":
        return [*pic_common, "--enable-cacheblend", "--recomp-ratio", str(recomp_ratio),
                "--disable-cuda-graph", "--disable-overlap-schedule",
                "--disable-radix-cache"]
    raise ValueError(f"unknown mode: {mode}")


def run_mode(
    mode: str,
    port: int,
    model: str,
    tp: int,
    prompts: Dict,
    per_seg_samples: int,
    recomp_ratio: float,
) -> None:
    print(f"\n{'#' * 72}")
    print(f"  Mode: {mode}")
    print(f"{'#' * 72}")

    _wait_port_free(port)
    proc = _start_server(port, model, tp, _mode_extra_args(mode, recomp_ratio), mode)
    try:
        print(f"  [ready] 等待服务器就绪 (最多 {READY_TIMEOUT}s)……")
        _wait_server_ready(port, proc)
        print("  [ready] 服务器就绪")

        # warmup: C1, C3 分别命中一次
        for i, wp in enumerate((prompts["pic_w1"], prompts["pic_w3"]), 1):
            print(f"  [warmup {i}/2] {len(wp)} 字符")
            _post_generate(port, wp, max_new_tokens=4)

        # test 请求 —— 触发 pic_a3_pick_imp
        print("  [test] 发起测试请求 (max_new_tokens=1)……")
        offset = _log_offset(proc)
        t0 = time.perf_counter()
        _post_generate(port, prompts["pic_test"], max_new_tokens=1)
        elapsed = time.perf_counter() - t0
        print(f"  [test] TTFT ≈ {elapsed:.3f}s")

        # 解析 [PIC-A3-IMP-LIST]
        parsed = _parse_imp_list(proc, offset)
        if not parsed:
            print(f"\n  [错误] 日志里没有找到 [PIC-A3-IMP-LIST] 行!\n"
                  f"        请检查 model_runner._pic_a3_pick_imp 末尾是否包含该 log。\n"
                  f"        或看服务器日志: {getattr(proc, '_log_path', '?')}")
            return

        # 多 rank 时应该拿到多条; 只用第一条 (跨 rank positions 应该相同)
        record = parsed[0]
        if len(parsed) > 1:
            # 快速核对: 所有 rank 的 positions 是否一致
            _all_same = all(r["positions"] == record["positions"] for r in parsed[1:])
            print(f"  [解析] {len(parsed)} 条 IMP-LIST 记录 (跨 rank), "
                  f"positions 一致={_all_same}")
        else:
            print(f"  [解析] 1 条 IMP-LIST 记录")

        # 顺带把 [PIC-A3-IMP-DIST] 也拿出来核对
        dist = _parse_imp_dist(proc, offset)
        if dist:
            print(f"  [解析] server-side 分段统计: {dist['segs_raw']}")

        print(f"\n  rid={record['rid']}  reuse={record['reuse']}  "
              f"N={record['N']}  last_len={record['last_len']}  "
              f"prefix_len={record['prefix_len']}  n_imp={record['n_imp']}")

        # 用 tokenizer 把 test prompt 完整 encode 出来, 与服务器端应完全一致
        # (段边界通过 build_prompts 里已算好)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        # 用 pic_test 里的段拼 (SEP 在 split_and_tokenize 里不产生 token,
        # 所以拼段 encode 得到的 ids 跟服务器 pic_segments 里位置一致)
        _no_sep = (prompts["sys_seg"] + prompts["c1_seg"] + prompts["c2_seg"]
                   + prompts["c3_seg"] + prompts["q_seg"])
        input_ids = tok.encode(_no_sep, add_special_tokens=False)

        if len(input_ids) != record["N"]:
            print(f"\n  [警告] 本脚本 encode 得到 {len(input_ids)} tokens, "
                  f"服务器报的 N={record['N']} 不一致 (可能 tokenizer 差异或 pad "
                  f"策略不同). 分段位置以段长为准, 抽样文本可能有 1-2 token 偏差。")

        seg_ranges = _segments_from_lens(
            prompts["sys_len"], prompts["c1_len"], prompts["c2_len"],
            prompts["c3_len"], prompts["q_len"],
        )
        # warmup 命中 SYS+C1+Q 和 SYS+C3+Q → PIC 段缓存里存了 SYS/C1/C3;
        # test 请求 SYS+C1+C2+C3+Q → C2 是唯一没在 warmup 出现的段 → miss;
        # Q 每次都新 (query 区域, last_len 强制 imp)。
        # 这三个集合决定表格里 role 列显示 HIT / MISS / Q。
        summary = summarize_imp(
            record["positions"], seg_ranges, input_ids, tok, per_seg_samples,
            hit_seg_names={"SYS", "C1", "C3"},
            miss_seg_names={"C2"},
        )
        print(summary)

    finally:
        print(f"\n  [shutdown] 关闭服务器……")
        _shutdown(proc)
        _wait_port_free(port)


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────

def build_prompts(model: str) -> Dict:
    """跟 quick_test_online.py 完全一致的段 pad 到 64 + prompt 拼接。

    SYS/Q 已经在文件顶部通过重复扩展到接近 64 tokens (SYS×10=60, Q×7=63),
    pad_to_multiple 只需要补 <5 个 padding token, padding 占段 <7% —— attention
    不再被 dominant padding token 淹没。
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    print(f"  [tokenize] 每段 pad 到 {PIC_ALIGN} 的倍数……")
    sys_seg, sys_orig, sys_len = _pad_text_to_multiple(SYS, tok, PIC_ALIGN)
    c1_seg, c1_orig, c1_len = _pad_text_to_multiple(C1, tok, PIC_ALIGN)
    c2_seg, c2_orig, c2_len = _pad_text_to_multiple(C2, tok, PIC_ALIGN)
    c3_seg, c3_orig, c3_len = _pad_text_to_multiple(C3, tok, PIC_ALIGN)
    q_seg, q_orig, q_len = _pad_text_to_multiple(Q, tok, PIC_ALIGN)
    total = sys_len + c1_len + c2_len + c3_len + q_len
    print(f"    SYS={sys_len} (real={sys_orig}, {sys_orig/sys_len:.1%})  "
          f"C1={c1_len}  C2={c2_len}  C3={c3_len}  "
          f"Q={q_len} (real={q_orig}, {q_orig/q_len:.1%})  total={total}")

    pic_test = f"{sys_seg}{SEP}{c1_seg}{SEP}{c2_seg}{SEP}{c3_seg}{SEP}{q_seg}"
    pic_w1 = f"{sys_seg}{SEP}{c1_seg}{SEP}{q_seg}"
    pic_w3 = f"{sys_seg}{SEP}{c3_seg}{SEP}{q_seg}"

    return {
        "sys_seg": sys_seg, "c1_seg": c1_seg, "c2_seg": c2_seg,
        "c3_seg": c3_seg, "q_seg": q_seg,
        "sys_len": sys_len, "c1_len": c1_len, "c2_len": c2_len,
        "c3_len": c3_len, "q_len": q_len,
        "pic_test": pic_test, "pic_w1": pic_w1, "pic_w3": pic_w3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View pic_a3 / pic_cacheblend imp token selection "
                     "(reuses quick_test_online 3-doc scenario)"
    )
    parser.add_argument("--model", default="/workspace/models/GLM-5.2-FP8")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument(
        "--modes", nargs="+", default=["pic_a3", "pic_cacheblend"],
        choices=["pic_a3", "pic_cacheblend"],
        help="要跑的模式 (只支持 pic_a3 和 pic_cacheblend, 其他模式没有 imp 选择)",
    )
    parser.add_argument(
        "--per-seg-samples", type=int, default=15,
        help="每个段最多打印多少个 imp 位置的邻域文本 (默认 15)",
    )
    parser.add_argument(
        "--recomp-ratio", type=float, default=0.15,
        help="A3/CacheBlend recomp_ratio (与 quick_test_online 一致, 默认 0.15)",
    )
    parser.add_argument(
        "--out-log", type=str, default=None, metavar="PATH",
        help="把本脚本 stdout/stderr 全量 tee 到该文件, 跑完后可以 less 翻看全部输出。"
             "默认 /tmp/pic_a3_imp_view_<timestamp>.log。传 '' (空串) 禁用 tee。",
    )
    args = parser.parse_args()

    # ── 输出 tee ──
    # 终端 scrollback 有限, 长跑输出会被顶掉。默认把所有 print 复制一份到磁盘文件,
    # 用户跑完可以 `less /tmp/pic_a3_imp_view_<ts>.log` 翻完整历史。
    # 传 --out-log '' 关掉这个行为。传 --out-log <path> 用指定路径。
    _tee_fh = None
    if args.out_log is None:
        args.out_log = f"/tmp/pic_a3_imp_view_{int(time.time())}.log"
    if args.out_log:
        try:
            _tee_fh = open(args.out_log, "w", buffering=1)  # line-buffered
        except OSError as _e:
            print(f"  [警告] 无法打开 --out-log={args.out_log!r}: {_e}, 继续但不 tee")
            _tee_fh = None
        else:
            class _Tee:
                """把 write / flush 转发到原 stream + 磁盘文件。"""
                def __init__(self, orig, fh):
                    self._orig = orig
                    self._fh = fh
                def write(self, s):
                    self._orig.write(s)
                    try:
                        self._fh.write(s)
                    except Exception:
                        pass
                    return len(s)
                def flush(self):
                    self._orig.flush()
                    try:
                        self._fh.flush()
                    except Exception:
                        pass
                def __getattr__(self, name):
                    return getattr(self._orig, name)
            sys.stdout = _Tee(sys.__stdout__, _tee_fh)
            sys.stderr = _Tee(sys.__stderr__, _tee_fh)
            print(f"  [tee] 全量输出复制到 {args.out_log} (--out-log '' 可关闭)")

    print(f"\n{'=' * 72}")
    print(f"  pic_a3 / pic_cacheblend imp token viewer")
    print(f"  Model: {args.model}   TP={args.tp}")
    print(f"  Modes: {args.modes}   per_seg_samples={args.per_seg_samples}")
    print(f"  recomp_ratio={args.recomp_ratio}")
    print(f"{'=' * 72}\n")

    prompts = build_prompts(args.model)
    for mode in args.modes:
        try:
            run_mode(
                mode=mode, port=args.port, model=args.model, tp=args.tp,
                prompts=prompts, per_seg_samples=args.per_seg_samples,
                recomp_ratio=args.recomp_ratio,
            )
        except Exception as _e:
            import traceback
            print(f"\n>>> [{mode}] 失败: {_e}")
            traceback.print_exc()

    print(f"\n{'=' * 72}")
    print("  完成")
    if _tee_fh is not None:
        print(f"  完整输出已保存到: {args.out_log}")
        print(f"    less {args.out_log}         # 翻看")
        print(f"    grep -E 'imp|C[123]' {args.out_log}   # 只看关键行")
    print(f"{'=' * 72}\n")

    if _tee_fh is not None:
        try:
            _tee_fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
