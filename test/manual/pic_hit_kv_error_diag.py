#!/usr/bin/env python3
"""pic_hit_kv_error_diag.py — 诊断 PIC 命中缓存 vs 全量重算的逐层 token 级误差

背景:
    PIC 在 quick_test_online.py 里让 warmup2(SYS+C3+Q)先缓存 C3 段的 KV,
    随后测试请求(SYS+C1+C2+C3+Q)直接复用这份 C3 缓存。但复用位置(test 中的
    C3)的注意力上下文与缓存位置(warmup2 中的 C3)不同 —— 前者能看到 SYS+C1+C2,
    后者只看到 SYS。这套"上下文错位"会让 K/V 产生数值漂移。

本脚本量化这个漂移:
    1. Run A: full_recompute 跑 warmup2 prompt (SYS+C3+Q),抓每层 latent
    2. Run B: full_recompute 跑 test prompt (SYS+C1+C2+C3+Q),抓每层 latent
    3. 提取两次运行中 C3 段的 latent,element-wise 相减
    4. 生成 PNG:
         (a) 每层 mean L2 error 折线图
         (b) 每层 max L2 error 折线图  (峰值 outlier 位置)
         (c) heatmap: rows=layer, cols=C3 内 token 位置, cell=L2 error

技术要点:
    - GLM-5.2 是 MQA(num_kv_heads=1)→ KV 不做 TP 切分 → 只用 rank0 分片即可
    - latent = concat(k_nope, k_pe) —— k_nope 是位置无关,k_pe 位置相关
      默认对整个 latent 做误差;--separate-nope-pe 可拆开画
    - 跑两次 Engine 各自要 --tp N 起服务(GLM-5.2 至少 tp=8)

用法:
    python test/manual/pic_hit_kv_error_diag.py \\
        --model /workspace/models/GLM-5.2-FP8 \\
        --tp 8 \\
        --output pic_hit_kv_error.png

    # 加速二次运行:如果两次的 capture 已存在,跳过 Engine 启动直接分析:
    python test/manual/pic_hit_kv_error_diag.py --skip-capture --output error.png
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch

# ── 与 quick_test_online.py 保持一致的常量 ──
SYS = "You are a helpful assistant."
C1 = "Document A about cats. " * 800
C2 = "Document B about dogs. " * 800
C3 = "Document C about birds. " * 800
Q = "Question: which animal is in document B?"
PIC_ALIGN = int(os.environ.get("PIC_ALIGN", "64"))

# 分片文件名正则(与 precompute_kv_mla.py 一致)
_SHARD_RE = re.compile(
    r"rank(?P<rank>\d+)_layer(?P<layer>\d+)_(?P<kind>latent|index_k)\.pt"
)


# ─────────────────────────────────────────────────────────────────
# Padding 对齐(与 quick_test_online.py 相同的算法)
# ─────────────────────────────────────────────────────────────────

def _pad_to_multiple(text: str, tokenizer, multiple: int = PIC_ALIGN):
    ids = tokenizer.encode(text, add_special_tokens=False)
    orig = len(ids)
    if orig % multiple == 0:
        return text, orig, orig
    target = ((orig // multiple) + 1) * multiple
    for pad_char in [" ", "\n", ".", "!", "a", "0"]:
        t, cur, guard = text, orig, 0
        while cur < target and guard < multiple * 8:
            t = t + pad_char
            cur = len(tokenizer.encode(t, add_special_tokens=False))
            guard += 1
        if cur == target:
            return t, orig, cur
    return t, orig, cur


def build_prompts(model: str):
    """构造 warmup2 prompt (SYS+C3+Q) 与 test prompt (SYS+C1+C2+C3+Q),
    以及各段在两个 prompt 里的 token 位置区间。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    print(f"  [tokenize] 各段对齐到 {PIC_ALIGN} 的倍数……")
    sys_seg, _, sys_len = _pad_to_multiple(SYS, tok)
    c1_seg,  _, c1_len  = _pad_to_multiple(C1, tok)
    c2_seg,  _, c2_len  = _pad_to_multiple(C2, tok)
    c3_seg,  _, c3_len  = _pad_to_multiple(C3, tok)
    q_seg,   _, q_len   = _pad_to_multiple(Q, tok)
    print(f"    SYS={sys_len}  C1={c1_len}  C2={c2_len}  C3={c3_len}  Q={q_len}")

    warmup2_prompt = f"{sys_seg}{c3_seg}{q_seg}"
    test_prompt    = f"{sys_seg}{c1_seg}{c2_seg}{c3_seg}{q_seg}"

    # C3 位置区间:warmup2 里在 SYS 之后,test 里在 SYS+C1+C2 之后
    c3_range_warmup2 = (sys_len,                      sys_len + c3_len)
    c3_range_test    = (sys_len + c1_len + c2_len,    sys_len + c1_len + c2_len + c3_len)

    total_warmup2 = sys_len + c3_len + q_len
    total_test    = sys_len + c1_len + c2_len + c3_len + q_len

    return {
        "warmup2_prompt": warmup2_prompt,
        "test_prompt":    test_prompt,
        "c3_range_warmup2": c3_range_warmup2,
        "c3_range_test":    c3_range_test,
        "total_warmup2":  total_warmup2,
        "total_test":     total_test,
        "c3_len":         c3_len,
    }


# ─────────────────────────────────────────────────────────────────
# 捕获(用 SGLANG_A3_CAPTURE_KV_DIR env 钩子跑 sglang.Engine)
# ─────────────────────────────────────────────────────────────────

def run_capture(model: str, tp: int, prompt: str, capture_dir: str,
                mem_frac: float = 0.82) -> None:
    """起一个 Engine,发一次 prefill(max_new=1),捕获每层每 rank 的 latent 到 capture_dir。"""
    os.makedirs(capture_dir, exist_ok=True)
    # 清理 stale 分片
    for f in os.listdir(capture_dir):
        if _SHARD_RE.match(f):
            os.remove(os.path.join(capture_dir, f))

    # 设置 env(会被 TP worker 继承)
    os.environ["SGLANG_A3_CAPTURE_KV_DIR"] = capture_dir
    print(f"  [engine] tp={tp}, capture_dir={capture_dir}")

    # 延迟 import 避免污染主进程
    from sglang import Engine
    engine = Engine(
        model_path=model,
        tp_size=tp,
        trust_remote_code=True,
        mem_fraction_static=mem_frac,
        chunked_prefill_size=-1,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        # 关键:piecewise cuda graph 会在 warmup 时 torch.compile forward,
        # 编译后的 graph 不含 capture 分支(因 is_compiling() guard),后续
        # 真实请求都走编译版 → capture 永远不触发。必须显式关掉。
        disable_piecewise_cuda_graph=True,
    )
    try:
        t0 = time.perf_counter()
        engine.generate(
            prompt=prompt,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        )
        print(f"  [engine] prefill done in {time.perf_counter()-t0:.2f}s")
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass
        os.environ.pop("SGLANG_A3_CAPTURE_KV_DIR", None)

    # 给 TP worker 一点时间刷 torch.save
    time.sleep(1.5)
    # 强制 gc,让 Engine 的资源尽早释放
    gc.collect()
    torch.cuda.empty_cache()


def load_rank0_latent(capture_dir: str, num_layers: int) -> torch.Tensor:
    """从 capture_dir 里加载 rank0 的每层 latent,拼成 (num_layers, total_len, latent_dim)。

    对 MQA(GLM-5.2 num_kv_heads=1),KV 在 rank 间是复制而不是切分,只读 rank0 即可。
    """
    per_layer: Dict[int, torch.Tensor] = {}
    for fname in os.listdir(capture_dir):
        m = _SHARD_RE.match(fname)
        if not m or int(m["rank"]) != 0 or m["kind"] != "latent":
            continue
        layer_id = int(m["layer"])
        t = torch.load(os.path.join(capture_dir, fname), map_location="cpu")
        per_layer[layer_id] = t

    if not per_layer:
        raise RuntimeError(f"no rank0 latent shards in {capture_dir}")

    first = per_layer[min(per_layer)]
    T, D = first.shape
    out = torch.zeros((num_layers, T, D), dtype=first.dtype)
    for lid, t in per_layer.items():
        if t.shape != (T, D):
            print(f"  [warn] layer{lid} shape {t.shape} != {(T, D)},跳过",
                  file=sys.stderr)
            continue
        out[lid] = t
    return out


# ─────────────────────────────────────────────────────────────────
# 误差计算 + 绘图
# ─────────────────────────────────────────────────────────────────

def compute_per_layer_errors(
    lat_warmup2: torch.Tensor,       # (L, T_w2, D)
    lat_test:    torch.Tensor,       # (L, T_test, D)
    c3_range_warmup2: Tuple[int, int],
    c3_range_test:    Tuple[int, int],
    kv_lora_rank: int,
    separate_nope_pe: bool = False,
) -> Dict[str, np.ndarray]:
    """返回:
        {
          "l2_per_layer_per_token": (L, N_C3) — 每层每 token 的 L2 距离
          "mean_per_layer":          (L,)     — 每层平均
          "max_per_layer":           (L,)     — 每层最大
          # 若 separate_nope_pe:
          "l2_nope_per_layer_per_token": (L, N_C3)  — 仅 k_nope 部分
          "l2_pe_per_layer_per_token":   (L, N_C3)  — 仅 k_pe 部分
        }
    """
    s_w, e_w = c3_range_warmup2
    s_t, e_t = c3_range_test
    if (e_w - s_w) != (e_t - s_t):
        raise ValueError(f"C3 长度不一致: warmup2={e_w-s_w}, test={e_t-s_t}")

    # 切出 C3 段,shape (L, N_C3, D)
    a = lat_warmup2[:, s_w:e_w, :].float()
    b = lat_test   [:, s_t:e_t, :].float()

    diff = a - b                             # (L, N_C3, D)
    l2   = diff.pow(2).sum(dim=-1).sqrt()    # (L, N_C3)

    out = {
        "l2_per_layer_per_token": l2.numpy(),
        "mean_per_layer":         l2.mean(dim=-1).numpy(),
        "max_per_layer":          l2.max(dim=-1).values.numpy(),
    }

    if separate_nope_pe:
        a_nope, a_pe = a[..., :kv_lora_rank], a[..., kv_lora_rank:]
        b_nope, b_pe = b[..., :kv_lora_rank], b[..., kv_lora_rank:]
        l2_nope = (a_nope - b_nope).pow(2).sum(dim=-1).sqrt().numpy()
        l2_pe   = (a_pe   - b_pe  ).pow(2).sum(dim=-1).sqrt().numpy()
        out["l2_nope_per_layer_per_token"] = l2_nope
        out["l2_pe_per_layer_per_token"]   = l2_pe

    return out


def plot(errors: Dict[str, np.ndarray], output: str,
         c3_len: int, separate_nope_pe: bool = False) -> None:
    """出图:
      - Panel 1: 每层 mean / max / p95 L2 折线
      - Panel 2: heatmap (layer × C3 position)
      - Panel 3(可选): 分开 nope / pe 的 mean 折线
    """
    import matplotlib.pyplot as plt

    L = errors["mean_per_layer"].shape[0]
    n_panels = 3 if separate_nope_pe else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels),
                              constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1: 折线 mean / max / p95 ──
    ax = axes[0]
    layer_ids = np.arange(L)
    p95 = np.percentile(errors["l2_per_layer_per_token"], 95, axis=-1)
    ax.plot(layer_ids, errors["mean_per_layer"], label="mean", marker="o", markersize=3)
    ax.plot(layer_ids, p95,                      label="p95",  marker="s", markersize=3, linestyle="--")
    ax.plot(layer_ids, errors["max_per_layer"],  label="max",  marker="^", markersize=3, linestyle=":")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("L2 error  (||kv_warmup2 - kv_test||_2 per token)")
    ax.set_title(f"PIC hit KV error at C3 positions ({c3_len} tokens) — per-layer aggregate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # 用 log Y 让层间差异更清晰
    if errors["mean_per_layer"].max() / max(errors["mean_per_layer"].min(), 1e-9) > 100:
        ax.set_yscale("log")

    # ── Panel 2: heatmap ──
    ax = axes[1]
    hm_data = errors["l2_per_layer_per_token"]
    im = ax.imshow(hm_data, aspect="auto", cmap="viridis", origin="lower",
                    extent=[0, c3_len, 0, L])
    ax.set_xlabel("Position within C3 (token index)")
    ax.set_ylabel("Layer index")
    ax.set_title("L2 error heatmap  (rows=layer, cols=token in C3)")
    plt.colorbar(im, ax=ax, label="L2 error")

    # ── Panel 3(可选): nope vs pe 分开 ──
    if separate_nope_pe:
        ax = axes[2]
        m_nope = errors["l2_nope_per_layer_per_token"].mean(axis=-1)
        m_pe   = errors["l2_pe_per_layer_per_token"].mean(axis=-1)
        ax.plot(layer_ids, m_nope, label="k_nope (position-free)", marker="o", markersize=3)
        ax.plot(layer_ids, m_pe,   label="k_pe (position-encoded)", marker="s", markersize=3)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("mean L2 error")
        ax.set_title("Nope vs PE component error  "
                      "(nope=context drift only; pe includes RoPE-position mismatch)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.savefig(output, dpi=120, bbox_inches="tight")
    print(f"\n  [plot] 已保存 → {output}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PIC hit-cache token-level error diagnostic (per-layer)")
    parser.add_argument("--model", default="/workspace/models/GLM-5.2-FP8")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--mem-fraction", type=float, default=0.82)
    parser.add_argument("--output", default="pic_hit_kv_error.png",
                         help="输出 PNG 路径")
    parser.add_argument("--capture-dir-a", default="/tmp/errdiag_warmup2",
                         help="Run A(warmup2)捕获目录")
    parser.add_argument("--capture-dir-b", default="/tmp/errdiag_test",
                         help="Run B(test)捕获目录")
    parser.add_argument("--skip-capture", action="store_true",
                         help="跳过 Engine 捕获(用现成的 capture 目录直接分析)")
    parser.add_argument("--keep-shards", action="store_true",
                         help="跑完不清理 capture 目录")
    parser.add_argument("--separate-nope-pe", action="store_true",
                         help="第 3 个子图:拆开 k_nope vs k_pe 的误差")
    parser.add_argument("--csv", type=str, default=None,
                         help="同时导出 per-layer mean/max/p95 到 CSV")
    args = parser.parse_args()

    print(f"\n{'#'*64}")
    print(f"  PIC hit-KV error diagnostic")
    print(f"  Model: {args.model}   TP={args.tp}")
    print(f"{'#'*64}\n")

    # 1. 构造 prompts + 位置
    p = build_prompts(args.model)
    print(f"  warmup2 prompt: {p['total_warmup2']} tokens, C3=[{p['c3_range_warmup2'][0]}, {p['c3_range_warmup2'][1]})")
    print(f"  test prompt   : {p['total_test']} tokens, C3=[{p['c3_range_test'][0]}, {p['c3_range_test'][1]})")

    # 2. 两次 Engine 捕获(可跳过)
    if not args.skip_capture:
        print(f"\n[Run A] full_recompute on warmup2 prompt (SYS+C3+Q)")
        run_capture(args.model, args.tp, p["warmup2_prompt"],
                    args.capture_dir_a, args.mem_fraction)

        print(f"\n[Run B] full_recompute on test prompt (SYS+C1+C2+C3+Q)")
        run_capture(args.model, args.tp, p["test_prompt"],
                    args.capture_dir_b, args.mem_fraction)
    else:
        print(f"\n  [--skip-capture] 复用已有 capture 目录")

    # 3. 读回 rank0 latent
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    num_layers = cfg.num_hidden_layers
    kv_lora_rank = getattr(cfg, "kv_lora_rank", None)

    print(f"\n  [load] rank0 latent from {args.capture_dir_a}")
    lat_a = load_rank0_latent(args.capture_dir_a, num_layers)
    print(f"    shape = {tuple(lat_a.shape)}, dtype = {lat_a.dtype}")

    print(f"  [load] rank0 latent from {args.capture_dir_b}")
    lat_b = load_rank0_latent(args.capture_dir_b, num_layers)
    print(f"    shape = {tuple(lat_b.shape)}, dtype = {lat_b.dtype}")

    # 4. 计算逐层误差
    print(f"\n  [compute] per-layer L2 error at C3 positions")
    errors = compute_per_layer_errors(
        lat_a, lat_b,
        p["c3_range_warmup2"], p["c3_range_test"],
        kv_lora_rank=kv_lora_rank or 0,
        separate_nope_pe=args.separate_nope_pe,
    )

    # 5. 打印摘要
    print(f"\n  Layer-wise error summary:")
    print(f"  {'layer':>5}  {'mean':>10}  {'max':>10}  {'p95':>10}")
    p95_all = np.percentile(errors["l2_per_layer_per_token"], 95, axis=-1)
    for i in range(num_layers):
        print(f"  {i:>5}  {errors['mean_per_layer'][i]:>10.4f}  "
              f"{errors['max_per_layer'][i]:>10.4f}  {p95_all[i]:>10.4f}")

    # 6. CSV(可选)
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "mean_l2", "max_l2", "p95_l2"])
            for i in range(num_layers):
                w.writerow([i, float(errors["mean_per_layer"][i]),
                              float(errors["max_per_layer"][i]),
                              float(p95_all[i])])
        print(f"\n  [csv] 已导出 → {args.csv}")

    # 7. 出图
    plot(errors, args.output, c3_len=p["c3_len"],
         separate_nope_pe=args.separate_nope_pe)

    # 8. cleanup
    if not args.keep_shards and not args.skip_capture:
        for d in (args.capture_dir_a, args.capture_dir_b):
            for f in os.listdir(d):
                if _SHARD_RE.match(f):
                    os.remove(os.path.join(d, f))
        print(f"  [cleanup] 清理 capture 目录(--keep-shards 保留)")


if __name__ == "__main__":
    main()
