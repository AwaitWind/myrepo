#!/usr/bin/env python3
"""pic_mode_kv_error_diag.py — 逐层 KV 误差诊断（full_recompute vs pic / pic_a3 / pic_cacheblend）

回答用户的问题:
    位置编码在 full_compute / pic / pic_a3 / pic_cacheblend 四种模式下是否有误差？
    如果有,每层每 token 的量级是多少?最坏在哪一层?

方法:
    对同一个 test prompt (SYS+C1+C2+C3+Q) 分别以 4 种模式各起一次 sglang.Engine:
      1. full_recompute   -- 无 PIC / 无 A3 / 无 CacheBlend, 是 ground truth
      2. pic              -- --pic-enable, warmup 后 C3 段命中, 走 pub K + delta-RoPE
      3. pic_a3           -- --pic-enable --enable-a3, 命中段进 forward + A³ K 覆盖
      4. pic_cacheblend   -- --pic-enable --enable-cacheblend, 同上但 V-diff 选 imp

    每个模式跑完 warmup(C1、C3 段命中)后, 发起 test 请求 (SYS+C1+C2+C3+Q),
    通过 SGLANG_PIC_POOL_DUMP_DIR 环境变量在 _pic_writeback_mla_kv 末尾 dump
    kv_pool 里请求所有 token 位置(通过 req_to_token 遍历)的最终 K, 逐层逐 rank
    保存到 /tmp/errmodes_<mode>/。这是 attention 未来读取的真实 K, 涵盖:
      * pic 的 pub K 复制 + delta-RoPE'd k_pe
      * pic_a3 / pic_cacheblend 的 postchecking K 覆盖

    读回后用 full_recompute 做 baseline, 对每个非 baseline 模式:
      diff[layer, token] = ||K_mode[layer, token] - K_full[layer, token]||_2
    再切出 C3 段 (test prompt 里位置 [SYS+C1+C2, SYS+C1+C2+C3)) 做逐层可视化。

输出:
    一张多面板 PNG:
      Row 1 (共享): 3 个模式的 per-layer mean L2 曲线叠加
      Row 2: pic 的 layer × C3-token 热力图
      Row 3: pic_a3 的热力图
      Row 4: pic_cacheblend 的热力图
      Row 5 (可选 --separate-nope-pe): 3 模式 nope vs pe 分量 mean 曲线

用法:
    python test/manual/pic_mode_kv_error_diag.py \\
        --model /workspace/models/GLM-5.2-FP8 \\
        --tp 8 \\
        --output pic_mode_kv_error.png

    # 加速二次运行:如果 4 个 mode 的 pool dump 已存在,跳过 Engine 启动:
    python test/manual/pic_mode_kv_error_diag.py --skip-capture --output error.png

    # 只跑指定几个模式(全部模式默认都跑,但可以裁减):
    python test/manual/pic_mode_kv_error_diag.py --modes full_recompute pic pic_a3

依赖:
    * SGLANG_PIC_POOL_DUMP_DIR hook 在 model_runner.py::_pic_writeback_mla_kv 末尾
      (每个 rank 每层各写一个 .pt: rank{r}_layer{l}_latent.pt)。
    * GLM-5.2 是 MQA (num_kv_heads=1) → KV 在 rank 间是复制,只读 rank0 分片。
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

# ── 与 quick_test_online.py / pic_hit_kv_error_diag.py 对齐 ──
SEP = "<<PIC_SEP>>"
SYS = "You are a helpful assistant."
C1 = "Document A about cats. " * 800
C2 = "Document B about dogs. " * 800
C3 = "Document C about birds. " * 800
Q = "Question: which animal is in document B?"
PIC_ALIGN = int(os.environ.get("PIC_ALIGN", "64"))

MODES_ALL = ("full_recompute", "pic", "pic_a3", "pic_cacheblend")
NON_BASELINE = ("pic", "pic_a3", "pic_cacheblend")

_SHARD_RE = re.compile(
    r"rank(?P<rank>\d+)_layer(?P<layer>\d+)_latent\.pt"
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
    """构造 test 主 prompt (SYS+C1+C2+C3+Q) + 两个 warmup (SYS+C1+Q, SYS+C3+Q),
    以及 C3 段在 test prompt 里的 token 位置区间(所有模式共享)。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    print(f"  [tokenize] 各段对齐到 {PIC_ALIGN} 的倍数……")
    sys_seg, _, sys_len = _pad_to_multiple(SYS, tok)
    c1_seg, _, c1_len = _pad_to_multiple(C1, tok)
    c2_seg, _, c2_len = _pad_to_multiple(C2, tok)
    c3_seg, _, c3_len = _pad_to_multiple(C3, tok)
    q_seg, _, q_len = _pad_to_multiple(Q, tok)
    print(f"    SYS={sys_len}  C1={c1_len}  C2={c2_len}  C3={c3_len}  Q={q_len}")

    # 非 PIC prompt: 直接拼接(SEP 在 split_and_tokenize 里不产生 token, 所以
    # 加与不加 SEP 得到的 token 序列一致)
    test_prompt = f"{sys_seg}{c1_seg}{c2_seg}{c3_seg}{q_seg}"
    w1 = f"{sys_seg}{c1_seg}{q_seg}"
    w3 = f"{sys_seg}{c3_seg}{q_seg}"
    # PIC prompt: 段间插 SEP
    pic_test = f"{sys_seg}{SEP}{c1_seg}{SEP}{c2_seg}{SEP}{c3_seg}{SEP}{q_seg}"
    pic_w1 = f"{sys_seg}{SEP}{c1_seg}{SEP}{q_seg}"
    pic_w3 = f"{sys_seg}{SEP}{c3_seg}{SEP}{q_seg}"

    # C3 在 test prompt 里的位置 (以完整 tokenized 序列的位置为准):
    #   SYS[0:sys_len)  C1[sys_len:sys_len+c1_len)  C2[.., .. + c2_len)  C3[.., .. + c3_len)  Q[..)
    c3_start = sys_len + c1_len + c2_len
    c3_end = c3_start + c3_len
    total = sys_len + c1_len + c2_len + c3_len + q_len

    return {
        "test": test_prompt,
        "w1": w1,
        "w3": w3,
        "pic_test": pic_test,
        "pic_w1": pic_w1,
        "pic_w3": pic_w3,
        "c3_range": (c3_start, c3_end),
        "c3_len": c3_len,
        "total": total,
    }


# ─────────────────────────────────────────────────────────────────
# 各模式的 Engine 启动配置
# ─────────────────────────────────────────────────────────────────

# 与 quick_test_online.py 一致的 --pic-enable 通用参数
def _pic_common_kwargs(sep: str) -> Dict:
    return {
        "pic_enable": True,
        "page_size": 1,
        "chunked_prefill_size": -1,
        "pic_separator_str": sep,
    }


# 与 quick_test_online.py 一致的 A3 / CacheBlend 通用参数(禁 cuda_graph
# / overlap / radix cache,与 baseline A3 保持一致)
_A3_COMMON_KWARGS = dict(
    disable_cuda_graph=True,
    disable_overlap_schedule=True,
    disable_radix_cache=True,
    chunked_prefill_size=-1,
)


def _mode_engine_kwargs(mode: str) -> Dict:
    """返回该模式给 sglang.Engine 用的 kwargs(不含 model_path / tp_size 等)。"""
    if mode == "full_recompute":
        # 与 quick_test_online.py 的 full_recompute 保持一致的 flags(禁一切优化)
        return dict(
            disable_radix_cache=True,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            chunked_prefill_size=-1,
        )
    if mode == "pic":
        return _pic_common_kwargs(SEP)
    if mode == "pic_a3":
        return {**_pic_common_kwargs(SEP), **_A3_COMMON_KWARGS, "enable_a3": True}
    if mode == "pic_cacheblend":
        return {**_pic_common_kwargs(SEP), **_A3_COMMON_KWARGS, "enable_cacheblend": True}
    raise ValueError(f"unknown mode: {mode}")


def _mode_prompts(mode: str, prompts: Dict) -> Tuple[str, List[str]]:
    """返回 (test_prompt, [warmup_prompts])。"""
    if mode == "full_recompute":
        # baseline 不需要 warmup(cache 是空的)
        return prompts["test"], []
    if mode in ("pic", "pic_a3", "pic_cacheblend"):
        # PIC 系列需要用 PIC prompt(段间有 SEP), warmup 命中 C1 + C3
        return prompts["pic_test"], [prompts["pic_w1"], prompts["pic_w3"]]
    raise ValueError(f"unknown mode: {mode}")


# ─────────────────────────────────────────────────────────────────
# 捕获(用 SGLANG_PIC_POOL_DUMP_DIR env hook 跑 sglang.Engine)
# ─────────────────────────────────────────────────────────────────

def run_capture(
    mode: str,
    model: str,
    tp: int,
    prompts: Dict,
    dump_dir: str,
    mem_frac: float,
) -> None:
    """一个模式起一次 Engine:先跑 warmup(命中 PIC), 再跑 test 请求 dump kv_pool。
    dump 结果落到 dump_dir。

    关键: SGLANG_PIC_POOL_DUMP_DIR 必须在 sglang.Engine 构造 **之前** 就设进
    os.environ, 因为 TP worker 是 Engine 构造时 spawn 的子进程 —— 子进程只
    继承 spawn 那一刻的 env, 之后父进程再改 env 子进程看不到。这也是为什么
    warmup 也会 dump: 我们靠 test 是最后一个请求, 相同文件名覆盖 → 最终文件
    反映 test 请求的 KV 状态。
    """
    os.makedirs(dump_dir, exist_ok=True)
    for f in os.listdir(dump_dir):
        if _SHARD_RE.match(f):
            os.remove(os.path.join(dump_dir, f))

    # ── env 必须在 Engine 构造前设 (TP worker 子进程 spawn 时继承) ──
    os.environ["SGLANG_PIC_POOL_DUMP_DIR"] = dump_dir
    # PIC / A3 需要的额外 env(quick_test_online.py 里也是这样注入)
    if mode in ("pic", "pic_a3", "pic_cacheblend"):
        os.environ["SGLANG_EAGER_INPUT_NO_COPY"] = "1"
        os.environ["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"
    else:
        os.environ.pop("SGLANG_EAGER_INPUT_NO_COPY", None)
        os.environ.pop("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE", None)

    test_prompt, warmup_prompts = _mode_prompts(mode, prompts)
    engine_kwargs = _mode_engine_kwargs(mode)

    print(f"  [engine] mode={mode}  tp={tp}  dump_dir={dump_dir}")
    print(f"           kwargs={engine_kwargs}")

    from sglang import Engine
    engine = Engine(
        model_path=model,
        tp_size=tp,
        trust_remote_code=True,
        mem_fraction_static=mem_frac,
        disable_piecewise_cuda_graph=True,
        **engine_kwargs,
    )
    try:
        # 1. warmup:填满 PIC 段缓存(baseline 模式跳过)。warmup 会 dump 但会
        #    被下一次请求 (最后是 test) 用相同文件名覆盖。
        for i, wp in enumerate(warmup_prompts, 1):
            print(f"  [warmup {i}/{len(warmup_prompts)}] {len(wp)} 字符")
            engine.generate(
                prompt=wp,
                sampling_params={"max_new_tokens": 4, "temperature": 0.0},
            )

        # 2. 真正 test 请求 —— 这是最后一个请求, dump 文件最终反映 test 状态
        t0 = time.perf_counter()
        engine.generate(
            prompt=test_prompt,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        )
        print(f"  [engine] prefill done in {time.perf_counter()-t0:.2f}s")
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass
        # dump env 在下个 mode 会被重新设置; 这里 pop 避免残留影响 shutdown 逻辑
        os.environ.pop("SGLANG_PIC_POOL_DUMP_DIR", None)

    # 给 TP worker 一点时间刷 torch.save
    time.sleep(1.5)
    gc.collect()
    torch.cuda.empty_cache()

    # ── sanity check: dump 目录里应该有 .pt 文件 ──
    _shards = [f for f in os.listdir(dump_dir) if _SHARD_RE.match(f)]
    if not _shards:
        raise RuntimeError(
            f"[{mode}] Engine 跑完了但 {dump_dir} 里没有 dump 文件。\n"
            f"        可能原因:\n"
            f"          1. SGLANG_PIC_POOL_DUMP_DIR env 没传到 TP worker 子进程\n"
            f"             (必须在 sglang.Engine 构造之前 os.environ 设好)\n"
            f"          2. _pic_writeback_mla_kv 里的 dump 钩子被异常吞了\n"
            f"             (查 sglang 服务器日志 grep 'PIC pool dump failed')\n"
            f"          3. 请求根本没走 extend 分支\n"
            f"             (dump 钩子有 forward_mode.is_extend() 守卫)"
        )
    print(f"  [dump-check] {dump_dir}: {len(_shards)} 个 .pt 文件")


def load_rank0_latent(dump_dir: str, num_layers: int, expected_T: int) -> torch.Tensor:
    """加载 rank0 每层 latent,拼成 (num_layers, T, D)。

    MQA(num_kv_heads=1)→ KV 在 rank 间是复制而不是切分,只读 rank0 即可。
    expected_T: 期望的 seq_len(用来做一致性检查)。

    Shape 处理: kv_pool.get_key_buffer(layer_id) 对 MLA 是 3D
    (total_slots, 1, kv_lora_rank+qk_rope_head_dim), dump 出来也是 3D
    (T, 1, D)。这里把大小为 1 的维度都 squeeze 掉,统一成 2D (T, D)。
    """
    def _to_2d(t: torch.Tensor) -> torch.Tensor:
        # 只 squeeze 大小为 1 的中间维;末维 (kv_dim) 一般 >= 512, 不会被误伤
        while t.ndim > 2:
            # 找第一个 size==1 的维 (通常是 num_kv_heads=1) squeeze 掉
            squeezed = False
            for d in range(t.ndim):
                if t.shape[d] == 1:
                    t = t.squeeze(d)
                    squeezed = True
                    break
            if not squeezed:
                raise ValueError(
                    f"cannot reduce to 2D: shape={tuple(t.shape)}"
                )
        return t

    per_layer: Dict[int, torch.Tensor] = {}
    for fname in os.listdir(dump_dir):
        m = _SHARD_RE.match(fname)
        if not m or int(m["rank"]) != 0:
            continue
        layer_id = int(m["layer"])
        t = torch.load(os.path.join(dump_dir, fname), map_location="cpu")
        per_layer[layer_id] = _to_2d(t)

    if not per_layer:
        raise RuntimeError(f"no rank0 latent shards in {dump_dir}")

    first = per_layer[min(per_layer)]
    if first.ndim != 2:
        raise ValueError(
            f"{dump_dir}: expected 2D tensor after squeeze, got shape "
            f"{tuple(first.shape)}"
        )
    T, D = first.shape
    if T != expected_T:
        print(f"  [warn] {dump_dir}: layer0 T={T} != expected {expected_T} "
              f"(前一个请求 KV 可能被截断; 继续用实际 T)",
              file=sys.stderr)

    out = torch.zeros((num_layers, T, D), dtype=first.dtype)
    for lid, t in per_layer.items():
        if t.shape != (T, D):
            print(f"  [warn] layer{lid} shape {tuple(t.shape)} != {(T, D)},跳过",
                  file=sys.stderr)
            continue
        out[lid] = t
    return out


# ─────────────────────────────────────────────────────────────────
# 误差计算 + 绘图
# ─────────────────────────────────────────────────────────────────

def compute_c3_errors(
    lat_full: torch.Tensor,          # (L, T, D)  baseline (full_recompute)
    lat_mode: torch.Tensor,          # (L, T, D)
    c3_range: Tuple[int, int],
    kv_lora_rank: int,
    separate_nope_pe: bool = False,
) -> Dict[str, np.ndarray]:
    s, e = c3_range
    if lat_full.shape[1] < e or lat_mode.shape[1] < e:
        raise ValueError(
            f"C3 range [{s}, {e}) exceeds seq_len: "
            f"full={lat_full.shape[1]}, mode={lat_mode.shape[1]}"
        )
    a = lat_full[:, s:e, :].float()
    b = lat_mode[:, s:e, :].float()

    diff = a - b                             # (L, N_C3, D)
    l2 = diff.pow(2).sum(dim=-1).sqrt()      # (L, N_C3)

    out: Dict[str, np.ndarray] = {
        "l2_per_layer_per_token": l2.numpy(),
        "mean_per_layer": l2.mean(dim=-1).numpy(),
        "max_per_layer": l2.max(dim=-1).values.numpy(),
    }

    if separate_nope_pe:
        a_nope, a_pe = a[..., :kv_lora_rank], a[..., kv_lora_rank:]
        b_nope, b_pe = b[..., :kv_lora_rank], b[..., kv_lora_rank:]
        out["l2_nope_per_layer_per_token"] = (
            (a_nope - b_nope).pow(2).sum(dim=-1).sqrt().numpy()
        )
        out["l2_pe_per_layer_per_token"] = (
            (a_pe - b_pe).pow(2).sum(dim=-1).sqrt().numpy()
        )
    return out


# 图例颜色(3 个非 baseline 模式)
_MODE_COLORS = {
    "pic": "#1f77b4",
    "pic_a3": "#d62728",
    "pic_cacheblend": "#2ca02c",
}


def plot(
    errors: Dict[str, Dict[str, np.ndarray]],
    output: str,
    c3_len: int,
    separate_nope_pe: bool = False,
) -> None:
    """
    errors: {mode_name: compute_c3_errors 的返回值},非 baseline 模式各一个。
    图布局:
      Row 1: 3 个模式 per-layer mean L2 折线叠加
      Row 2..k: 每个模式一张 heatmap (layer × C3 token)
      Row -1(可选): nope vs pe 分量 mean 曲线
    """
    import matplotlib.pyplot as plt

    modes = [m for m in NON_BASELINE if m in errors]
    if not modes:
        print("  [plot] 没有非 baseline 模式的误差数据,不出图")
        return

    L = errors[modes[0]]["mean_per_layer"].shape[0]
    n_rows = 1 + len(modes) + (1 if separate_nope_pe else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows),
                              constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    # ── Row 1: per-layer mean L2 曲线叠加 ──
    ax = axes[0]
    layer_ids = np.arange(L)
    for m in modes:
        ax.plot(
            layer_ids, errors[m]["mean_per_layer"],
            label=f"{m}  (mean)", marker="o", markersize=3,
            color=_MODE_COLORS.get(m),
        )
        # max 用相同颜色的虚线
        ax.plot(
            layer_ids, errors[m]["max_per_layer"],
            label=f"{m}  (max)", marker=None, linestyle="--", alpha=0.6,
            color=_MODE_COLORS.get(m),
        )
    ax.set_xlabel("Layer index")
    ax.set_ylabel("L2 error  ||K_mode - K_full||_2")
    ax.set_title(
        f"C3 段 ({c3_len} tokens) KV 逐层误差 — {', '.join(modes)} vs full_recompute"
    )
    ax.legend(fontsize=8, ncol=len(modes))
    ax.grid(True, alpha=0.3)
    # log-Y 让层间差异更清晰
    means = np.concatenate([errors[m]["mean_per_layer"] for m in modes])
    if means[means > 0].size > 0 and means.max() / max(means[means > 0].min(), 1e-9) > 100:
        ax.set_yscale("symlog", linthresh=1e-6)

    # 全 0 就在图上标出来(用户问的"是否有误差"直接可见)
    for m in modes:
        if float(errors[m]["max_per_layer"].max()) == 0.0:
            ax.text(
                0.02, 0.95 - 0.05 * modes.index(m),
                f"{m}: ALL ZERO (完全匹配 baseline)",
                transform=ax.transAxes,
                color=_MODE_COLORS.get(m),
                fontsize=10, fontweight="bold",
            )

    # ── Row 2..: 每个模式一张 heatmap ──
    for i, m in enumerate(modes):
        ax = axes[1 + i]
        hm = errors[m]["l2_per_layer_per_token"]
        vmax = float(hm.max()) if hm.max() > 0 else 1.0
        im = ax.imshow(
            hm, aspect="auto", cmap="viridis", origin="lower",
            extent=[0, c3_len, 0, L], vmin=0, vmax=vmax,
        )
        ax.set_xlabel("Position within C3 (token index)")
        ax.set_ylabel("Layer index")
        ax.set_title(
            f"[{m}] heatmap  layer × C3-token  (vmax={vmax:.4g})"
        )
        plt.colorbar(im, ax=ax, label="L2 error")

    # ── 可选 Row: nope vs pe 分量 mean 曲线 ──
    if separate_nope_pe:
        ax = axes[-1]
        for m in modes:
            e = errors[m]
            m_nope = e["l2_nope_per_layer_per_token"].mean(axis=-1)
            m_pe = e["l2_pe_per_layer_per_token"].mean(axis=-1)
            ax.plot(layer_ids, m_nope,
                     label=f"{m} k_nope", marker="o", markersize=3,
                     color=_MODE_COLORS.get(m))
            ax.plot(layer_ids, m_pe,
                     label=f"{m} k_pe", marker="s", markersize=3, linestyle="--",
                     color=_MODE_COLORS.get(m))
        ax.set_xlabel("Layer index")
        ax.set_ylabel("mean L2 error")
        ax.set_title(
            "Nope vs PE 分量误差 "
            "(nope=context drift only; pe 含 RoPE 位置误差)"
        )
        ax.legend(fontsize=8, ncol=len(modes))
        ax.grid(True, alpha=0.3)

    fig.savefig(output, dpi=120, bbox_inches="tight")
    print(f"\n  [plot] 已保存 → {output}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="4-mode PIC KV error diagnostic "
                     "(full_recompute vs pic / pic_a3 / pic_cacheblend)"
    )
    parser.add_argument("--model", default="/workspace/models/GLM-5.2-FP8")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--mem-fraction", type=float, default=0.82)
    parser.add_argument("--output", default="pic_mode_kv_error.png")
    parser.add_argument(
        "--modes", nargs="+", default=list(MODES_ALL),
        choices=list(MODES_ALL),
        help="要跑的模式列表(至少要包含 full_recompute 做 baseline)",
    )
    parser.add_argument(
        "--dump-root", default="/tmp/errmodes",
        help="pool dump 根目录(每个 mode 一个子目录: <root>_<mode>)",
    )
    parser.add_argument(
        "--skip-capture", action="store_true",
        help="跳过 Engine 起 4 次,直接从已有 dump 目录读并画图",
    )
    parser.add_argument("--keep-shards", action="store_true",
                         help="跑完不清理 dump 目录")
    parser.add_argument("--separate-nope-pe", action="store_true",
                         help="额外画一个 nope vs pe 分量的曲线子图")
    parser.add_argument("--csv", type=str, default=None,
                         help="导出 per-layer mean/max/p95 到 CSV")
    args = parser.parse_args()

    if "full_recompute" not in args.modes:
        print("[ERROR] --modes 必须包含 full_recompute (作为 baseline)",
              file=sys.stderr)
        sys.exit(2)
    non_baseline = [m for m in args.modes if m != "full_recompute"]
    if not non_baseline:
        print("[ERROR] 除 full_recompute 外还需至少一个对比模式",
              file=sys.stderr)
        sys.exit(2)

    print(f"\n{'#'*64}")
    print(f"  4-mode PIC KV error diagnostic")
    print(f"  Model: {args.model}   TP={args.tp}")
    print(f"  Modes: {args.modes}")
    print(f"{'#'*64}\n")

    # 1. 构造 prompts + C3 位置
    p = build_prompts(args.model)
    print(f"  test prompt: {p['total']} tokens, "
          f"C3 range=[{p['c3_range'][0]}, {p['c3_range'][1]})")

    # 2. 各模式跑一次 Engine + capture
    dump_dirs = {m: f"{args.dump_root}_{m}" for m in args.modes}
    if not args.skip_capture:
        for mode in args.modes:
            print(f"\n[Run {mode}]")
            run_capture(mode, args.model, args.tp, p, dump_dirs[mode],
                          args.mem_fraction)
    else:
        print(f"\n  [--skip-capture] 复用已有 dump 目录: {list(dump_dirs.values())}")

    # 3. 读回 rank0 latent
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    num_layers = cfg.num_hidden_layers
    kv_lora_rank = getattr(cfg, "kv_lora_rank", None)
    print(f"\n  [config] num_layers={num_layers}  kv_lora_rank={kv_lora_rank}")

    lats: Dict[str, torch.Tensor] = {}
    for mode in args.modes:
        print(f"  [load] {mode:<15} ← {dump_dirs[mode]}")
        lat = load_rank0_latent(dump_dirs[mode], num_layers, p["total"])
        print(f"           shape={tuple(lat.shape)} dtype={lat.dtype}")
        lats[mode] = lat

    # 4. 逐层 C3 段误差(baseline = full_recompute)
    baseline = lats["full_recompute"]
    print(f"\n  [compute] baseline=full_recompute; C3 range={p['c3_range']}")
    errors: Dict[str, Dict[str, np.ndarray]] = {}
    for mode in non_baseline:
        try:
            errors[mode] = compute_c3_errors(
                baseline, lats[mode], p["c3_range"],
                kv_lora_rank=kv_lora_rank or 0,
                separate_nope_pe=args.separate_nope_pe,
            )
        except ValueError as e:
            print(f"  [warn] mode={mode}: {e}; 跳过")

    # 5. 打印摘要
    print(f"\n  逐层误差摘要 ({len(non_baseline)} 个模式):")
    header = f"  {'layer':>5}"
    for m in non_baseline:
        header += f"  {m+'/mean':>16}  {m+'/max':>16}"
    print(header)
    for i in range(num_layers):
        row = f"  {i:>5}"
        for m in non_baseline:
            if m not in errors:
                row += f"  {'-':>16}  {'-':>16}"
                continue
            row += (f"  {errors[m]['mean_per_layer'][i]:>16.4g}"
                     f"  {errors[m]['max_per_layer'][i]:>16.4g}")
        print(row)

    # 6. 全 0 判定 & 打印
    print(f"\n  是否有位置编码误差?")
    for m in non_baseline:
        if m not in errors:
            print(f"    {m:<16}: [skip]")
            continue
        max_err = float(errors[m]["max_per_layer"].max())
        mean_err = float(errors[m]["mean_per_layer"].mean())
        if max_err == 0.0:
            print(f"    {m:<16}: ✓ 完全匹配 (max=0, mean=0)")
        else:
            print(f"    {m:<16}: ✗ 有误差  max={max_err:.4g}  "
                  f"整体 mean={mean_err:.4g}")

    # 7. CSV(可选)
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            head = ["layer"]
            for m in non_baseline:
                head += [f"{m}/mean", f"{m}/max", f"{m}/p95"]
            w.writerow(head)
            p95_all = {
                m: np.percentile(errors[m]["l2_per_layer_per_token"], 95, axis=-1)
                for m in non_baseline if m in errors
            }
            for i in range(num_layers):
                row = [i]
                for m in non_baseline:
                    if m not in errors:
                        row += ["", "", ""]
                        continue
                    row += [
                        float(errors[m]["mean_per_layer"][i]),
                        float(errors[m]["max_per_layer"][i]),
                        float(p95_all[m][i]),
                    ]
                w.writerow(row)
        print(f"\n  [csv] 已导出 → {args.csv}")

    # 8. 出图
    plot(errors, args.output, c3_len=p["c3_len"],
          separate_nope_pe=args.separate_nope_pe)

    # 9. cleanup
    if not args.keep_shards and not args.skip_capture:
        for d in dump_dirs.values():
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if _SHARD_RE.match(f):
                    os.remove(os.path.join(d, f))
        print(f"  [cleanup] 清理 dump 目录(--keep-shards 保留)")


if __name__ == "__main__":
    main()
