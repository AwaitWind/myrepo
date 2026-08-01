# Copyright 2025-2026 SGLang Team
#
# A³ (Attention-Aware Approximate Acceleration) / CacheBlend
# 选择性重算 Token 的工具函数集合。
#
# 本模块为**全新增文件**，不修改任何现有代码路径。只有在显式开启
# `--enable-a3` / `--enable-cacheblend` 且请求携带 precomputed_kv 时，
# glm4_moe 的 reuse 分支才会调用这里的函数；默认情况下完全不生效。
#
# 目标模型：GLM5.2（glm4_moe 架构）。
#
# 参考文档：
#   - A3_CACHEBLEND_SGLANG_INTEGRATION.md
#   - A3_RECOMPUTE_TOKEN_LOGIC.md
#   - A3_GLM52_ADAPTATION_GUIDE.md
"""A³ / CacheBlend 重算 Token 选择与非对称注意力工具。"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# reuse 策略常量
REUSE_A3 = "debug"  # A³：基于 Query-Key 注意力分数
REUSE_BLEND = "blend"  # CacheBlend：基于 Value L2 差异

# 状态机常量
CHECK_NONE = None  # 第 0 层：全量计算
CHECK_CHECKING = "checking"  # 第 1 层：决策层
CHECK_POST = "postchecking"  # 第 2+ 层：KV 融合层

# 决策层固定为第 1 层（A³ 与 CacheBlend 共用）
CHECK_LAYER = 1


def get_check_layer(reuse_method: Optional[str]) -> int:
    """返回做重要 token 决策的层号。两种策略均为第 1 层。"""
    return CHECK_LAYER


def compute_attention_sum_wo_head(
    query_states: torch.Tensor,  # (1, num_heads, total_len, head_dim)
    key_states: torch.Tensor,  # (1, num_heads, context_len, head_dim)
    last_len: int,
) -> torch.Tensor:
    """A³ 策略：计算 query 区域对 context token 的注意力分数并求和。

    GLM5.2 适配说明：
    - 输入 Q/K 已经过 partial RoPE（factor=0.5），只有 head_dim 前 50% 旋转。
      这对注意力分数计算没有影响，直接使用即可。

    Returns:
        attn_weights_sum: shape (1, context_len)，float32
    """
    query_states = query_states.permute(0, 2, 1, 3).reshape(
        1, query_states.shape[2], -1
    )
    key_states = key_states.permute(0, 2, 1, 3).reshape(1, key_states.shape[2], -1)

    dim = key_states.shape[-1]
    attn_weights = (
        torch.matmul(query_states[:, -last_len:, :], key_states.transpose(1, 2))
        / math.sqrt(dim)
    )

    # 因果掩码（query 区域内部下三角）
    q_q_len = last_len
    mask = torch.full(
        (q_q_len, q_q_len),
        torch.finfo(attn_weights.dtype).min,
        device=attn_weights.device,
        dtype=attn_weights.dtype,
    )
    mask_cond = torch.arange(q_q_len, device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(q_q_len, 1), 0)
    attn_weights[:, -q_q_len:, -q_q_len:] = (
        attn_weights[:, -q_q_len:, -q_q_len:] + mask
    )

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    context_len = key_states.shape[1] - last_len
    attn_weights_sum = attn_weights[:, :, :context_len].sum(dim=1)  # (1, context_len)
    return attn_weights_sum.float()


def get_topindices(
    reuse_config: dict,
    query_states: torch.Tensor,  # (1, num_heads, total_len, head_dim)
    key_states: torch.Tensor,  # (1, num_kv_heads, total_len, head_dim)
    value_states: torch.Tensor,  # (1, num_kv_heads, total_len, head_dim)
    value_old: torch.Tensor,  # (1, num_kv_heads, total_len, head_dim) 预计算 V
    num_key_value_groups: int,  # GQA 分组数 = num_heads / num_kv_heads
) -> torch.Tensor:
    """根据策略类型，选择需要重算的 token 索引。

    Returns:
        top_indices: shape (imp_len,)，包含重要 token + query token 的索引（已排序）。
    """
    recomp_ratio = reuse_config["recomp_ratio"]
    last_len = reuse_config["last_len"]
    prefix_len = reuse_config.get("prefix_len", 0)
    total_len = value_states.shape[2]
    context_len = total_len - last_len

    topk_num = max(1, int(context_len * recomp_ratio))

    # query 区域索引（始终保留重算）
    last_indices = torch.arange(
        total_len - last_len, total_len, device=value_states.device
    )

    reuse_method = reuse_config["reuse"]

    if REUSE_BLEND in reuse_method:
        # ── CacheBlend：基于 Value L2 差异 ──
        diff = value_states[:, :, :context_len, :] - value_old[:, :, :context_len, :]
        temp_diff = torch.sum(diff ** 2, dim=[0, 1, 3])  # (context_len,)
        top_indices = torch.topk(temp_diff, k=topk_num).indices
        top_indices, _ = torch.sort(top_indices)

    elif REUSE_A3 in reuse_method:
        # ── A³：基于 Query-Key 注意力分数 ──
        # GLM5.2 GQA：repeat kv heads 对齐 query heads
        if num_key_value_groups > 1:
            key_for_attn = key_states.repeat_interleave(num_key_value_groups, dim=1)
        else:
            key_for_attn = key_states

        key_context = key_for_attn[:, :, prefix_len:context_len, :]
        attn_weights_sum = compute_attention_sum_wo_head(
            query_states, key_context, last_len
        )  # (1, context_len - prefix_len)

        # 平均池化平滑（窗口 5）
        attn_cache = F.avg_pool1d(
            attn_weights_sum, kernel_size=5, padding=2, stride=1
        )
        top_indices = (
            attn_cache.topk(topk_num, dim=-1).indices.squeeze(0) + prefix_len
        )
        top_indices, _ = torch.sort(top_indices)
    else:
        raise ValueError(
            f"Unknown reuse method: '{reuse_method}'. "
            f"Expected substring '{REUSE_BLEND}' or '{REUSE_A3}'."
        )

    return torch.cat([top_indices, last_indices])


def compute_prefill_attention_with_custom_kv(
    q: torch.Tensor,  # (1, num_heads, q_len, head_dim)，已做 partial RoPE
    k: torch.Tensor,  # (1, num_kv_heads, kv_len, head_dim)
    v: torch.Tensor,  # (1, num_kv_heads, kv_len, head_dim)
    num_kv_groups: int,
    scale: float,
    imp_indices: torch.Tensor,  # (q_len,)，重要 token 在 kv_len 中的位置
    total_len: int,
) -> torch.Tensor:
    """计算非对称 Q/KV 的注意力（q_len <= kv_len）。

    因果掩码：imp_indices[i] 位置的 query 只能 attend 到 position <= imp_indices[i]
    的 token。

    Returns:
        output: (q_len, num_heads * head_dim)，flat 格式，与 SGLang packed tensor 一致。
    """
    bsz, num_heads, q_len, head_dim = q.shape
    _, num_kv_heads, kv_len, _ = k.shape

    # GQA repeat
    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)

    # 因果掩码：(q_len, kv_len)，True = block（不允许 attend）
    kv_positions = torch.arange(kv_len, device=q.device)
    imp_pos = imp_indices.to(q.device).unsqueeze(1)  # (q_len, 1)
    causal_mask = kv_positions.unsqueeze(0) > imp_pos  # (q_len, kv_len)

    try:
        import flashinfer

        q_fi = q.permute(0, 2, 1, 3).squeeze(0)  # (q_len, num_heads, head_dim)
        k_fi = k.permute(0, 2, 1, 3).squeeze(0)  # (kv_len, num_heads, head_dim)
        v_fi = v.permute(0, 2, 1, 3).squeeze(0)  # (kv_len, num_heads, head_dim)
        attn_out = flashinfer.single_prefill_with_kv_cache(
            q_fi,
            k_fi,
            v_fi,
            causal=False,
            custom_mask=~causal_mask,  # True = allow attend
        )  # (q_len, num_heads, head_dim)
        output = attn_out.reshape(q_len, num_heads * head_dim)
    except Exception:
        # PyTorch fallback（无 flashinfer 或接口不兼容时）
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask_val = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), mask_val
        )
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_weights, v)  # (1, num_heads, q_len, head_dim)
        output = attn_out.permute(0, 2, 1, 3).reshape(q_len, num_heads * head_dim)

    return output  # (q_len, num_heads * head_dim)


def align_precomputed_key_rope(
    rotary_emb,
    org_positions: torch.Tensor,  # (total_len,) flat
    key_old_4d: torch.Tensor,  # (1, num_kv_heads, total_len, head_dim)
    num_kv_heads: int,
    head_dim: int,
    fake_q_flat: torch.Tensor,  # (total_len, num_heads * head_dim)
) -> torch.Tensor:
    """对预计算的旧 K 使用原始完整位置做 partial RoPE 对齐。

    GLM5.2 的 rotary_emb 接受 flat 格式: (total_len,) + (total_len, dim)。
    fake_q 仅用于驱动 rotary_emb 接口，不参与真实计算。

    Returns:
        key_old_roped_4d: (1, num_kv_heads, total_len, head_dim)
    """
    total_len = org_positions.shape[0]
    key_old_flat = key_old_4d.permute(0, 2, 1, 3).reshape(total_len, -1)
    _, key_old_roped_flat = rotary_emb(org_positions, fake_q_flat, key_old_flat)
    return key_old_roped_flat.view(1, total_len, num_kv_heads, head_dim).permute(
        0, 2, 1, 3
    )


# ============================================================================
# DSA/MLA-native latent-space imp selection (pic_a3 new path, see
# scripts/pic_a3_full_recompute_plan.md §3.2 / §3.10)
# ----------------------------------------------------------------------------
# Mathematical justification for latent-space scoring on MLA:
#     per-head logit = q_nope · k_nope + q_pe · k_pe
#                    = q_nope · (W_UK · c_kv) + q_pe · k_pe
#                    = (q_nope · W_UK) · c_kv + q_pe · k_pe
#                    = q_absorbed · c_kv + q_pe · k_pe
# where q_absorbed is precisely forward_mla.py:459's `q_nope_out`. So this
# scoring is NOT an approximation — it is mathematically identical to
# per-head Q · K softmax logit (associativity is exact). No kv_b_proj
# reconstruction needed.
# ============================================================================


def pick_imp_latent_a3(
    q_absorbed: torch.Tensor,     # (N, H, D_lat)  = q_nope_out
    k_latent: torch.Tensor,       # (N, 1, D_lat)  = latent_cache[..., :kv_lora_rank]
    q_pe: torch.Tensor,           # (N, H, D_pe)
    k_pe: torch.Tensor,           # (N, 1, D_pe)   (post-RoPE at fresh positions)
    softmax_scale: float,         # kept for API compat (see note below), unused
    last_len: int,
    recomp_ratio: float,
    prefix_len: int = 0,
) -> torch.Tensor:
    """A³ imp selection in MLA latent space. Merged-head softmax to align
    with the existing MHA-a3 impl (compute_attention_sum_wo_head, reuse_utils.py:44).

    Returns imp_indices: (imp_len,) sorted ascending, imp_len = topk_num + last_len.

    ⚠ Scale note: earlier design used per-head `self.scaling = 1/sqrt(qk_head_dim)`,
    but merged-head matmul aggregates H heads into a single dim → correct
    softmax scale is `1/sqrt(H*(D_lat+D_pe))`, matching how
    compute_attention_sum_wo_head (:64) scales by `1/sqrt(dim)` with
    `dim = key_states.shape[-1] = H*d`. Using the per-head scale on a
    merged-head matmul makes softmax too sharp → topk degenerates.
    """
    N, H, D_lat = q_absorbed.shape
    D_pe = q_pe.shape[-1]

    # Build per-head Q/K (broadcast latent K to H heads)
    q_full = torch.cat([q_absorbed, q_pe], dim=-1)           # (N, H, D_lat+D_pe)
    k_nope = k_latent.expand(-1, H, -1)                      # (N, H, D_lat)
    k_rope = k_pe.expand(-1, H, -1)                          # (N, H, D_pe)
    k_full = torch.cat([k_nope, k_rope], dim=-1)             # (N, H, D_lat+D_pe)

    # Merge heads (align to compute_attention_sum_wo_head at :58-61 which does
    # permute+reshape → single softmax over merged-dim).
    D = H * (D_lat + D_pe)
    q_merged = q_full.reshape(N, D)                          # (N, H*d)
    k_merged = k_full.reshape(N, D)                          # (N, H*d)

    # Correct scale for merged-head matmul softmax: 1/sqrt(D_merged) where
    # D_merged = H * (D_lat + D_pe). NOT the model's per-head self.scaling
    # (which would make softmax too sharp — see docstring).
    merged_scale = 1.0 / math.sqrt(D)
    score = torch.matmul(q_merged, k_merged.transpose(0, 1)) * merged_scale  # (N, N)

    causal = torch.triu(
        torch.full((N, N), float("-inf"), device=score.device, dtype=score.dtype),
        diagonal=1,
    )
    score = score + causal
    score = F.softmax(score.float(), dim=-1).to(score.dtype)

    # query region (last last_len rows) attends over context region → attention mass
    context_len = N - last_len
    attn_sum = score[-last_len:, prefix_len:context_len].sum(dim=0)  # (context_len - prefix_len,)
    # TP consistency (pic_a3 multi-rank): each attention-TP rank holds only a
    # SUBSET of heads, so its attn_sum reflects only those heads → each rank
    # would pick a DIFFERENT imp set → inconsistent KV routing across ranks →
    # corrupted all-reduced output. Sum attn_sum across the attention-TP group
    # so every rank sees the GLOBAL per-head attention mass and selects the SAME
    # imp set. (Post-softmax sum: guarantees identical topk across ranks; not
    # bit-identical to the single-GPU merged-head score, which would need an
    # all-reduce of the pre-softmax logits instead.)
    try:
        from sglang.srt.layers.dp_attention import (
            attn_tp_all_reduce,
            get_attention_tp_size,
        )

        if get_attention_tp_size() > 1:
            attn_sum = attn_tp_all_reduce(attn_sum.contiguous())
    except Exception:
        pass
    # Guard: topk_num must be <= attn_sum length. Ratio is over the full
    # context but attn_sum excludes prefix. Also, if last_len covers everything
    # (context_len == 0 or context_len <= prefix_len), fall back to "no top-k,
    # only last_indices" — i.e. imp == miss.
    _avail = int(attn_sum.numel())
    _requested = max(1, int(context_len * recomp_ratio))
    topk_num = min(_requested, _avail)
    import logging as _lg
    _lg.getLogger(__name__).warning(
        f"[PIC-A3-IMP] N={N} last_len={last_len} prefix_len={prefix_len} "
        f"context_len={context_len} recomp_ratio={recomp_ratio} "
        f"_requested={_requested} _avail={_avail} topk_num={topk_num}"
    )

    last_indices = torch.arange(context_len, N, device=score.device)
    if topk_num <= 0 or _avail == 0:
        # Degenerate case: no hit context to pick from — imp is just the query region
        return torch.sort(last_indices)[0]

    # ragkv-style local smoothing (A3_IMPORTANT_TOKEN_SELECTION.md §3): avg_pool1d
    # over the attention-mass scores BEFORE top-k → prefer CONTIGUOUS important
    # spans (answers are usually a contiguous text span) instead of isolated peaks.
    # kernel_size from env (default 5 = ragkv); <=1 disables. Applied AFTER the TP
    # all-reduce so every rank smooths the same global attn_sum → identical top-k.
    import os as _os_sm

    _ks = int(_os_sm.environ.get("SGLANG_PIC_A3_IMP_SMOOTH_K", "5"))
    _scores = attn_sum
    if _ks > 1 and _avail >= _ks:
        _smoothed = F.avg_pool1d(
            attn_sum.float().reshape(1, 1, -1),
            kernel_size=_ks, padding=_ks // 2, stride=1,
        ).reshape(-1)
        if int(_smoothed.numel()) == int(attn_sum.numel()):  # odd k → same length
            _scores = _smoothed
    top_indices = _scores.topk(topk_num).indices + prefix_len
    return torch.sort(torch.cat([top_indices, last_indices]))[0]


def pick_imp_latent_a3_hit_only(
    q_absorbed: torch.Tensor,     # (N, H, D_lat)  = q_nope_out
    k_latent: torch.Tensor,       # (N, 1, D_lat)  = latent_cache[..., :kv_lora_rank]
    q_pe: torch.Tensor,           # (N, H, D_pe)
    k_pe: torch.Tensor,           # (N, 1, D_pe)
    softmax_scale: float,         # unused, kept for API symmetry with pick_imp_latent_a3
    last_len: int,
    recomp_ratio: float,
    hit_segments,                 # Iterable[Tuple[int, int]] in forward-batch positions
    prefix_len: int = 0,
    page_align: int = 64,
) -> torch.Tensor:
    """A³ imp selection restricted to hit segments, per-hit-segment percentage.

    Semantic: imp = subset of HIT tokens whose stale K/V drift most impacts the
    query, so they must be recomputed fresh at layer 2+. Miss tokens are always
    recomputed regardless, so they never enter the topk pool.

    Difference vs pick_imp_latent_a3:
      • K columns eligible for topk = union(hit_segments) ∩ [prefix_len, context_len)
        instead of the full [prefix_len : context_len) range (which lets
        middle-miss tokens compete for and steal imp slots).
      • Budget is per hit segment: imp_k = ceil(seg_len * recomp_ratio),
        page-aligned to `page_align` (default 64, matches
        SGLANG_PIC_A3_IMP_ONLY=1 static picker and DSA page size). Prevents
        Q-mass concentration on one segment from starving the others.

    Softmax denominator stays over the full context (miss tokens remain
    competitors in the attention distribution — attention mass on hit tokens
    is preserved). Only the topk index set is restricted to hit ranges.

    Returns imp_indices: sorted ascending, contains per-segment top picks +
    the entire query region (last_indices). The caller unions with miss
    positions.
    """
    N, H, D_lat = q_absorbed.shape
    D_pe = q_pe.shape[-1]

    # Merged-head construction identical to pick_imp_latent_a3.
    q_full = torch.cat([q_absorbed, q_pe], dim=-1)
    k_nope = k_latent.expand(-1, H, -1)
    k_rope = k_pe.expand(-1, H, -1)
    k_full = torch.cat([k_nope, k_rope], dim=-1)

    D = H * (D_lat + D_pe)
    q_merged = q_full.reshape(N, D)
    k_merged = k_full.reshape(N, D)

    merged_scale = 1.0 / math.sqrt(D)
    score = torch.matmul(q_merged, k_merged.transpose(0, 1)) * merged_scale

    causal = torch.triu(
        torch.full((N, N), float("-inf"), device=score.device, dtype=score.dtype),
        diagonal=1,
    )
    score = score + causal
    score = F.softmax(score.float(), dim=-1).to(score.dtype)

    context_len = N - last_len
    # Full-context softmax → per-position attention mass from the query rows.
    # Not renormalized over hit-only K, so per-hit-position mass stays
    # comparable to what layer-2+ attention would see.
    attn_full = score[-last_len:, :].sum(dim=0).float()  # (N,)
    # TP consistency: sum per-head attention mass across the attention-TP group
    # so all ranks select the SAME imp set (see pick_imp_latent_a3 for rationale).
    try:
        from sglang.srt.layers.dp_attention import (
            attn_tp_all_reduce,
            get_attention_tp_size,
        )

        if get_attention_tp_size() > 1:
            attn_full = attn_tp_all_reduce(attn_full.contiguous())
    except Exception:
        pass

    last_indices = torch.arange(context_len, N, device=score.device)

    picked_lists = []
    seg_dbg = []
    for (s, e) in hit_segments:
        s_c = max(int(s), int(prefix_len))
        e_c = min(int(e), int(context_len))
        seg_len = e_c - s_c
        if seg_len <= 0:
            seg_dbg.append((int(s), int(e), 0, 0))
            continue
        imp_k = max(1, int(math.ceil(seg_len * recomp_ratio)))
        if page_align > 1:
            imp_k = ((imp_k + page_align - 1) // page_align) * page_align
        imp_k = min(imp_k, seg_len)
        seg_scores = attn_full[s_c:e_c]
        top = seg_scores.topk(imp_k).indices + s_c
        picked_lists.append(top)
        seg_dbg.append((int(s), int(e), seg_len, imp_k))

    import logging as _lg
    _lg.getLogger(__name__).warning(
        f"[PIC-A3-HIT-ONLY-IMP] N={N} last_len={last_len} prefix_len={prefix_len} "
        f"context_len={context_len} recomp_ratio={recomp_ratio} "
        f"page_align={page_align} hit_segs=(s,e,seg_len,imp_k)={seg_dbg}"
    )

    if not picked_lists:
        return torch.sort(last_indices)[0]

    top_indices = torch.cat(picked_lists)
    return torch.sort(torch.cat([top_indices, last_indices]))[0]


def pick_imp_latent_blend(
    latent_new: torch.Tensor,     # (N, 1, D_lat + D_pe)  full latent, fresh
    latent_old: torch.Tensor,     # (N, 1, D_lat + D_pe)  from PICache
    kv_lora_rank: int,            # split point: [:kv_lora_rank] = k_nope
    last_len: int,
    recomp_ratio: float,
    prefix_len: int = 0,
) -> torch.Tensor:
    """CacheBlend imp selection in MLA latent space. Diff only on k_nope
    (`[:kv_lora_rank]`) — k_pe part encodes RoPE at write-time positions,
    would give position-noise not content-diff (see plan §7.2 Q2).

    In MLA, V is also derived from latent (`W_UV · c_kv`), so latent-diff on
    the k_nope slice serves as a proxy for V-diff without needing reconstruction.
    """
    N, _, _ = latent_new.shape
    context_end = N - last_len

    # Diff only on the k_nope (content) part; skip k_pe (position-encoded)
    new_nope = latent_new[prefix_len:context_end, 0, :kv_lora_rank]
    old_nope = latent_old[prefix_len:context_end, 0, :kv_lora_rank]
    diff = (new_nope - old_nope).float()
    dist = (diff ** 2).sum(dim=-1)                            # (context_len - prefix_len,)

    topk_num = max(1, int((N - last_len) * recomp_ratio))
    top_indices = dist.topk(topk_num).indices + prefix_len

    last_indices = torch.arange(context_end, N, device=latent_new.device)
    return torch.sort(torch.cat([top_indices, last_indices]))[0]


def read_layer_latent_from_pic_public(
    kv_buffer: torch.Tensor,          # get_key_buffer(layer_id), shape (num_slots+pad, 1, D)
    hit_pub_slots: torch.Tensor,      # (N,) int64 — public slot per position, -1 for non-hit
) -> torch.Tensor:
    """Read stored latent from PICache-populated public slots for a given layer.

    Non-hit positions (miss/imp with -1) get zero-filled (they'll be masked out
    by causal / imp selection anyway).

    Returns: (N, 1, D) — same shape as fresh latent_cache from forward_mla.
    """
    N = hit_pub_slots.numel()
    D = kv_buffer.shape[-1]
    result = torch.zeros(
        (N, 1, D), dtype=kv_buffer.dtype, device=kv_buffer.device
    )
    hit_mask = hit_pub_slots >= 0
    if hit_mask.any():
        result[hit_mask] = kv_buffer[hit_pub_slots[hit_mask]].to(result.dtype)
    return result
