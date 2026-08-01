"""Unit tests for A³ / CacheBlend reuse_utils.

Covered functions (from python/sglang/srt/models/reuse_utils.py):
  - compute_attention_sum_wo_head
  - get_topindices (both 'debug'=A³ and 'blend'=CacheBlend branches)
  - compute_prefill_attention_with_custom_kv (flashinfer + PyTorch fallback)
  - align_precomputed_key_rope

These tests are pure PyTorch and do not depend on a launched sglang server.
They can be run on CPU (fallback path) or GPU.
"""

from __future__ import annotations

import math
import unittest

import torch

from sglang.srt.models.reuse_utils import (
    CHECK_LAYER,
    REUSE_A3,
    REUSE_BLEND,
    align_precomputed_key_rope,
    compute_attention_sum_wo_head,
    compute_prefill_attention_with_custom_kv,
    get_topindices,
)


# =====================================================================
# helpers
# =====================================================================
def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _dtype() -> torch.dtype:
    # Prefer bfloat16 on GPU (matches GLM5.2 inference dtype), float32 on CPU.
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _make_qkv(
    num_heads: int, num_kv_heads: int, head_dim: int, total_len: int
):
    dev, dt = _device(), _dtype()
    q = torch.randn(1, num_heads, total_len, head_dim, device=dev, dtype=dt)
    k = torch.randn(1, num_kv_heads, total_len, head_dim, device=dev, dtype=dt)
    v = torch.randn(1, num_kv_heads, total_len, head_dim, device=dev, dtype=dt)
    return q, k, v


# =====================================================================
# compute_attention_sum_wo_head
# =====================================================================
class TestComputeAttentionSum(unittest.TestCase):
    def test_shape_and_dtype(self):
        total_len, last_len = 32, 8
        q = torch.randn(1, 4, total_len, 16, device=_device(), dtype=_dtype())
        # key length here is context_len + last_len as compute_attention_sum uses last_len from tail
        k = torch.randn(1, 4, total_len, 16, device=_device(), dtype=_dtype())
        out = compute_attention_sum_wo_head(q, k, last_len)
        context_len = total_len - last_len
        self.assertEqual(out.shape, (1, context_len))
        self.assertEqual(out.dtype, torch.float32)

    def test_nonnegative(self):
        # attention weights are non-negative after softmax; sums must be ≥ 0.
        total_len, last_len = 16, 4
        q = torch.randn(1, 2, total_len, 8, device=_device(), dtype=_dtype())
        k = torch.randn(1, 2, total_len, 8, device=_device(), dtype=_dtype())
        out = compute_attention_sum_wo_head(q, k, last_len)
        self.assertTrue(torch.all(out >= 0))

    def test_causal_mask_applied_within_query(self):
        # If we make query positions strongly attend to themselves via identity
        # keys, causal masking should still ensure earlier query positions get
        # zero attention to later ones. This only exercises the mask code path.
        total_len, last_len = 8, 4
        head_dim = 4
        q = torch.zeros(1, 1, total_len, head_dim, device=_device(), dtype=torch.float32)
        k = torch.zeros(1, 1, total_len, head_dim, device=_device(), dtype=torch.float32)
        for i in range(total_len):
            q[0, 0, i, i % head_dim] = 1.0
            k[0, 0, i, i % head_dim] = 1.0
        out = compute_attention_sum_wo_head(q, k, last_len)
        self.assertFalse(torch.isnan(out).any())


# =====================================================================
# get_topindices — A³ (debug) branch
# =====================================================================
class TestGetTopIndicesA3(unittest.TestCase):
    def test_output_length_and_range(self):
        total_len, last_len = 64, 16
        recomp_ratio = 0.25
        q, k, v = _make_qkv(num_heads=4, num_kv_heads=2, head_dim=8, total_len=total_len)
        v_old = v + 0.01 * torch.randn_like(v)
        cfg = {
            "reuse": REUSE_A3,
            "recomp_ratio": recomp_ratio,
            "last_len": last_len,
            "prefix_len": 0,
        }
        top = get_topindices(cfg, q, k, v, v_old, num_key_value_groups=2)
        context_len = total_len - last_len
        topk_num = max(1, int(context_len * recomp_ratio))
        expected_len = topk_num + last_len
        self.assertEqual(top.shape[0], expected_len)
        self.assertTrue(top.min().item() >= 0)
        self.assertTrue(top.max().item() < total_len)

    def test_query_tail_always_included(self):
        total_len, last_len = 32, 4
        q, k, v = _make_qkv(num_heads=2, num_kv_heads=1, head_dim=8, total_len=total_len)
        cfg = {"reuse": REUSE_A3, "recomp_ratio": 0.1, "last_len": last_len, "prefix_len": 0}
        top = get_topindices(cfg, q, k, v, v, num_key_value_groups=2)
        # last `last_len` entries must be the query tail positions
        tail = top[-last_len:].tolist()
        self.assertEqual(tail, list(range(total_len - last_len, total_len)))

    def test_prefix_len_respected(self):
        # A³ path is expected to skip positions in [0, prefix_len).
        total_len, last_len, prefix_len = 64, 8, 16
        q, k, v = _make_qkv(num_heads=4, num_kv_heads=2, head_dim=8, total_len=total_len)
        cfg = {
            "reuse": REUSE_A3,
            "recomp_ratio": 0.5,
            "last_len": last_len,
            "prefix_len": prefix_len,
        }
        top = get_topindices(cfg, q, k, v, v, num_key_value_groups=2)
        # Ignore the tail (last_len) query indices — the selected context ones
        # must all be >= prefix_len.
        context_selected = top[:-last_len]
        self.assertTrue((context_selected >= prefix_len).all())


# =====================================================================
# get_topindices — CacheBlend (blend) branch
# =====================================================================
class TestGetTopIndicesBlend(unittest.TestCase):
    def test_picks_positions_with_largest_v_diff(self):
        total_len, last_len = 32, 4
        head_dim = 8
        recomp_ratio = 0.25
        context_len = total_len - last_len

        # v == v_old for all positions EXCEPT a controlled subset which is
        # forced to differ. Those positions should be preferentially selected.
        dev = _device()
        v = torch.zeros(1, 2, total_len, head_dim, device=dev, dtype=torch.float32)
        v_old = torch.zeros_like(v)
        big_diff_positions = [3, 10, 20]
        for p in big_diff_positions:
            v[:, :, p, :] = 5.0  # v_old stays at 0 -> big L2 diff.

        # dummy q, k (unused by 'blend' branch, but required by signature)
        q = torch.zeros(1, 4, total_len, head_dim, device=dev, dtype=torch.float32)
        k = torch.zeros(1, 2, total_len, head_dim, device=dev, dtype=torch.float32)

        cfg = {
            "reuse": REUSE_BLEND,
            "recomp_ratio": recomp_ratio,
            "last_len": last_len,
            "prefix_len": 0,
        }
        top = get_topindices(cfg, q, k, v, v_old, num_key_value_groups=2)
        # The controlled big-diff positions must all appear in the selection
        # (they have strictly larger L2 diff than everything else, which is 0).
        topk_num = max(1, int(context_len * recomp_ratio))
        chosen = set(top[:topk_num].tolist())
        self.assertTrue(set(big_diff_positions).issubset(chosen))

    def test_output_shape(self):
        total_len, last_len = 40, 8
        recomp_ratio = 0.2
        q, k, v = _make_qkv(num_heads=4, num_kv_heads=2, head_dim=8, total_len=total_len)
        v_old = v + 0.01 * torch.randn_like(v)
        cfg = {"reuse": REUSE_BLEND, "recomp_ratio": recomp_ratio, "last_len": last_len}
        top = get_topindices(cfg, q, k, v, v_old, num_key_value_groups=2)
        context_len = total_len - last_len
        topk_num = max(1, int(context_len * recomp_ratio))
        self.assertEqual(top.shape[0], topk_num + last_len)


# =====================================================================
# get_topindices — error branch
# =====================================================================
class TestGetTopIndicesError(unittest.TestCase):
    def test_unknown_method_raises(self):
        q, k, v = _make_qkv(2, 1, 4, 8)
        cfg = {"reuse": "totally-unknown", "recomp_ratio": 0.1, "last_len": 2}
        with self.assertRaises(ValueError):
            get_topindices(cfg, q, k, v, v, num_key_value_groups=2)


# =====================================================================
# compute_prefill_attention_with_custom_kv
# =====================================================================
class TestComputePrefillAttentionCustomKV(unittest.TestCase):
    def test_shape_matches_flat_packed_format(self):
        num_heads, num_kv_heads, head_dim = 4, 2, 8
        total_len, imp_len = 32, 12
        q, k, v = _make_qkv(num_heads, num_kv_heads, head_dim, total_len)
        # Restrict q to imp_len positions
        imp_indices = torch.arange(total_len - imp_len, total_len, device=_device())
        q_imp = q[:, :, imp_indices, :]

        out = compute_prefill_attention_with_custom_kv(
            q=q_imp,
            k=k,
            v=v,
            num_kv_groups=num_heads // num_kv_heads,
            scale=1.0 / math.sqrt(head_dim),
            imp_indices=imp_indices,
            total_len=total_len,
        )
        # flat packed format: (q_len, num_heads * head_dim)
        self.assertEqual(out.shape, (imp_len, num_heads * head_dim))

    def test_causal_mask_blocks_future_positions(self):
        # Set imp_indices to [3, 6]; value at kv position 7 has a distinctive
        # magnitude — if causal masking is respected, neither query should be
        # influenced by kv position 7 (since 7 > 3 and 7 > 6).
        num_heads, num_kv_heads, head_dim = 1, 1, 4
        total_len = 8
        dev = _device()
        q = torch.ones(1, num_heads, 2, head_dim, device=dev, dtype=torch.float32)
        k = torch.ones(1, num_kv_heads, total_len, head_dim, device=dev, dtype=torch.float32)
        v = torch.zeros(1, num_kv_heads, total_len, head_dim, device=dev, dtype=torch.float32)
        v[:, :, 7, :] = 999.0  # would blow up output if it leaks in
        imp_indices = torch.tensor([3, 6], device=dev)
        out = compute_prefill_attention_with_custom_kv(
            q=q, k=k, v=v,
            num_kv_groups=1,
            scale=1.0 / math.sqrt(head_dim),
            imp_indices=imp_indices,
            total_len=total_len,
        )
        # Output should never carry the 999 signal — it would dominate any leak.
        self.assertTrue(out.abs().max().item() < 100.0)


# =====================================================================
# align_precomputed_key_rope — smoke test with a stub rotary_emb
# =====================================================================
class TestAlignPrecomputedKeyRope(unittest.TestCase):
    def test_returns_shape_matches_input(self):
        num_kv_heads, head_dim, total_len = 2, 8, 16
        dev, dt = _device(), _dtype()

        key_old = torch.randn(1, num_kv_heads, total_len, head_dim, device=dev, dtype=dt)
        fake_q = torch.zeros(total_len, 4 * head_dim, device=dev, dtype=dt)
        pos = torch.arange(total_len, device=dev, dtype=torch.int64)

        # rotary_emb stub: identity — returns q,k unchanged. Real rotary_emb
        # signature: (positions, q_flat, k_flat) -> (q, k).
        def rotary_emb(positions, q_flat, k_flat):
            return q_flat, k_flat

        out = align_precomputed_key_rope(
            rotary_emb=rotary_emb,
            org_positions=pos,
            key_old_4d=key_old,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            fake_q_flat=fake_q,
        )
        self.assertEqual(out.shape, key_old.shape)
        # Identity rotary — content should round-trip via the flat/4D reshape.
        self.assertTrue(torch.allclose(out.float(), key_old.float()))


# =====================================================================
# Sanity constants
# =====================================================================
class TestConstants(unittest.TestCase):
    def test_check_layer_is_1(self):
        # Both A³ and CacheBlend fix decision on layer 1 (baseline invariant).
        self.assertEqual(CHECK_LAYER, 1)


if __name__ == "__main__":
    unittest.main()
