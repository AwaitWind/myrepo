"""State-machine dispatch tests — pure-Torch, no heavy sglang imports.

These tests avoid pulling in ServerArgs/Req (which transitively imports
deep_gemm and pins libnvrtc). Instead they:

  * Re-implement the same reuse dispatch as a small standalone driver, using
    the pure-Torch reuse_utils API surface.
  * Verify the state-machine contract that Glm4MoeModel.forward and
    Glm4MoeAttention._reuse_forward_core rely on:
      - checking layer: Q is clipped to imp_indices, output has imp_len rows.
      - postchecking layer: KV is fused (imp positions = new, others = old).
      - positions/residual shape agreement across layers.

For the full E2E test with real GLM5.2 weights, see A3_CACHEBLEND_TESTING.md.
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
    compute_prefill_attention_with_custom_kv,
    get_topindices,
)


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


class _MiniAttentionState:
    """Standalone driver mirroring what Glm4MoeAttention._reuse_forward_core does.

    This is a *reference* implementation used to exercise the same algorithm
    and cross-check invariants — it duplicates the model wiring on purpose so
    the test does not need the entire modeling stack to load.
    """

    def __init__(self, num_heads=4, num_kv_heads=2, head_dim=8):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = 1.0 / math.sqrt(head_dim)

    def run_checking(
        self,
        q_4d: torch.Tensor,
        k_4d: torch.Tensor,
        v_4d: torch.Tensor,
        old_kv: torch.Tensor,  # (2, num_kv_heads, total_len, head_dim)
        org_positions: torch.Tensor,
        reuse_method: str,
        recomp_ratio: float,
        last_len: int,
        prefix_len: int = 0,
    ):
        """Emulate the checking layer's algorithm; returns (output, imp_indices)."""
        # Simulated rotary_emb (identity — RoPE alignment tested elsewhere).
        total_len = org_positions.shape[0]
        fake_q_flat = torch.zeros(
            total_len, self.num_heads * self.head_dim,
            dtype=q_4d.dtype, device=q_4d.device,
        )
        key_old_4d = align_precomputed_key_rope(
            rotary_emb=lambda p, q, k: (q, k),
            org_positions=org_positions,
            key_old_4d=old_kv[0].unsqueeze(0),
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            fake_q_flat=fake_q_flat,
        )
        value_old_4d = old_kv[1].unsqueeze(0)

        cfg = {
            "reuse": reuse_method,
            "recomp_ratio": recomp_ratio,
            "last_len": last_len,
            "prefix_len": prefix_len,
        }
        imp_indices = get_topindices(
            reuse_config=cfg,
            query_states=q_4d,
            key_states=k_4d,
            value_states=v_4d,
            value_old=value_old_4d,
            num_key_value_groups=self.num_kv_groups,
        )
        q_4d_imp = q_4d[:, :, imp_indices, :]
        out = compute_prefill_attention_with_custom_kv(
            q=q_4d_imp, k=k_4d, v=v_4d,
            num_kv_groups=self.num_kv_groups,
            scale=self.scaling,
            imp_indices=imp_indices,
            total_len=total_len,
        )
        return out, imp_indices, key_old_4d, value_old_4d

    def run_postchecking(
        self,
        q_4d_imp: torch.Tensor,
        k_4d_imp: torch.Tensor,
        v_4d_imp: torch.Tensor,
        key_old_4d: torch.Tensor,
        value_old_4d: torch.Tensor,
        imp_indices: torch.Tensor,
        total_len: int,
    ):
        key_fused = key_old_4d.clone()
        value_fused = value_old_4d.clone()
        key_fused[:, :, imp_indices, :] = k_4d_imp
        value_fused[:, :, imp_indices, :] = v_4d_imp
        return compute_prefill_attention_with_custom_kv(
            q=q_4d_imp, k=key_fused, v=value_fused,
            num_kv_groups=self.num_kv_groups,
            scale=self.scaling,
            imp_indices=imp_indices,
            total_len=total_len,
        )


class TestCheckingLayerContract(unittest.TestCase):
    def test_output_shape_and_imp_len(self):
        s = _MiniAttentionState()
        total_len, last_len, recomp_ratio = 32, 8, 0.25
        dev, dt = _device(), _dtype()
        q = torch.randn(1, s.num_heads, total_len, s.head_dim, device=dev, dtype=dt)
        k = torch.randn(1, s.num_kv_heads, total_len, s.head_dim, device=dev, dtype=dt)
        v = torch.randn(1, s.num_kv_heads, total_len, s.head_dim, device=dev, dtype=dt)
        old_k = torch.randn_like(k.squeeze(0))
        old_v = torch.randn_like(v.squeeze(0))
        old_kv = torch.stack([old_k, old_v], dim=0)
        org_pos = torch.arange(total_len, device=dev, dtype=torch.int64)

        out, imp, _, _ = s.run_checking(
            q, k, v, old_kv, org_pos,
            reuse_method=REUSE_A3, recomp_ratio=recomp_ratio, last_len=last_len,
        )
        expected_imp_len = max(1, int((total_len - last_len) * recomp_ratio)) + last_len
        self.assertEqual(imp.shape[0], expected_imp_len)
        self.assertEqual(out.shape, (expected_imp_len, s.num_heads * s.head_dim))


class TestPostcheckingLayerContract(unittest.TestCase):
    def test_output_shape_matches_imp_len(self):
        s = _MiniAttentionState()
        total_len, last_len, recomp_ratio = 32, 8, 0.25
        dev, dt = _device(), _dtype()

        # First, run a checking layer to obtain imp_indices.
        q = torch.randn(1, s.num_heads, total_len, s.head_dim, device=dev, dtype=dt)
        k = torch.randn(1, s.num_kv_heads, total_len, s.head_dim, device=dev, dtype=dt)
        v = torch.randn(1, s.num_kv_heads, total_len, s.head_dim, device=dev, dtype=dt)
        old_k = torch.randn_like(k.squeeze(0))
        old_v = torch.randn_like(v.squeeze(0))
        old_kv = torch.stack([old_k, old_v], dim=0)
        org_pos = torch.arange(total_len, device=dev, dtype=torch.int64)
        _, imp, key_old_4d, value_old_4d = s.run_checking(
            q, k, v, old_kv, org_pos,
            reuse_method=REUSE_A3, recomp_ratio=recomp_ratio, last_len=last_len,
        )

        # For the postchecking layer, inputs are only imp_len tokens wide.
        imp_len = imp.shape[0]
        q_imp = torch.randn(1, s.num_heads, imp_len, s.head_dim, device=dev, dtype=dt)
        k_imp = torch.randn(1, s.num_kv_heads, imp_len, s.head_dim, device=dev, dtype=dt)
        v_imp = torch.randn(1, s.num_kv_heads, imp_len, s.head_dim, device=dev, dtype=dt)

        out = s.run_postchecking(
            q_imp, k_imp, v_imp, key_old_4d, value_old_4d, imp, total_len,
        )
        self.assertEqual(out.shape, (imp_len, s.num_heads * s.head_dim))

    def test_kv_fusion_preserves_precomputed_at_non_imp_positions(self):
        """Fused KV must equal old_kv at every position NOT in imp_indices."""
        s = _MiniAttentionState()
        total_len, imp_len = 20, 8
        dev, dt = _device(), torch.float32

        key_old = torch.zeros(1, s.num_kv_heads, total_len, s.head_dim, device=dev, dtype=dt)
        value_old = torch.zeros_like(key_old)
        # Mark old positions with 1.0 so we can detect them.
        key_old.fill_(1.0)
        value_old.fill_(2.0)

        imp_indices = torch.tensor([0, 3, 5, 8, 11, 13, 17, 19], device=dev)
        # New values at imp positions
        k_imp = torch.full(
            (1, s.num_kv_heads, imp_len, s.head_dim), 99.0, device=dev, dtype=dt
        )
        v_imp = torch.full_like(k_imp, 42.0)

        # Manually build the fused KV — this is what postchecking does internally.
        key_fused = key_old.clone()
        value_fused = value_old.clone()
        key_fused[:, :, imp_indices, :] = k_imp
        value_fused[:, :, imp_indices, :] = v_imp

        # Now check: non-imp positions still 1.0/2.0, imp positions replaced.
        imp_set = set(imp_indices.tolist())
        for pos in range(total_len):
            if pos in imp_set:
                self.assertTrue(
                    torch.all(key_fused[0, :, pos, :] == 99.0),
                    f"pos {pos} in imp_indices should be new K",
                )
                self.assertTrue(
                    torch.all(value_fused[0, :, pos, :] == 42.0),
                    f"pos {pos} in imp_indices should be new V",
                )
            else:
                self.assertTrue(
                    torch.all(key_fused[0, :, pos, :] == 1.0),
                    f"pos {pos} outside imp_indices should keep old K",
                )
                self.assertTrue(
                    torch.all(value_fused[0, :, pos, :] == 2.0),
                    f"pos {pos} outside imp_indices should keep old V",
                )


class TestPositionsUpdateAcrossCheckLayer(unittest.TestCase):
    """After the checking layer, positions collapse to imp_indices."""

    def test_positions_indexed_by_imp_indices(self):
        dev = _device()
        total_len = 16
        org_positions = torch.arange(total_len, device=dev, dtype=torch.int64)
        imp_indices = torch.tensor([1, 4, 7, 10, 15], device=dev)

        new_positions = org_positions[imp_indices]
        self.assertEqual(new_positions.tolist(), [1, 4, 7, 10, 15])
        self.assertEqual(new_positions.shape[0], imp_indices.shape[0])


class TestResidualClipInvariant(unittest.TestCase):
    """DecoderLayer must clip residual to imp_len on the checking layer.

    We simulate the shape contract: pre-attn residual = (total_len, dim);
    attn output = (imp_len, dim); if we clip residual with imp_indices, the
    add becomes shape-compatible.
    """

    def test_clip_makes_shapes_compatible(self):
        dev = _device()
        total_len, hidden_dim = 24, 64
        imp_indices = torch.tensor([2, 5, 7, 11, 13, 20, 23], device=dev)
        imp_len = imp_indices.shape[0]

        residual_pre = torch.randn(total_len, hidden_dim, device=dev)
        attn_out = torch.randn(imp_len, hidden_dim, device=dev)

        # Simulate clip
        residual_clipped = residual_pre[imp_indices, :]
        self.assertEqual(residual_clipped.shape, (imp_len, hidden_dim))
        summed = residual_clipped + attn_out
        self.assertEqual(summed.shape, (imp_len, hidden_dim))


class TestCheckLayerIndex(unittest.TestCase):
    def test_constant_is_1_and_shared(self):
        # Both strategies decide on the same layer.
        self.assertEqual(CHECK_LAYER, 1)


if __name__ == "__main__":
    unittest.main()
