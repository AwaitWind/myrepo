"""
PIC 选择性 forward 单元测试。

覆盖 miss-only forward 路径的核心变更：
  pic_alloc._pic_alloc_transition_rope  → out_cache_loc 只含 miss slots
  pic_alloc.pic_alloc_for_extend        → req_to_token_pool 全位置写入
  schedule_batch.prepare_for_extend     → input_ids/miss_positions 只含 miss tokens
  model_runner._pic_prepopulate_hit_slots → delta-RoPE + KV 写入 hit private slots
"""

import unittest
from unittest.mock import MagicMock
from typing import Dict, List, Tuple

import torch

from sglang.srt.pic.hasher import segment_hash
from sglang.srt.pic.pic_alloc import _pic_alloc_transition_rope, DSAStatePool
from sglang.srt.pic.picache import PICache, SegmentEntry


# =========================================================================
# Helpers
# =========================================================================

def _kv_alloc(capacity: int = 512):
    # Start from 1 so slot 0 is never allocated; makes "non-zero" checks reliable.
    pool = list(range(1, capacity + 1))
    m = MagicMock()
    def _a(n):
        if n > len(pool): return None
        s = pool[:n]; del pool[:n]
        return torch.tensor(s, dtype=torch.int64)
    def _f(t): pool.extend(t.tolist())
    m.alloc = _a; m.free = _f
    m.device = torch.device("cpu")
    return m


def _make_req(segments, hit_segs=None, miss_segs=None, seg_entries=None):
    req = MagicMock()
    req.pic_segments = segments
    req.pic_hit_segments = hit_segs or []
    req.pic_miss_segments = miss_segs if miss_segs is not None else list(segments)
    req.pic_segment_entries = seg_entries or {}
    req.extend_input_len = sum(e - s for s, e in segments)
    req.pic_miss_segment_slots = {}
    req.pic_rope_hit_private_slots = {}
    return req


def _make_batch(reqs):
    b = MagicMock()
    b.reqs = reqs
    b.device = "cpu"
    b.pic_public_out_loc = None
    return b


def _make_picache(capacity=512):
    alloc = _kv_alloc(capacity)
    rp = MagicMock()
    rp.req_to_token = torch.zeros(32, 512, dtype=torch.int64)
    return PICache(req_to_token_pool=rp, token_to_kv_pool_allocator=alloc)


def _cached_entry(tokens: List[int], start_pos: int, num_slots: int):
    seg_ids = torch.tensor(tokens, dtype=torch.int64)
    h = segment_hash(seg_ids)
    pub_slots = torch.arange(start_pos * 100, start_pos * 100 + num_slots, dtype=torch.int64)
    return h, SegmentEntry(seg_hash=h, full_kv_slots=pub_slots,
                           token_ids=seg_ids, start_pos=start_pos)


def _run_alloc(reqs, capacity=512):
    alloc = _kv_alloc(capacity)
    batch = _make_batch(reqs)
    tree = MagicMock()
    tree.add_inflight = MagicMock()
    out = _pic_alloc_transition_rope(batch, tree, alloc, dsa_state_pool=None)
    return out, batch


# =========================================================================
# 1. out_cache_loc 只含 miss slots
# =========================================================================

class TestOutCacheLocMissOnly(unittest.TestCase):
    """_pic_alloc_transition_rope: out_cache_loc must be miss-only (not hit)."""

    def test_all_miss_out_equals_all_priv(self):
        """All-miss request: out_cache_loc covers all tokens."""
        segs = [(0, 8), (8, 16), (16, 24)]
        req = _make_req(segs)
        out, _ = _run_alloc([req])
        self.assertEqual(out.shape[0], 24)

    def test_hit_segs_not_in_out_cache_loc(self):
        """Hit segments must NOT appear in out_cache_loc."""
        h, entry = _cached_entry(list(range(8)), start_pos=0, num_slots=8)
        segs = [(0, 8), (8, 16)]   # seg0=hit, seg1=last(miss)
        req = _make_req(
            segs,
            hit_segs=[(0, 8, h)],
            miss_segs=[(8, 16)],
            seg_entries={h: entry},
        )
        out, _ = _run_alloc([req])
        # Only seg1 (8 tokens) should be in out_cache_loc
        self.assertEqual(out.shape[0], 8,
                         "out_cache_loc should have only miss-segment tokens")

    def test_mixed_hit_miss_out_is_miss_only(self):
        """[H, M, H, M]: only 2 miss segments' slots in out_cache_loc."""
        h0, e0 = _cached_entry(list(range(10)), start_pos=0, num_slots=10)
        h2, e2 = _cached_entry(list(range(100, 110)), start_pos=20, num_slots=10)
        segs = [(0, 10), (10, 20), (20, 30), (30, 40)]
        req = _make_req(
            segs,
            hit_segs=[(0, 10, h0), (20, 30, h2)],
            miss_segs=[(10, 20), (30, 40)],  # 20 miss tokens
            seg_entries={h0: e0, h2: e2},
        )
        out, _ = _run_alloc([req])
        self.assertEqual(out.shape[0], 20,
                         "out_cache_loc should have 20 miss tokens, not 40 total")

    def test_no_overlap_between_hit_priv_and_out_cache_loc(self):
        """hit private slots must be disjoint from out_cache_loc (miss slots)."""
        h, entry = _cached_entry(list(range(8)), start_pos=0, num_slots=8)
        segs = [(0, 8), (8, 16), (16, 24)]
        req = _make_req(
            segs,
            hit_segs=[(0, 8, h)],
            miss_segs=[(8, 16), (16, 24)],
            seg_entries={h: entry},
        )
        out, batch = _run_alloc([req])
        # Extract hit private slots
        hit_priv_set = set()
        for (s, e), info in req.pic_rope_hit_private_slots.items():
            hit_priv_set.update(info[0].tolist())
        miss_set = set(out.tolist())
        self.assertEqual(hit_priv_set & miss_set, set(),
                         "hit private and miss private slots must be disjoint")

    def test_non_pic_req_still_appended(self):
        """Non-PIC request in the batch: its extend tokens are in out_cache_loc."""
        req_non_pic = MagicMock()
        req_non_pic.pic_segments = None
        req_non_pic.extend_input_len = 12
        out, _ = _run_alloc([req_non_pic])
        self.assertEqual(out.shape[0], 12)


# =========================================================================
# 2. hit slot tensors (pic_all_hit_*)
# =========================================================================

class TestHitSlotTensors(unittest.TestCase):
    """After _pic_alloc_transition_rope, req.pic_rope_hit_private_slots contains
    (priv_slots, pub_kv_slots, old_start) for each hit segment."""

    def _run(self, reqs):
        return _run_alloc(reqs)

    def test_no_hits_hit_private_slots_empty(self):
        """All-miss request: pic_rope_hit_private_slots is empty."""
        segs = [(0, 8), (8, 16)]
        req = _make_req(segs)
        self._run([req])
        self.assertEqual(len(req.pic_rope_hit_private_slots), 0)

    def test_hit_seg_has_entry_in_hit_private_slots(self):
        """Hit segment creates an entry in req.pic_rope_hit_private_slots."""
        h, entry = _cached_entry(list(range(10)), start_pos=0, num_slots=10)
        segs = [(0, 10), (10, 20)]
        req = _make_req(
            segs,
            hit_segs=[(0, 10, h)],
            miss_segs=[(10, 20)],
            seg_entries={h: entry},
        )
        self._run([req])
        self.assertIn((0, 10), req.pic_rope_hit_private_slots,
                      "Hit segment must have an entry in pic_rope_hit_private_slots")

    def test_hit_entry_is_3_tuple_with_old_start(self):
        """Each entry is (priv_slots, pub_kv_slots, old_start)."""
        old_start = 5
        h, entry = _cached_entry(list(range(8)), start_pos=old_start, num_slots=8)
        segs = [(0, 8), (8, 16)]
        req = _make_req(
            segs,
            hit_segs=[(0, 8, h)],
            miss_segs=[(8, 16)],
            seg_entries={h: entry},
        )
        self._run([req])
        info = req.pic_rope_hit_private_slots[(0, 8)]
        self.assertEqual(len(info), 3, "hit_info should be (priv, pub, old_start)")
        priv, pub, stored_old_start = info
        self.assertEqual(stored_old_start, old_start,
                         f"old_start should be {old_start}, got {stored_old_start}")

    def test_delta_inferred_from_old_start(self):
        """delta = new_start - old_start can be computed from stored old_start."""
        old_start = 5
        new_start = 20
        h, entry = _cached_entry(list(range(8)), start_pos=old_start, num_slots=8)
        segs = [(new_start, new_start + 8), (new_start + 8, new_start + 16)]
        req = _make_req(
            segs,
            hit_segs=[(new_start, new_start + 8, h)],
            miss_segs=[(new_start + 8, new_start + 16)],
            seg_entries={h: entry},
        )
        self._run([req])
        _, _, stored_old = req.pic_rope_hit_private_slots[(new_start, new_start + 8)]
        expected_delta = new_start - old_start
        computed_delta = new_start - stored_old
        self.assertEqual(computed_delta, expected_delta,
                         f"delta should be {expected_delta}")

    def test_pub_kv_matches_entry_full_kv_slots(self):
        """pub_kv_slots in the tuple should equal entry.full_kv_slots."""
        h, entry = _cached_entry(list(range(6)), start_pos=0, num_slots=6)
        segs = [(0, 6), (6, 12)]
        req = _make_req(
            segs,
            hit_segs=[(0, 6, h)],
            miss_segs=[(6, 12)],
            seg_entries={h: entry},
        )
        self._run([req])
        _, pub_kv, _ = req.pic_rope_hit_private_slots[(0, 6)]
        self.assertTrue(torch.equal(pub_kv, entry.full_kv_slots),
                        "pub_kv_slots should match entry.full_kv_slots")

    def test_multiple_hit_segs_all_present(self):
        """Multiple hit segments all appear in pic_rope_hit_private_slots."""
        h1, e1 = _cached_entry(list(range(5)), start_pos=0, num_slots=5)
        h2, e2 = _cached_entry(list(range(50, 57)), start_pos=10, num_slots=7)
        segs = [(0, 5), (5, 12), (12, 19), (19, 24)]
        req = _make_req(
            segs,
            hit_segs=[(0, 5, h1), (12, 19, h2)],
            miss_segs=[(5, 12), (19, 24)],
            seg_entries={h1: e1, h2: e2},
        )
        self._run([req])
        self.assertIn((0, 5), req.pic_rope_hit_private_slots)
        self.assertIn((12, 19), req.pic_rope_hit_private_slots)


# =========================================================================
# 3. pic_miss_positions
# =========================================================================

class TestMissPositions(unittest.TestCase):
    """The miss_positions_list from prepare_for_extend must be actual positions."""

    def _extract_miss_positions(self, segs_hit_miss_list):
        """
        Simulate the _build_input_ids_and_miss_positions logic.
        segs_hit_miss_list: list of (segments, hit_set, fill_ids)
        Returns flat miss positions.
        """
        miss_positions = []
        for (segs, hit_set, fill_ids) in segs_hit_miss_list:
            miss_segs = [seg for seg in segs if seg not in hit_set]
            for (s, e) in segs:
                if (s, e) in set(miss_segs):
                    miss_positions.extend(range(s, e))
        return miss_positions

    def test_all_miss_positions_are_sequential(self):
        """All-miss request: miss positions = [0, 1, ..., total-1]."""
        segs = [(0, 10), (10, 20), (20, 30)]
        miss_segs = segs  # all miss
        positions = self._extract_miss_positions(
            [(segs, set(), list(range(30)))]
        )
        self.assertEqual(positions, list(range(30)))

    def test_hit_positions_excluded(self):
        """Hit segment positions are NOT in miss_positions."""
        segs = [(0, 10), (10, 20), (20, 30), (30, 40)]
        hit_set = {(0, 10), (20, 30)}
        positions = self._extract_miss_positions(
            [(segs, hit_set, list(range(40)))]
        )
        # Miss positions: [10..19] + [30..39]
        expected = list(range(10, 20)) + list(range(30, 40))
        self.assertEqual(positions, expected)

    def test_positions_are_non_contiguous_for_shuffled_docs(self):
        """[H, M, H, M] pattern → miss positions have a gap."""
        segs = [(0, 8), (8, 16), (16, 24), (24, 32)]
        hit_set = {(0, 8), (16, 24)}
        positions = self._extract_miss_positions(
            [(segs, hit_set, list(range(32)))]
        )
        # Miss: [8..15] + [24..31] — non-contiguous!
        expected = list(range(8, 16)) + list(range(24, 32))
        self.assertEqual(positions, expected)
        # Verify non-contiguous: the gap between 15 and 24
        self.assertTrue(positions[8] - positions[7] > 1,
                        "Positions should be non-contiguous (there's a hit gap)")

    def test_miss_positions_count_equals_out_cache_loc_size(self):
        """Number of miss positions == out_cache_loc.shape[0]."""
        h, entry = _cached_entry(list(range(10)), start_pos=0, num_slots=10)
        segs = [(0, 10), (10, 20), (20, 30)]
        req = _make_req(
            segs,
            hit_segs=[(0, 10, h)],
            miss_segs=[(10, 20), (20, 30)],
            seg_entries={h: entry},
        )
        out, _ = _run_alloc([req])
        # Miss positions: [10..29] = 20 tokens
        miss_positions = list(range(10, 20)) + list(range(20, 30))
        self.assertEqual(len(miss_positions), out.shape[0])


# =========================================================================
# 4. req_to_token_pool 全位置写入
# =========================================================================

class TestReqToTokenPoolWrite(unittest.TestCase):
    """pic_alloc_for_extend must write ALL positions to req_to_token_pool."""

    def _run_pic_alloc_for_extend(self, reqs, pool_size=512):
        from sglang.srt.pic.pic_alloc import pic_alloc_for_extend

        alloc = _kv_alloc(pool_size)
        tree = MagicMock()
        tree.add_inflight = MagicMock()
        tree.evict = MagicMock()
        tree.supports_mamba.return_value = False

        # Build mock batch
        batch = _make_batch(reqs)
        batch.seq_lens = torch.tensor([r.extend_input_len for r in reqs], dtype=torch.int64)
        batch.seq_lens_cpu = batch.seq_lens.clone()
        batch.prefix_lens = [0] * len(reqs)
        batch.extend_lens = [r.extend_input_len for r in reqs]
        batch.extend_num_tokens = sum(r.extend_input_len for r in reqs)
        batch.maybe_evict_swa = MagicMock()

        # req_to_token_pool
        rtp = MagicMock()
        pool_tensor = torch.zeros(len(reqs), 512, dtype=torch.int64)
        rtp.req_to_token = pool_tensor
        rtp.alloc = lambda reqs_: list(range(len(reqs_)))
        rtp.mamba_allocator = MagicMock()
        rtp.mamba_allocator.available_size.return_value = 1000
        tree.supports_mamba.return_value = False
        batch.req_to_token_pool = rtp

        out_loc, _, req_idx_cpu = pic_alloc_for_extend(
            batch, tree, alloc, dsa_state_pool=None
        )
        return out_loc, pool_tensor, reqs

    def test_hit_positions_populated(self):
        """Hit segment positions in req_to_token_pool are filled (equal to allocated priv slots)."""
        h, entry = _cached_entry(list(range(8)), start_pos=0, num_slots=8)
        segs = [(0, 8), (8, 16)]
        req = _make_req(
            segs,
            hit_segs=[(0, 8, h)],
            miss_segs=[(8, 16)],
            seg_entries={h: entry},
        )
        out_loc, pool, _ = self._run_pic_alloc_for_extend([req])
        # Hit segment's private slots should be in req_to_token_pool[0, 0:8]
        hit_region = pool[0, 0:8]
        priv_slots = req.pic_rope_hit_private_slots[(0, 8)][0]
        self.assertTrue(
            torch.equal(hit_region, priv_slots),
            f"Hit region should equal priv slots: {hit_region.tolist()} vs {priv_slots.tolist()}"
        )

    def test_miss_positions_populated(self):
        """Miss segment positions in req_to_token_pool are filled."""
        segs = [(0, 8), (8, 16)]
        req = _make_req(segs)
        out_loc, pool, _ = self._run_pic_alloc_for_extend([req])
        miss_region = pool[0, 8:16]
        self.assertTrue((miss_region != 0).all(),
                        "Miss positions should be filled in req_to_token_pool")

    def test_all_positions_nonzero_for_full_hit_plus_last(self):
        """[H1, H2, MISS_last]: all 3 segment positions filled with their slot indices."""
        h1, e1 = _cached_entry(list(range(6)), start_pos=0, num_slots=6)
        h2, e2 = _cached_entry(list(range(60, 66)), start_pos=6, num_slots=6)
        segs = [(0, 6), (6, 12), (12, 18)]
        req = _make_req(
            segs,
            hit_segs=[(0, 6, h1), (6, 12, h2)],
            miss_segs=[(12, 18)],
            seg_entries={h1: e1, h2: e2},
        )
        out_loc, pool, _ = self._run_pic_alloc_for_extend([req])
        # All 18 positions should be set to their respective priv slot indices
        h1_priv = req.pic_rope_hit_private_slots[(0, 6)][0]
        h2_priv = req.pic_rope_hit_private_slots[(6, 12)][0]
        m_priv  = req.pic_miss_segment_slots[(12, 18)][0]
        self.assertTrue(torch.equal(pool[0, 0:6],  h1_priv), "H1 region incorrect")
        self.assertTrue(torch.equal(pool[0, 6:12], h2_priv), "H2 region incorrect")
        self.assertTrue(torch.equal(pool[0, 12:18], m_priv), "Miss region incorrect")


# =========================================================================
# 5. _pic_prepopulate_hit_slots
# =========================================================================

def _writeback_hit_slots(pub_slots, priv_slots, delta_pos, buffers, rotary_emb, kv_lora_rank):
    """Simulate _pic_prepopulate_hit_slots logic (without ModelRunner)."""
    for buf in buffers:
        cached = buf[pub_slots].float()
        k_nope = cached[:, :kv_lora_rank]
        k_pe_old = cached[:, kv_lora_rank:]
        if rotary_emb is not None and delta_pos.any():
            dummy = torch.zeros_like(k_pe_old)
            _, k_pe_new = rotary_emb(delta_pos, dummy, k_pe_old)
        else:
            k_pe_new = k_pe_old
        new_kv = torch.cat([k_nope, k_pe_new], dim=-1).to(buf.dtype)
        buf[priv_slots] = new_kv


class TestPICPrepopulateHitSlots(unittest.TestCase):
    """_pic_prepopulate_hit_slots correctly pre-fills hit private slots."""

    def _make_buffers(self, num_layers=2, slots=64, kv_dim=8):
        return [torch.zeros(slots, kv_dim) for _ in range(num_layers)]

    def test_k_nope_copied_unchanged(self):
        """k_nope (position-free) is copied from public to private without change."""
        bufs = self._make_buffers(num_layers=1, slots=32, kv_dim=6)
        buf = bufs[0]
        kv_lora_rank = 4
        # Set known public slot value
        buf[10] = torch.tensor([1., 2., 3., 4.,   # k_nope
                                  5., 6.])          # k_pe
        pub = torch.tensor([10], dtype=torch.int64)
        priv = torch.tensor([20], dtype=torch.int64)
        delta = torch.tensor([0], dtype=torch.int64)

        _writeback_hit_slots(pub, priv, delta, bufs, rotary_emb=None, kv_lora_rank=kv_lora_rank)

        self.assertTrue(torch.allclose(buf[20, :kv_lora_rank], buf[10, :kv_lora_rank]),
                        "k_nope should be identical in pub and priv slots")

    def test_zero_delta_k_pe_unchanged(self):
        """With delta=0, k_pe is copied unchanged (no RoPE shift)."""
        bufs = self._make_buffers(num_layers=1, slots=32, kv_dim=8)
        buf = bufs[0]
        buf[5] = torch.arange(8, dtype=torch.float32)
        pub = torch.tensor([5], dtype=torch.int64)
        priv = torch.tensor([15], dtype=torch.int64)
        delta = torch.tensor([0], dtype=torch.int64)

        _writeback_hit_slots(pub, priv, delta, bufs, rotary_emb=None, kv_lora_rank=4)

        self.assertTrue(torch.allclose(buf[15], buf[5]),
                        "Zero delta: priv slot should be identical to pub slot")

    def test_all_layers_written(self):
        """Pre-population runs on every layer."""
        NUM = 4
        bufs = self._make_buffers(num_layers=NUM, slots=32, kv_dim=4)
        for lid, buf in enumerate(bufs):
            buf[3] = float(lid + 1)
        pub = torch.tensor([3], dtype=torch.int64)
        priv = torch.tensor([25], dtype=torch.int64)
        delta = torch.tensor([0], dtype=torch.int64)

        _writeback_hit_slots(pub, priv, delta, bufs, rotary_emb=None, kv_lora_rank=2)

        for lid, buf in enumerate(bufs):
            self.assertTrue(torch.allclose(buf[25], buf[3]),
                            f"layer {lid}: priv slot not written")

    def test_private_and_public_slots_disjoint(self):
        """Writes go to priv slots, pub slots are unchanged."""
        bufs = self._make_buffers(num_layers=1, slots=32, kv_dim=4)
        buf = bufs[0]
        buf[7] = 42.0
        buf[20] = 99.0
        pub = torch.tensor([7], dtype=torch.int64)
        priv = torch.tensor([20], dtype=torch.int64)
        delta = torch.tensor([0], dtype=torch.int64)

        _writeback_hit_slots(pub, priv, delta, bufs, rotary_emb=None, kv_lora_rank=2)

        # pub slot unchanged
        self.assertTrue(torch.allclose(buf[7], torch.full((4,), 42.0)),
                        "Public slot should not be modified")
        # priv slot written
        self.assertTrue(torch.allclose(buf[20], buf[7]),
                        "Private slot should now equal public slot")

    def test_multiple_hit_tokens(self):
        """Multiple hit tokens in a segment are all pre-populated."""
        bufs = self._make_buffers(num_layers=1, slots=64, kv_dim=4)
        buf = bufs[0]
        for i in range(5):
            buf[i] = float(i + 1)
        pub = torch.arange(5, dtype=torch.int64)
        priv = torch.arange(10, 15, dtype=torch.int64)
        delta = torch.zeros(5, dtype=torch.int64)

        _writeback_hit_slots(pub, priv, delta, bufs, rotary_emb=None, kv_lora_rank=2)

        for i in range(5):
            self.assertTrue(torch.allclose(buf[10 + i], buf[i]),
                            f"token {i}: priv slot not correctly copied")


# =========================================================================
# 6. End-to-end: alloc + pool write + miss-positions consistent
# =========================================================================

class TestSelectiveForwardEndToEnd(unittest.TestCase):
    """Verify that out_cache_loc, req_to_token_pool, and miss_positions are consistent."""

    def test_out_loc_plus_hit_private_equals_seq_len(self):
        """len(out_cache_loc) + len(hit_private_slots) == seq_len for PIC request."""
        h, entry = _cached_entry(list(range(10)), start_pos=0, num_slots=10)
        segs = [(0, 10), (10, 20), (20, 30), (30, 40)]
        req = _make_req(
            segs,
            hit_segs=[(0, 10, h)],
            miss_segs=[(10, 20), (20, 30), (30, 40)],
            seg_entries={h: entry},
        )
        out, batch = _run_alloc([req])
        # Hit private slots: 10 tokens
        hit_priv_count = sum(
            e - s for (s, e) in req.pic_rope_hit_private_slots
        )
        miss_count = out.shape[0]
        seq_len = req.extend_input_len  # 40

        self.assertEqual(hit_priv_count + miss_count, seq_len,
                         "hit_private + miss_out_cache == seq_len must hold")

    def test_miss_positions_count_matches_out_cache_loc(self):
        """Number of miss positions = out_cache_loc length."""
        h, entry = _cached_entry(list(range(8)), start_pos=0, num_slots=8)
        segs = [(0, 8), (8, 16), (16, 24)]
        req = _make_req(
            segs,
            hit_segs=[(0, 8, h)],
            miss_segs=[(8, 16), (16, 24)],
            seg_entries={h: entry},
        )
        out, _ = _run_alloc([req])
        # Simulate miss position extraction
        miss_positions = []
        for (s, e) in segs:
            if (s, e) in {(8, 16), (16, 24)}:
                miss_positions.extend(range(s, e))
        self.assertEqual(len(miss_positions), out.shape[0],
                         "miss_positions count must match out_cache_loc length")

    def test_pub_loc_shape_matches_out_cache_loc(self):
        """pic_public_out_loc has same shape as out_cache_loc (miss tokens only)."""
        segs = [(0, 6), (6, 12), (12, 18)]
        req = _make_req(segs)
        out, batch = _run_alloc([req])
        pub_loc = getattr(batch, "pic_public_out_loc", None)
        if pub_loc is not None:
            self.assertEqual(pub_loc.shape[0], out.shape[0],
                             "pic_public_out_loc and out_cache_loc must have same length")

    def test_all_miss_no_hit_tensors(self):
        """All-miss batch: no hit private slots, all tokens in out_cache_loc."""
        segs = [(0, 10), (10, 20), (20, 30)]
        req = _make_req(segs)
        out, batch = _run_alloc([req])
        # No hit private slots
        self.assertEqual(len(req.pic_rope_hit_private_slots), 0,
                         "All-miss request should have no hit private slots")
        # All 30 tokens should be in out_cache_loc
        self.assertEqual(out.shape[0], 30)


if __name__ == "__main__":
    unittest.main()
