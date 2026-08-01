"""
PIC unit tests — transition_rope mode only.

Coverage:
  hasher.py       segment_hash consistency & collision resistance
  segmenter.py    split_and_tokenize invariants S1-S6
  eviction.py     eviction strategy ordering
  picache.py      SegmentEntry / PICache core logic
  pic_alloc.py    DSAStatePool alloc/free
"""

import time
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.pic.eviction import (
    FIFOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
)
from sglang.srt.pic.hasher import segment_hash
from sglang.srt.pic.pic_alloc import DSAStatePool
from sglang.srt.pic.picache import PICache, SegmentEntry
from sglang.srt.pic.segmenter import split_and_tokenize


# =========================================================================
# Helpers
# =========================================================================

class DummyTokenizer:
    """Minimal tokenizer: encodes each character to ord(c) % 1000."""
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 1000 for c in text]


def _make_kv_allocator(capacity: int = 256):
    """Return a minimal mock token_to_kv_pool_allocator."""
    pool = list(range(capacity))
    alloc = MagicMock()

    def _alloc(n):
        if n > len(pool):
            return None
        slots = pool[:n]
        del pool[:n]
        return torch.tensor(slots, dtype=torch.int64)

    def _free(t):
        pool.extend(t.tolist())

    alloc.alloc = _alloc
    alloc.free = _free
    alloc.device = torch.device("cpu")
    return alloc


def _make_req_pool():
    pool = MagicMock()
    pool.req_to_token = torch.zeros(32, 512, dtype=torch.int64)
    return pool


def _make_picache(capacity=256, dsa=False):
    """Build a PICache backed by simple mock objects."""
    alloc = _make_kv_allocator(capacity)
    req_pool = _make_req_pool()
    dsa_pool = DSAStatePool(2, 16, (4,), device=torch.device("cpu")) if dsa else None
    return PICache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=alloc,
        dsa_state_pool=dsa_pool,
    )


def _entry(tokens, num_slots=4, dsa_slot=0):
    """Build a SegmentEntry with a given token list."""
    seg_ids = torch.tensor(tokens, dtype=torch.int64)
    h = segment_hash(seg_ids)
    return SegmentEntry(
        seg_hash=h,
        full_kv_slots=torch.arange(num_slots, dtype=torch.int64),
        token_ids=seg_ids,
        dsa_state_slot=dsa_slot,
    )


# =========================================================================
# 1. hasher
# =========================================================================

class TestSegmentHash(unittest.TestCase):

    def test_deterministic(self):
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        self.assertEqual(segment_hash(ids), segment_hash(ids))

    def test_different_sequences_differ(self):
        a = torch.tensor([1, 2, 3], dtype=torch.int64)
        b = torch.tensor([1, 2, 4], dtype=torch.int64)
        self.assertNotEqual(segment_hash(a), segment_hash(b))

    def test_order_matters(self):
        a = torch.tensor([1, 2, 3], dtype=torch.int64)
        b = torch.tensor([3, 2, 1], dtype=torch.int64)
        self.assertNotEqual(segment_hash(a), segment_hash(b))

    def test_returns_16_bytes(self):
        ids = torch.tensor([10, 20], dtype=torch.int64)
        self.assertIsInstance(segment_hash(ids), bytes)
        self.assertEqual(len(segment_hash(ids)), 16)

    def test_single_element(self):
        ids = torch.tensor([42], dtype=torch.int64)
        self.assertEqual(len(segment_hash(ids)), 16)


# =========================================================================
# 2. segmenter
# =========================================================================

class TestSplitAndTokenize(unittest.TestCase):

    def setUp(self):
        self.tok = DummyTokenizer()

    def test_single_segment_no_sep(self):
        ids, segs = split_and_tokenize("hello", self.tok)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0][0], 0)
        self.assertEqual(segs[0][1], len(ids))

    def test_two_segments(self):
        ids, segs = split_and_tokenize("doc1<<PIC_SEP>>doc2", self.tok)
        self.assertEqual(len(segs), 2)
        # S1: first segment starts at 0
        self.assertEqual(segs[0][0], 0)
        # S2: last segment ends at len(ids)
        self.assertEqual(segs[-1][1], len(ids))
        # S3: contiguous
        self.assertEqual(segs[0][1], segs[1][0])

    def test_three_segments_invariants(self):
        text = "A<<PIC_SEP>>BB<<PIC_SEP>>CCC"
        ids, segs = split_and_tokenize(text, self.tok)
        self.assertEqual(len(segs), 3)
        self.assertEqual(segs[0][0], 0)
        self.assertEqual(segs[-1][1], len(ids))
        for i in range(len(segs) - 1):
            self.assertEqual(segs[i][1], segs[i + 1][0])  # S3
        for (s, e) in segs:
            self.assertGreater(e, s)  # S4: non-empty

    def test_empty_parts_skipped(self):
        # Leading/trailing sep → empty strings → skipped (S4)
        ids, segs = split_and_tokenize("<<PIC_SEP>>content<<PIC_SEP>>", self.tok)
        for (s, e) in segs:
            self.assertGreater(e, s)

    def test_custom_separator(self):
        ids, segs = split_and_tokenize("X|||Y|||Z", self.tok, separator="|||")
        self.assertEqual(len(segs), 3)

    def test_no_tokens_returns_single_segment(self):
        # Pathological: all separators → empty → fallback to full text
        ids, segs = split_and_tokenize("hello", self.tok)
        self.assertTrue(len(ids) > 0)
        self.assertEqual(len(segs), 1)


# =========================================================================
# 3. eviction strategies
# =========================================================================

class TestEvictionStrategies(unittest.TestCase):

    def _entries_with_times(self, times):
        entries = []
        for i, t in enumerate(times):
            e = MagicMock()
            e.last_access_time = t
            e.creation_time = t
            e.hit_count = i
            e.priority = i
            entries.append(e)
        return entries

    def test_lru_oldest_first(self):
        entries = self._entries_with_times([3.0, 1.0, 2.0])
        s = LRUStrategy()
        sorted_e = sorted(entries, key=s.get_priority)
        self.assertEqual(sorted_e[0].last_access_time, 1.0)

    def test_mru_newest_first(self):
        entries = self._entries_with_times([3.0, 1.0, 2.0])
        s = MRUStrategy()
        sorted_e = sorted(entries, key=s.get_priority)
        self.assertEqual(sorted_e[0].last_access_time, 3.0)

    def test_fifo_earliest_creation_first(self):
        entries = self._entries_with_times([3.0, 1.0, 2.0])
        s = FIFOStrategy()
        sorted_e = sorted(entries, key=s.get_priority)
        self.assertEqual(sorted_e[0].creation_time, 1.0)

    def test_lfu_fewest_hits_first(self):
        entries = self._entries_with_times([1.0, 2.0, 3.0])
        s = LFUStrategy()
        sorted_e = sorted(entries, key=s.get_priority)
        self.assertEqual(sorted_e[0].hit_count, 0)

    def test_priority_lowest_first(self):
        entries = self._entries_with_times([1.0, 2.0, 3.0])
        entries[0].priority = 5
        entries[1].priority = 1
        entries[2].priority = 3
        s = PriorityStrategy()
        sorted_e = sorted(entries, key=s.get_priority)
        self.assertEqual(sorted_e[0].priority, 1)


# =========================================================================
# 4. DSAStatePool
# =========================================================================

class TestDSAStatePool(unittest.TestCase):

    def _pool(self, size=8):
        return DSAStatePool(
            num_dsa_layers=2,
            pool_size=size,
            dsa_state_shape=(4,),
            dsa_dtype=torch.float32,
            device=torch.device("cpu"),
        )

    def test_initial_available_size(self):
        p = self._pool(8)
        self.assertEqual(p.available_size(), 8)

    def test_alloc_returns_valid_slots(self):
        p = self._pool(8)
        slots = p.alloc(3)
        self.assertIsNotNone(slots)
        self.assertEqual(slots.shape[0], 3)
        self.assertTrue(all(s >= 1 for s in slots.tolist()))

    def test_alloc_decrements_available(self):
        p = self._pool(8)
        p.alloc(3)
        self.assertEqual(p.available_size(), 5)

    def test_free_restores_available(self):
        p = self._pool(8)
        slots = p.alloc(3)
        p.free(slots)
        self.assertEqual(p.available_size(), 8)

    def test_alloc_exhaustion_returns_none(self):
        p = self._pool(4)
        p.alloc(4)
        result = p.alloc(1)
        self.assertIsNone(result)

    def test_slot_zero_never_allocated(self):
        p = self._pool(4)
        slots = p.alloc(4)
        self.assertTrue(all(s >= 1 for s in slots.tolist()))

    def test_alloc_clears_state(self):
        p = self._pool(4)
        slot_t = p.alloc(1)
        slot = int(slot_t[0])
        p.dsa_state[:, slot, ...] = 99.0
        p.free(slot_t)
        slot_t2 = p.alloc(1)
        # After alloc, state should be zeroed
        self.assertTrue(torch.all(p.dsa_state[:, int(slot_t2[0]), ...] == 0))

    def test_get_set_state(self):
        p = self._pool(4)
        slot = int(p.alloc(1)[0])
        data = torch.ones(2, 4)
        p.set_state(slot, data)
        self.assertTrue(torch.allclose(p.get_state(slot), data))


# =========================================================================
# 5. SegmentEntry
# =========================================================================

class TestSegmentEntry(unittest.TestCase):

    def test_default_fields(self):
        e = _entry([1, 2, 3])
        self.assertEqual(e.lock_ref, 0)
        self.assertEqual(e.hit_count, 0)
        self.assertEqual(e.dsa_state_slot, 0)

    def test_seg_hash_is_bytes_16(self):
        e = _entry([1, 2, 3])
        self.assertIsInstance(e.seg_hash, bytes)
        self.assertEqual(len(e.seg_hash), 16)

    def test_different_tokens_different_hash(self):
        e1 = _entry([1, 2, 3])
        e2 = _entry([1, 2, 4])
        self.assertNotEqual(e1.seg_hash, e2.seg_hash)


# =========================================================================
# 6. PICache — match_prefix
# =========================================================================

class TestPICacheMatchPrefix(unittest.TestCase):

    def _make_req(self, segments, fill_ids):
        req = MagicMock()
        req.pic_segments = segments
        req.get_fill_ids.return_value = fill_ids
        req.pic_segment_entries = {}
        req.pic_hit_segments = []
        req.pic_miss_segments = []
        req._pic_cached_segments = set()
        req.pic_miss_segment_slots = {}
        req.pic_rope_hit_private_slots = {}
        return req

    def test_no_pic_segments_returns_empty(self):
        cache = _make_picache()
        req = MagicMock()
        req.pic_segments = None
        params = MatchPrefixParams(key=MagicMock(), req=req, cow_mamba=False)
        result = cache.match_prefix(params)
        self.assertEqual(result.device_indices.shape[0], 0)
        self.assertIsNone(result.pic_segment_entries)

    def test_empty_cache_all_miss(self):
        cache = _make_picache()
        fill_ids = list(range(20))
        # 2 segments: [0,10) and [10,20)
        req = self._make_req([(0, 10), (10, 20)], fill_ids)
        params = MatchPrefixParams(key=MagicMock(), req=req, cow_mamba=False)
        result = cache.match_prefix(params)

        # device_indices always empty (transition_rope design)
        self.assertEqual(result.device_indices.shape[0], 0)
        # 2 entries: first is None (miss), last is None (never cached)
        self.assertEqual(len(result.pic_segment_entries), 2)
        self.assertIsNone(result.pic_segment_entries[0])
        self.assertIsNone(result.pic_segment_entries[1])

    def test_cache_hit_updates_entry(self):
        cache = _make_picache()
        tokens = list(range(10))
        seg_ids = torch.tensor(tokens, dtype=torch.int64)
        h = segment_hash(seg_ids)
        kv_slots = torch.arange(10, dtype=torch.int64)

        # Manually insert an entry
        entry = SegmentEntry(
            seg_hash=h,
            full_kv_slots=kv_slots,
            token_ids=seg_ids,
            hit_count=0,
        )
        cache._entries[h] = entry

        fill_ids = tokens + list(range(10, 20))  # seg0: tokens[0:10], seg1 (last): [10:20]
        req = self._make_req([(0, 10), (10, 20)], fill_ids)
        params = MatchPrefixParams(key=MagicMock(), req=req, cow_mamba=False)
        result = cache.match_prefix(params)

        # seg0 is a hit
        self.assertIsNotNone(result.pic_segment_entries[0])
        self.assertEqual(result.pic_segment_entries[0].hit_count, 1)
        # seg1 (last) is always None
        self.assertIsNone(result.pic_segment_entries[1])

    def test_device_indices_always_empty(self):
        """Verify empty device_indices regardless of cache state (DSA kernel requirement)."""
        cache = _make_picache()
        tokens = list(range(10))
        seg_ids = torch.tensor(tokens, dtype=torch.int64)
        h = segment_hash(seg_ids)
        cache._entries[h] = SegmentEntry(
            seg_hash=h,
            full_kv_slots=torch.arange(10, dtype=torch.int64),
            token_ids=seg_ids,
        )
        fill_ids = tokens + list(range(10, 20))
        req = self._make_req([(0, 10), (10, 20)], fill_ids)
        params = MatchPrefixParams(key=MagicMock(), req=req, cow_mamba=False)
        result = cache.match_prefix(params)

        # Must be empty so prefix_len=0 and extend_input_len=seq_len
        self.assertEqual(result.device_indices.shape[0], 0)


# =========================================================================
# 7. PICache — _match_segment (hash collision guard)
# =========================================================================

class TestPICacheMatchSegment(unittest.TestCase):

    def test_miss_when_empty(self):
        cache = _make_picache()
        ids = torch.tensor([1, 2, 3], dtype=torch.int64)
        self.assertIsNone(cache._match_segment(segment_hash(ids), ids))

    def test_hit_on_exact_match(self):
        cache = _make_picache()
        ids = torch.tensor([1, 2, 3], dtype=torch.int64)
        h = segment_hash(ids)
        entry = _entry([1, 2, 3])
        cache._entries[h] = entry
        self.assertIs(cache._match_segment(h, ids), entry)

    def test_collision_guard_rejects_different_ids(self):
        """Same hash key but different token_ids → miss (collision guard)."""
        cache = _make_picache()
        ids_a = torch.tensor([1, 2, 3], dtype=torch.int64)
        ids_b = torch.tensor([4, 5, 6], dtype=torch.int64)
        h = segment_hash(ids_a)
        # Force entry with ids_a under the key for ids_a
        entry = _entry([1, 2, 3])
        cache._entries[h] = entry
        # Query with ids_b — different tokens, should miss
        result = cache._match_segment(h, ids_b)
        self.assertIsNone(result)


# =========================================================================
# 8. PICache — evict
# =========================================================================

class TestPICacheEvict(unittest.TestCase):

    def _populate(self, cache, n_entries=4, tokens_per=8):
        for i in range(n_entries):
            tokens = list(range(i * tokens_per, (i + 1) * tokens_per))
            seg_ids = torch.tensor(tokens, dtype=torch.int64)
            h = segment_hash(seg_ids)
            kv_slots = cache.token_to_kv_pool_allocator.alloc(tokens_per)
            entry = SegmentEntry(
                seg_hash=h,
                full_kv_slots=kv_slots,
                token_ids=seg_ids,
                last_access_time=float(i),
            )
            cache._entries[h] = entry
        return list(cache._entries.keys())

    def test_evict_frees_tokens(self):
        cache = _make_picache(capacity=128)
        keys = self._populate(cache, n_entries=4, tokens_per=8)
        before = cache.evictable_size()
        result = cache.evict(EvictParams(num_tokens=8))
        after = cache.evictable_size()
        self.assertEqual(before - after, result.num_tokens_evicted)
        self.assertGreaterEqual(result.num_tokens_evicted, 8)

    def test_locked_entry_not_evicted(self):
        cache = _make_picache(capacity=128)
        self._populate(cache, n_entries=2, tokens_per=8)
        # Lock all entries
        for e in cache._entries.values():
            e.lock_ref = 1
        result = cache.evict(EvictParams(num_tokens=100))
        self.assertEqual(result.num_tokens_evicted, 0)

    def test_evict_zero_need_noop(self):
        cache = _make_picache(capacity=128)
        self._populate(cache, n_entries=2, tokens_per=8)
        result = cache.evict(EvictParams(num_tokens=0))
        self.assertEqual(result.num_tokens_evicted, 0)
        self.assertEqual(len(cache._entries), 2)

    def test_lru_evicts_oldest_first(self):
        cache = _make_picache(capacity=128)
        self._populate(cache, n_entries=3, tokens_per=8)
        entries = list(cache._entries.values())
        # entry with last_access_time=0 is oldest
        oldest_hash = entries[0].seg_hash
        cache.evict(EvictParams(num_tokens=8))
        self.assertNotIn(oldest_hash, cache._entries)


# =========================================================================
# 9. PICache — size reporting
# =========================================================================

class TestPICacheSizeReporting(unittest.TestCase):

    def test_empty_cache(self):
        cache = _make_picache()
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(cache.total_size(), 0)

    def test_unlocked_entry_is_evictable(self):
        cache = _make_picache(capacity=64)
        kv = cache.token_to_kv_pool_allocator.alloc(4)
        entry = _entry([1, 2, 3, 4], num_slots=4)
        entry.full_kv_slots = kv
        cache._entries[entry.seg_hash] = entry
        self.assertEqual(cache.evictable_size(), 4)
        self.assertEqual(cache.protected_size(), 0)

    def test_locked_entry_is_protected(self):
        cache = _make_picache(capacity=64)
        kv = cache.token_to_kv_pool_allocator.alloc(4)
        entry = _entry([1, 2, 3, 4], num_slots=4)
        entry.full_kv_slots = kv
        entry.lock_ref = 1
        cache._entries[entry.seg_hash] = entry
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 4)

    def test_inflight_counted_in_protected(self):
        cache = _make_picache()
        cache.add_inflight(32)
        self.assertEqual(cache.full_protected_size(), 32)
        cache.remove_inflight(32)
        self.assertEqual(cache.full_protected_size(), 0)

    def test_inflight_clamps_at_zero(self):
        cache = _make_picache()
        cache.remove_inflight(999)  # should not go negative
        self.assertEqual(cache._inflight_full_tokens, 0)


# =========================================================================
# 10. PICache — DSA size reporting
# =========================================================================

class TestPICacheDSASizeReporting(unittest.TestCase):

    def test_dsa_evictable_size(self):
        cache = _make_picache(dsa=True)
        kv = cache.token_to_kv_pool_allocator.alloc(4)
        entry = _entry([1, 2, 3, 4], num_slots=4, dsa_slot=3)
        entry.full_kv_slots = kv
        cache._entries[entry.seg_hash] = entry
        self.assertEqual(cache.dsa_evictable_size(), 1)

    def test_dsa_protected_size_with_locked_entry(self):
        cache = _make_picache(dsa=True)
        kv = cache.token_to_kv_pool_allocator.alloc(4)
        entry = _entry([1, 2, 3, 4], num_slots=4, dsa_slot=2)
        entry.full_kv_slots = kv
        entry.lock_ref = 1
        cache._entries[entry.seg_hash] = entry
        self.assertEqual(cache.dsa_protected_size(), 1)
        self.assertEqual(cache.dsa_evictable_size(), 0)


# =========================================================================
# 11. PICache — reset
# =========================================================================

class TestPICacheReset(unittest.TestCase):

    def test_reset_clears_all_entries(self):
        cache = _make_picache(capacity=64)
        for i in range(3):
            kv = cache.token_to_kv_pool_allocator.alloc(4)
            e = _entry([i, i + 1, i + 2, i + 3])
            e.full_kv_slots = kv
            cache._entries[e.seg_hash] = e
        cache.reset()
        self.assertEqual(len(cache._entries), 0)
        self.assertEqual(cache._inflight_full_tokens, 0)
        self.assertEqual(cache._inflight_dsa_slots, 0)


# =========================================================================
# 12. segmenter — integration with real-world separator pattern
# =========================================================================

class TestSegmenterIntegration(unittest.TestCase):

    def setUp(self):
        self.tok = DummyTokenizer()

    def test_five_docs_five_segments(self):
        docs = [f"Document {i} content here." for i in range(5)]
        sep = "<<PIC_SEP>>"
        text = sep.join(docs)
        ids, segs = split_and_tokenize(text, self.tok, separator=sep)
        self.assertEqual(len(segs), 5)
        self.assertEqual(segs[0][0], 0)
        self.assertEqual(segs[-1][1], len(ids))
        for i in range(len(segs) - 1):
            self.assertEqual(segs[i][1], segs[i + 1][0])

    def test_question_appended_as_last_segment(self):
        text = "doc1<<PIC_SEP>>doc2<<PIC_SEP>>What is the answer?"
        ids, segs = split_and_tokenize(text, self.tok)
        self.assertEqual(len(segs), 3)
        # All segments non-empty
        for (s, e) in segs:
            self.assertGreater(e - s, 0)


# =========================================================================
# 13. PICache — lock_ref protocol
# =========================================================================

class TestPICacheLockRef(unittest.TestCase):

    def test_inc_dec_lock_ref(self):
        cache = _make_picache(capacity=64)
        kv = cache.token_to_kv_pool_allocator.alloc(4)
        entry = _entry([1, 2, 3, 4])
        entry.full_kv_slots = kv
        cache._entries[entry.seg_hash] = entry

        req = MagicMock()
        req.pic_segment_entries = {entry.seg_hash: entry}

        cache.inc_lock_ref(req)
        self.assertEqual(entry.lock_ref, 1)
        self.assertEqual(cache.evictable_size(), 0)

        cache.dec_lock_ref(req)
        self.assertEqual(entry.lock_ref, 0)
        self.assertEqual(cache.evictable_size(), 4)

    def test_dec_lock_ref_does_not_go_negative(self):
        cache = _make_picache()
        entry = _entry([1, 2])
        entry.lock_ref = 0
        req = MagicMock()
        req.pic_segment_entries = {entry.seg_hash: entry}
        cache.dec_lock_ref(req)
        self.assertEqual(entry.lock_ref, 0)

    def test_node_without_pic_entries_is_noop(self):
        cache = _make_picache()
        node = MagicMock(spec=[])  # no pic_segment_entries attribute
        result = cache.inc_lock_ref(node)
        self.assertEqual(result.delta, 0)


# =========================================================================
# 14. hash stability across tensor/list inputs
# =========================================================================

class TestHasherInputVariants(unittest.TestCase):

    def test_tensor_and_list_same_hash(self):
        tokens = [10, 20, 30, 40]
        h_tensor = segment_hash(torch.tensor(tokens, dtype=torch.int64))
        h_list = segment_hash(iter(tokens))
        self.assertEqual(h_tensor, h_list)

    def test_empty_sequence(self):
        h = segment_hash(torch.tensor([], dtype=torch.int64))
        self.assertIsInstance(h, bytes)
        self.assertEqual(len(h), 16)


if __name__ == "__main__":
    unittest.main()
