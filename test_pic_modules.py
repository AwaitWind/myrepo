#!/usr/bin/env python3
"""
PIC 单元测试脚本 — 无需真实 GPU 即可运行

测试覆盖：
  1. hasher.py      — segment_hash 正确性、确定性、碰撞抵抗
  2. segmenter.py   — split_and_tokenize 不变量 S1-S5
  3. eviction.py    — 所有 7 种淘汰策略的优先级排序
  4. picache.py     — PICache 的 match_prefix / _insert_segment /
                      cache_unfinished_req / evict 核心逻辑
  5. pic_alloc.py   — DSAStatePool 分配/释放/清零

用法：
    python test_pic_modules.py              # 运行全部测试
    python test_pic_modules.py TestHasher   # 只运行指定类

注：picache 相关测试使用 Mock 对象替代真实的 KV pool，无需 GPU。
"""

from __future__ import annotations

import sys
import time
import unittest
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import torch

# ============================================================
# 被测模块导入
# ============================================================
from sglang.srt.pic.eviction import (
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)
from sglang.srt.pic.hasher import segment_hash
from sglang.srt.pic.picache import PICache, SegmentEntry
from sglang.srt.pic.segmenter import split_and_tokenize


# ============================================================
# 测试用辅助工具
# ============================================================

def _make_entry(
    seg_hash: bytes,
    ids: List[int],
    kv_slots: Optional[List[int]] = None,
    start_pos: int = 0,
    hit_count: int = 0,
    last_access_time: float = 0.0,
    creation_time: float = 0.0,
    priority: int = 0,
    lock_ref: int = 0,
) -> SegmentEntry:
    """构造一个 SegmentEntry 测试对象。"""
    if kv_slots is None:
        kv_slots = [0]
    return SegmentEntry(
        seg_hash=seg_hash,
        full_kv_slots=torch.tensor(kv_slots, dtype=torch.int64),
        token_ids=torch.tensor(ids, dtype=torch.int64),
        start_pos=start_pos,
        hit_count=hit_count,
        last_access_time=last_access_time,
        creation_time=creation_time,
        priority=priority,
        lock_ref=lock_ref,
    )


def _make_mock_pic_cache() -> Tuple[PICache, MagicMock, MagicMock]:
    """构造一个带 Mock 依赖的 PICache 实例（不需要 GPU）。"""
    req_to_token_pool = MagicMock()
    req_to_token_pool.req_to_token = torch.zeros((64, 2048), dtype=torch.int64)

    token_to_kv_pool = MagicMock()
    # 为 alloc 返回连续整数槽位，每次调用递增
    _slot_counter = [0]

    def alloc_slots(n: int):
        start = _slot_counter[0]
        _slot_counter[0] += n
        return torch.arange(start, start + n, dtype=torch.int64)

    token_to_kv_pool.alloc.side_effect = alloc_slots
    token_to_kv_pool.device = torch.device("cpu")

    cache = PICache(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool,
        dsa_state_pool=None,
        page_size=1,
        disable=False,
    )
    return cache, req_to_token_pool, token_to_kv_pool


# ============================================================
# 1. hasher.py 测试
# ============================================================

class TestHasher(unittest.TestCase):
    """segment_hash 正确性、确定性、格式测试。"""

    def test_returns_bytes_of_length_16(self):
        """哈希结果必须是 16 字节 bytes。"""
        ids = [1, 2, 3, 4, 5]
        h = segment_hash(ids)
        self.assertIsInstance(h, bytes)
        self.assertEqual(len(h), 16)

    def test_deterministic_list_input(self):
        """相同 token id 列表，两次调用结果相同。"""
        ids = [100, 200, 300]
        h1 = segment_hash(ids)
        h2 = segment_hash(ids)
        self.assertEqual(h1, h2)

    def test_deterministic_tensor_input(self):
        """torch.Tensor 输入与列表输入结果一致（协议兼容）。"""
        ids = [10, 20, 30, 40]
        h_list = segment_hash(ids)
        h_tensor = segment_hash(torch.tensor(ids, dtype=torch.int64))
        self.assertEqual(h_list, h_tensor)

    def test_different_ids_different_hash(self):
        """不同 token id 序列产生不同哈希。"""
        h1 = segment_hash([1, 2, 3])
        h2 = segment_hash([1, 2, 4])   # 只改了最后一个
        self.assertNotEqual(h1, h2)

    def test_order_matters(self):
        """顺序不同的 token id 序列产生不同哈希。"""
        h1 = segment_hash([1, 2, 3])
        h2 = segment_hash([3, 2, 1])
        self.assertNotEqual(h1, h2)

    def test_empty_ids(self):
        """空序列不崩溃，返回 16 字节。"""
        h = segment_hash([])
        self.assertIsInstance(h, bytes)
        self.assertEqual(len(h), 16)

    def test_single_token(self):
        """单 token 序列不崩溃。"""
        h = segment_hash([42])
        self.assertIsInstance(h, bytes)
        self.assertEqual(len(h), 16)

    def test_large_ids(self):
        """大量 token（如 20,000 个）不崩溃且结果确定。"""
        ids = list(range(20000))
        h1 = segment_hash(ids)
        h2 = segment_hash(ids)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_int32_serialization_consistent(self):
        """验证 int32 小端序序列化的一致性：直接构造 bytes 应与哈希函数一致。"""
        import array
        import hashlib
        ids = [1, 2, 3]
        raw = array.array("i", ids).tobytes()
        expected = hashlib.sha256(raw).digest()[:16]
        self.assertEqual(segment_hash(ids), expected)


# ============================================================
# 2. segmenter.py 测试
# ============================================================

class TestSegmenter(unittest.TestCase):
    """split_and_tokenize 不变量 S1-S5 测试。"""

    def setUp(self):
        # 简单 tokenizer：每个非空字符 → ord % 128
        tok = MagicMock()
        tok.encode.side_effect = lambda t, **kw: [ord(c) % 128 for c in t if c.strip()]
        self.tokenizer = tok

    def test_s1_first_segment_starts_at_zero(self):
        """不变量 S1：第一段起始位置为 0。"""
        text = "Hello<<PIC_SEP>>World"
        _, segs = split_and_tokenize(text, self.tokenizer)
        self.assertGreater(len(segs), 0)
        self.assertEqual(segs[0][0], 0, "S1 违反：pic_segments[0][0] != 0")

    def test_s2_last_segment_ends_at_total_len(self):
        """不变量 S2：最后一段末尾位置等于 input_ids 总长。"""
        text = "Hello<<PIC_SEP>>World<<PIC_SEP>>Foo"
        ids, segs = split_and_tokenize(text, self.tokenizer)
        self.assertEqual(segs[-1][1], len(ids), "S2 违反：pic_segments[-1][1] != len(input_ids)")

    def test_s3_adjacent_segments_connected(self):
        """不变量 S3：相邻段端点严格衔接。"""
        text = "A<<PIC_SEP>>B<<PIC_SEP>>C"
        _, segs = split_and_tokenize(text, self.tokenizer)
        for i in range(len(segs) - 1):
            self.assertEqual(
                segs[i][1], segs[i + 1][0],
                f"S3 违反：segs[{i}][1]={segs[i][1]} != segs[{i+1}][0]={segs[i+1][0]}"
            )

    def test_s4_no_empty_segments(self):
        """不变量 S4：不产生空段（start < end）。"""
        # 连续分隔符会产生空字符串，应被跳过
        text = "A<<PIC_SEP>><<PIC_SEP>>B"
        _, segs = split_and_tokenize(text, self.tokenizer)
        for (s, e) in segs:
            self.assertLess(s, e, f"S4 违反：存在空段 ({s}, {e})")

    def test_s5_separator_produces_no_tokens(self):
        """不变量 S5：分隔符本身不产生 token，段 token 只来自各部分。"""
        text = "Hello<<PIC_SEP>>World"
        ids, segs = split_and_tokenize(text, self.tokenizer)
        # 分段 token 的总数应等于 "Hello" 和 "World" 分别 encode 后的总数
        ids_hello = self.tokenizer.encode("Hello", add_special_tokens=False)
        ids_world = self.tokenizer.encode("World", add_special_tokens=False)
        self.assertEqual(len(ids), len(ids_hello) + len(ids_world))

    def test_no_separator_single_segment(self):
        """无分隔符时，整个文本作为单一段。"""
        text = "Hello World"
        ids, segs = split_and_tokenize(text, self.tokenizer)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0], (0, len(ids)))

    def test_three_segments_correct_split(self):
        """3 段 prompt 正确产生 3 段（验证 start/end 区间）。"""
        tok = MagicMock()
        # "AAA" → 3 tokens, "BB" → 2 tokens, "C" → 1 token
        tok.encode.side_effect = lambda t, **kw: [i + 1 for i in range(len(t.strip()))]
        text = "AAA<<PIC_SEP>>BB<<PIC_SEP>>C"
        ids, segs = split_and_tokenize(text, tok)
        self.assertEqual(len(segs), 3)
        self.assertEqual(segs[0], (0, 3))
        self.assertEqual(segs[1], (3, 5))
        self.assertEqual(segs[2], (5, 6))

    def test_ids_content_matches_segments(self):
        """segment 区间切出的 ids 子数组确实是对应段的 token。"""
        tok = MagicMock()
        tok.encode.side_effect = lambda t, **kw: [ord(c) for c in t.strip()]
        text = "abc<<PIC_SEP>>de"
        ids, segs = split_and_tokenize(text, tok)
        # 第一段：'abc' → [97, 98, 99]
        self.assertEqual(ids[segs[0][0]:segs[0][1]], [97, 98, 99])
        # 第二段：'de' → [100, 101]
        self.assertEqual(ids[segs[1][0]:segs[1][1]], [100, 101])

    def test_pic_w1_prompt_construction(self):
        """测试规格文档 §2.4 中的 PIC_W1 prompt 结构。"""
        SEP = "<<PIC_SEP>>"
        C1 = "cats " * 5
        Q = "question"
        pic_w1 = f"sys{SEP}{C1}{SEP}{Q}"
        ids, segs = split_and_tokenize(pic_w1, self.tokenizer)
        # 应有 3 段：sys / C1 / Q
        self.assertEqual(len(segs), 3)
        # S1
        self.assertEqual(segs[0][0], 0)
        # S2
        self.assertEqual(segs[-1][1], len(ids))
        # S3
        for i in range(len(segs) - 1):
            self.assertEqual(segs[i][1], segs[i + 1][0])


# ============================================================
# 3. eviction.py 测试
# ============================================================

class TestEvictionStrategies(unittest.TestCase):
    """7 种淘汰策略的优先级排序测试。"""

    def _make_entries(self) -> List[SegmentEntry]:
        """构造 3 个具有不同属性的测试 entry。"""
        e1 = _make_entry(b"\x01" * 16, [1], last_access_time=1.0, creation_time=1.0,
                         hit_count=5, priority=3)
        e2 = _make_entry(b"\x02" * 16, [2], last_access_time=2.0, creation_time=3.0,
                         hit_count=1, priority=1)
        e3 = _make_entry(b"\x03" * 16, [3], last_access_time=3.0, creation_time=2.0,
                         hit_count=3, priority=2)
        return [e1, e2, e3]

    def test_lru_oldest_first(self):
        """LRU：last_access_time 最小的优先淘汰。"""
        entries = self._make_entries()
        strategy = LRUStrategy()
        sorted_entries = sorted(entries, key=strategy.get_priority)
        self.assertEqual(sorted_entries[0].last_access_time, 1.0)
        self.assertEqual(sorted_entries[-1].last_access_time, 3.0)

    def test_lfu_least_hit_first(self):
        """LFU：hit_count 最小的优先淘汰。"""
        entries = self._make_entries()
        strategy = LFUStrategy()
        sorted_entries = sorted(entries, key=strategy.get_priority)
        self.assertEqual(sorted_entries[0].hit_count, 1)
        self.assertEqual(sorted_entries[-1].hit_count, 5)

    def test_fifo_oldest_creation_first(self):
        """FIFO：creation_time 最小的优先淘汰。"""
        entries = self._make_entries()
        strategy = FIFOStrategy()
        sorted_entries = sorted(entries, key=strategy.get_priority)
        self.assertEqual(sorted_entries[0].creation_time, 1.0)
        self.assertEqual(sorted_entries[-1].creation_time, 3.0)

    def test_mru_most_recent_first(self):
        """MRU：last_access_time 最大的优先淘汰。"""
        entries = self._make_entries()
        strategy = MRUStrategy()
        sorted_entries = sorted(entries, key=strategy.get_priority)
        self.assertEqual(sorted_entries[0].last_access_time, 3.0)
        self.assertEqual(sorted_entries[-1].last_access_time, 1.0)

    def test_filo_newest_creation_first(self):
        """FILO：creation_time 最大的优先淘汰（最晚创建先淘汰）。"""
        entries = self._make_entries()
        strategy = FILOStrategy()
        sorted_entries = sorted(entries, key=strategy.get_priority)
        self.assertEqual(sorted_entries[0].creation_time, 3.0)
        self.assertEqual(sorted_entries[-1].creation_time, 1.0)

    def test_priority_strategy(self):
        """Priority：priority 值最小的优先淘汰。"""
        entries = self._make_entries()
        strategy = PriorityStrategy()
        sorted_entries = sorted(entries, key=strategy.get_priority)
        self.assertEqual(sorted_entries[0].priority, 1)
        self.assertEqual(sorted_entries[-1].priority, 3)

    def test_slru_trial_before_protected(self):
        """SLRU：hit_count < threshold 的条目先于 hit_count >= threshold 的条目淘汰。"""
        strategy = SLRUStrategy(threshold=3)
        # e1.hit_count=5 >= 3 → 保护段
        # e2.hit_count=1 < 3  → 试用段（优先淘汰）
        # e3.hit_count=3 >= 3 → 保护段
        entries = self._make_entries()
        sorted_entries = sorted(entries, key=strategy.get_priority)
        trial_entries = [e for e in sorted_entries if e.hit_count < 3]
        protected_entries = [e for e in sorted_entries if e.hit_count >= 3]
        if trial_entries and protected_entries:
            trial_idx = sorted_entries.index(trial_entries[0])
            first_protected_idx = min(sorted_entries.index(e) for e in protected_entries)
            self.assertLess(trial_idx, first_protected_idx,
                            "SLRU：试用段应先于保护段出现在排序结果中")

    def test_slru_within_trial_sorted_by_time(self):
        """SLRU：同段内按 last_access_time 排序。"""
        strategy = SLRUStrategy(threshold=3)
        e_old = _make_entry(b"\xaa" * 16, [1], hit_count=1, last_access_time=1.0)
        e_new = _make_entry(b"\xbb" * 16, [2], hit_count=2, last_access_time=5.0)
        sorted_entries = sorted([e_new, e_old], key=strategy.get_priority)
        self.assertEqual(sorted_entries[0].last_access_time, 1.0)


# ============================================================
# 4. picache.py 测试
# ============================================================

class TestPICache(unittest.TestCase):
    """PICache 核心逻辑测试（Mock 掉 GPU 依赖）。"""

    def setUp(self):
        self.cache, self.req_to_token_pool, self.kv_pool = _make_mock_pic_cache()

    # --------------------------------------------------
    # _insert_segment / _match_segment
    # --------------------------------------------------

    def test_insert_and_match_segment(self):
        """插入一个段后，相同 token ids 应能匹配到。"""
        ids = [10, 20, 30]
        ids_tensor = torch.tensor(ids, dtype=torch.int64)
        seg_h = segment_hash(ids)
        kv_slots = torch.tensor([100, 101, 102], dtype=torch.int64)

        entry = self.cache._insert_segment(seg_h, ids_tensor, kv_slots, start_pos=0)
        self.assertIsNotNone(entry)
        self.assertEqual(self.cache._entries[seg_h].start_pos, 0)

        found = self.cache._match_segment(seg_h, ids_tensor)
        self.assertIsNotNone(found)
        self.assertTrue(torch.equal(found.token_ids, ids_tensor))

    def test_match_segment_wrong_ids_returns_none(self):
        """哈希相同但 token ids 不同（碰撞）时，_match_segment 返回 None。"""
        ids = [10, 20, 30]
        ids_tensor = torch.tensor(ids, dtype=torch.int64)
        seg_h = segment_hash(ids)
        kv_slots = torch.tensor([100, 101, 102], dtype=torch.int64)
        self.cache._insert_segment(seg_h, ids_tensor, kv_slots)

        wrong_ids = torch.tensor([10, 20, 99], dtype=torch.int64)
        found = self.cache._match_segment(seg_h, wrong_ids)
        self.assertIsNone(found)

    def test_insert_updates_creation_time(self):
        """插入的 entry 应有非零的 creation_time。"""
        ids_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        seg_h = segment_hash([1, 2, 3])
        kv_slots = torch.tensor([0, 1, 2], dtype=torch.int64)
        before = time.monotonic()
        entry = self.cache._insert_segment(seg_h, ids_tensor, kv_slots)
        after = time.monotonic()
        self.assertGreaterEqual(entry.creation_time, before)
        self.assertLessEqual(entry.creation_time, after)

    def test_insert_clones_kv_slots(self):
        """_insert_segment 应 clone kv_slots，修改原始张量不影响 entry。"""
        ids_tensor = torch.tensor([5, 6, 7], dtype=torch.int64)
        seg_h = segment_hash([5, 6, 7])
        kv_slots = torch.tensor([0, 1, 2], dtype=torch.int64)
        entry = self.cache._insert_segment(seg_h, ids_tensor, kv_slots)
        kv_slots[0] = 999  # 修改原始张量
        self.assertNotEqual(entry.full_kv_slots[0].item(), 999)

    # --------------------------------------------------
    # match_prefix（通过 Mock Req 对象）
    # --------------------------------------------------

    def _make_mock_req(self, segments: List[Tuple[int, int]], ids: List[int]):
        """构造一个满足 match_prefix 需要的 Mock Req 对象。"""
        req = MagicMock()
        req.pic_segments = segments
        req.get_fill_ids.return_value = ids
        return req

    def _match_prefix_params(self, req):
        from array import array
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        # MatchPrefixParams 需要 key 参数（RadixKey），PIC 不使用 key 但接口要求提供
        dummy_key = RadixKey(token_ids=array("q", []))
        return MatchPrefixParams(key=dummy_key, req=req)

    def test_match_prefix_hit_and_miss(self):
        """第一段命中，第二段未命中，最后一段永远为 None。"""
        ids = [1, 2, 3, 4, 5, 6]
        segs = [(0, 3), (3, 5), (5, 6)]

        # 先插入第一段
        seg1_ids = torch.tensor(ids[0:3], dtype=torch.int64)
        seg1_hash = segment_hash(ids[0:3])
        self.cache._insert_segment(seg1_hash, seg1_ids, torch.tensor([10, 11, 12]))

        req = self._make_mock_req(segs, ids)
        result = self.cache.match_prefix(self._match_prefix_params(req))

        entries = result.pic_segment_entries
        self.assertEqual(len(entries), 3)
        self.assertIsNotNone(entries[0])   # 第一段命中
        self.assertIsNone(entries[1])      # 第二段未命中
        self.assertIsNone(entries[2])      # 最后一段永远 None

    def test_match_prefix_updates_hit_count(self):
        """命中的段，hit_count 和 last_access_time 应被更新。"""
        ids = [10, 20, 30, 40]
        segs = [(0, 2), (2, 4)]
        seg1_ids = torch.tensor(ids[0:2], dtype=torch.int64)
        seg1_hash = segment_hash(ids[0:2])
        entry = self.cache._insert_segment(seg1_hash, seg1_ids, torch.tensor([0, 1]))
        entry.hit_count = 0
        entry.last_access_time = 0.0

        req = self._make_mock_req(segs, ids)
        self.cache.match_prefix(self._match_prefix_params(req))

        self.assertEqual(entry.hit_count, 1)
        self.assertGreater(entry.last_access_time, 0.0)

    def test_match_prefix_device_indices_always_empty(self):
        """match_prefix 的 device_indices 始终为空（PIC 协议约束）。"""
        ids = [1, 2, 3]
        segs = [(0, 2), (2, 3)]
        req = self._make_mock_req(segs, ids)
        result = self.cache.match_prefix(self._match_prefix_params(req))
        self.assertEqual(result.device_indices.numel(), 0)

    def test_match_prefix_none_req_returns_empty(self):
        """req 为 None 时，match_prefix 返回 empty result（不崩溃）。"""
        result = self.cache.match_prefix(self._match_prefix_params(None))
        self.assertIsNone(result.pic_segment_entries)

    def test_match_prefix_disabled_returns_empty(self):
        """disable=True 时，match_prefix 始终返回空结果。"""
        cache, _, _ = _make_mock_pic_cache()
        cache.disable = True
        ids = [1, 2, 3, 4]
        segs = [(0, 2), (2, 4)]
        cache._insert_segment(segment_hash(ids[:2]), torch.tensor(ids[:2]), torch.tensor([0, 1]))
        req = self._make_mock_req(segs, ids)
        result = cache.match_prefix(self._match_prefix_params(req))
        self.assertIsNone(result.pic_segment_entries)

    def test_match_prefix_no_pic_segments_returns_empty(self):
        """req 无 pic_segments 时，返回 empty result。"""
        req = MagicMock()
        req.pic_segments = []
        result = self.cache.match_prefix(self._match_prefix_params(req))
        self.assertIsNone(result.pic_segment_entries)

    # --------------------------------------------------
    # evict
    # --------------------------------------------------

    def test_evict_removes_unlocked_entries(self):
        """evict 应移除 lock_ref == 0 的条目，直到满足 need 为止。"""
        for i in range(3):
            ids = [i * 10, i * 10 + 1]
            ids_t = torch.tensor(ids, dtype=torch.int64)
            h = segment_hash(ids)
            kv = torch.tensor([i * 2, i * 2 + 1], dtype=torch.int64)
            entry = self.cache._insert_segment(h, ids_t, kv)
            entry.lock_ref = 0

        self.assertEqual(len(self.cache._entries), 3)

        from sglang.srt.mem_cache.base_prefix_cache import EvictParams
        self.cache.evict(EvictParams(num_tokens=2, dsa_num=0))
        self.assertLess(len(self.cache._entries), 3)

    def test_evict_does_not_remove_locked_entries(self):
        """evict 不应移除 lock_ref > 0 的条目。"""
        ids_locked = [99, 100, 101]
        h_locked = segment_hash(ids_locked)
        kv_locked = torch.tensor([900, 901, 902], dtype=torch.int64)
        locked_entry = self.cache._insert_segment(
            h_locked, torch.tensor(ids_locked, dtype=torch.int64), kv_locked
        )
        locked_entry.lock_ref = 1

        from sglang.srt.mem_cache.base_prefix_cache import EvictParams
        self.cache.evict(EvictParams(num_tokens=100, dsa_num=0))
        self.assertIn(h_locked, self.cache._entries)

    # --------------------------------------------------
    # reset
    # --------------------------------------------------

    def test_reset_clears_all_entries(self):
        """reset 后 _entries 应为空字典。"""
        ids = [7, 8, 9]
        h = segment_hash(ids)
        self.cache._insert_segment(
            h, torch.tensor(ids, dtype=torch.int64), torch.tensor([0, 1, 2])
        )
        self.assertGreater(len(self.cache._entries), 0)
        self.cache.reset()
        self.assertEqual(len(self.cache._entries), 0)

    def test_reset_clears_inflight_counters(self):
        """reset 后 _inflight 计数器应清零。"""
        self.cache._inflight_full_tokens = 5
        self.cache._inflight_dsa_slots = 3
        self.cache.reset()
        self.assertEqual(self.cache._inflight_full_tokens, 0)
        self.assertEqual(self.cache._inflight_dsa_slots, 0)

    # --------------------------------------------------
    # add_inflight / remove_inflight
    # --------------------------------------------------

    def test_add_inflight_increments_counter(self):
        """add_inflight 应增加 _inflight_full_tokens 计数。"""
        self.cache._inflight_full_tokens = 0
        self.cache.add_inflight(10, 0)
        self.assertEqual(self.cache._inflight_full_tokens, 10)

    def test_remove_inflight_decrements_counter(self):
        """remove_inflight 应减少 _inflight_full_tokens 计数。"""
        self.cache._inflight_full_tokens = 10
        self.cache.remove_inflight(4, 0)
        self.assertEqual(self.cache._inflight_full_tokens, 6)

    # --------------------------------------------------
    # supports_* 接口
    # --------------------------------------------------

    def test_supports_mamba_returns_false(self):
        self.assertFalse(self.cache.supports_mamba())

    def test_supports_swa_returns_false(self):
        self.assertFalse(self.cache.supports_swa())

    def test_is_chunk_cache_returns_false(self):
        self.assertFalse(self.cache.is_chunk_cache())


# ============================================================
# 5. pic_alloc.py — DSAStatePool 测试
# ============================================================

class TestDSAStatePool(unittest.TestCase):
    """DSAStatePool 分配/释放/清零逻辑测试（CPU 设备）。"""

    def _make_pool(self, pool_size: int = 4):
        from sglang.srt.pic.pic_alloc import DSAStatePool
        return DSAStatePool(
            num_dsa_layers=2,
            pool_size=pool_size,
            dsa_state_shape=(8,),
            dsa_dtype=torch.float32,
            device=torch.device("cpu"),
        )

    def test_initial_available_size(self):
        """初始可用槽位数等于 pool_size。"""
        pool = self._make_pool(pool_size=4)
        self.assertEqual(pool.available_size(), 4)

    def test_alloc_returns_tensor(self):
        """alloc 返回 int64 Tensor，槽位 id >= 1。"""
        pool = self._make_pool(pool_size=4)
        slots = pool.alloc(2)
        self.assertIsNotNone(slots)
        self.assertEqual(slots.dtype, torch.int64)
        self.assertEqual(slots.numel(), 2)
        for s in slots.tolist():
            self.assertGreaterEqual(s, 1)

    def test_alloc_decreases_available(self):
        """alloc n 个后，available_size 减少 n。"""
        pool = self._make_pool(pool_size=4)
        pool.alloc(2)
        self.assertEqual(pool.available_size(), 2)

    def test_alloc_exceeds_returns_none(self):
        """alloc 超出可用槽位时返回 None。"""
        pool = self._make_pool(pool_size=2)
        result = pool.alloc(3)
        self.assertIsNone(result)

    def test_free_increases_available(self):
        """free 后可用槽位数恢复。"""
        pool = self._make_pool(pool_size=4)
        slots = pool.alloc(3)
        self.assertEqual(pool.available_size(), 1)
        pool.free(slots)
        self.assertEqual(pool.available_size(), 4)

    def test_alloc_zeros_state(self):
        """重新分配的槽位应被清零（防止 stale state）。"""
        pool = self._make_pool(pool_size=4)
        slots = pool.alloc(1)
        slot_id = int(slots[0])
        # 写入非零值
        pool.dsa_state[:, slot_id, :] = 9.0
        # 释放再重新分配
        pool.free(slots)
        slots2 = pool.alloc(1)
        slot_id2 = int(slots2[0])
        state = pool.dsa_state[:, slot_id2, :]
        self.assertTrue(torch.all(state == 0).item(), "重新分配后状态未清零")

    def test_slot_zero_never_allocated(self):
        """Slot 0 是 dummy，永远不被分配。"""
        pool = self._make_pool(pool_size=4)
        all_slots = pool.alloc(4)
        self.assertIsNotNone(all_slots)
        self.assertNotIn(0, all_slots.tolist())

    def test_get_set_state(self):
        """get_state / set_state 接口正确读写。"""
        pool = self._make_pool(pool_size=4)
        slots = pool.alloc(1)
        slot_id = int(slots[0])
        state_val = torch.ones(8, dtype=torch.float32) * 3.14
        pool.set_state(slot_id, state_val)
        retrieved = pool.get_state(slot_id)
        # get_state 返回的是 view，形状为 (num_dsa_layers, 8)
        self.assertTrue(
            torch.allclose(retrieved[0], state_val),
            f"get_state 第 0 层与 set_state 不一致: {retrieved[0]} vs {state_val}"
        )


# ============================================================
# 6. 端到端流程：segment_hash → _insert_segment → match_prefix
# ============================================================

class TestEndToEndHashAndMatch(unittest.TestCase):
    """验证哈希 → 插入 → 匹配的完整流程。"""

    def _make_char_tokenizer(self):
        """字符级 tokenizer：每个字符 → ord % 64。"""
        tok = MagicMock()
        tok.encode.side_effect = lambda t, **kw: [ord(c) % 64 for c in t.replace(" ", "")]
        return tok

    def test_full_roundtrip_three_segments(self):
        """三段 prompt 的哈希-插入-匹配完整路径。"""
        cache, _, _ = _make_mock_pic_cache()
        tok = self._make_char_tokenizer()
        SEP = "<<PIC_SEP>>"
        C1 = "cats " * 5
        C2 = "dogs " * 5
        Q = "question"

        text = f"{C1}{SEP}{C2}{SEP}{Q}"
        ids, segs = split_and_tokenize(text, tok)

        self.assertEqual(len(segs), 3)
        self.assertEqual(segs[0][0], 0)
        self.assertEqual(segs[-1][1], len(ids))

        # 插入 C1 段
        seg0_ids = torch.tensor(ids[segs[0][0]:segs[0][1]], dtype=torch.int64)
        seg0_hash = segment_hash(seg0_ids.tolist())
        kv0 = torch.arange(len(seg0_ids), dtype=torch.int64)
        cache._insert_segment(seg0_hash, seg0_ids, kv0, start_pos=segs[0][0])

        req = MagicMock()
        req.pic_segments = segs
        req.get_fill_ids.return_value = ids

        from array import array as _array
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        dummy_key = RadixKey(token_ids=_array("q", []))
        result = cache.match_prefix(MatchPrefixParams(key=dummy_key, req=req))

        entries = result.pic_segment_entries
        self.assertIsNotNone(entries[0])   # C1 命中
        self.assertIsNone(entries[1])      # C2 未命中
        self.assertIsNone(entries[2])      # 最后段永远 None

    def test_cache_hit_after_warmup_noncontiguous(self):
        """
        模拟测试文档 §2.4 描述的 PIC 非连续复用场景：
        warmup 发了 PIC_W1（含 C1）和 PIC_W3（含 C3），
        正式请求 PIC_PROMPT（含 C1+C2+C3+Q）时，C1 和 C3 都命中（非连续）。
        """
        cache, _, _ = _make_mock_pic_cache()
        tok = self._make_char_tokenizer()
        SEP = "<<PIC_SEP>>"
        SYS = "You are a helpful assistant."
        C1 = "cats " * 20
        C2 = "dogs " * 20
        C3 = "birds " * 20
        Q = "which animal is in B"

        # Warmup 1: SYS + C1 + Q → 缓存 SYS 段和 C1 段
        w1_ids, w1_segs = split_and_tokenize(f"{SYS}{SEP}{C1}{SEP}{Q}", tok)
        for i, (s, e) in enumerate(w1_segs[:-1]):  # 排除最后一段 Q
            seg_ids = torch.tensor(w1_ids[s:e], dtype=torch.int64)
            seg_h = segment_hash(w1_ids[s:e])
            cache._insert_segment(seg_h, seg_ids, torch.arange(e - s, dtype=torch.int64))

        # Warmup 3: SYS + C3 + Q → 缓存 C3 段
        w3_ids, w3_segs = split_and_tokenize(f"{SYS}{SEP}{C3}{SEP}{Q}", tok)
        # 仅缓存 C3 段（第二段，非最后段）
        s3, e3 = w3_segs[1]
        seg_c3_ids = torch.tensor(w3_ids[s3:e3], dtype=torch.int64)
        seg_c3_h = segment_hash(w3_ids[s3:e3])
        if seg_c3_h not in cache._entries:
            cache._insert_segment(seg_c3_h, seg_c3_ids, torch.arange(1000, 1000 + e3 - s3, dtype=torch.int64))

        # 正式请求: SYS + C1 + C2 + C3 + Q
        full_text = f"{SYS}{SEP}{C1}{SEP}{C2}{SEP}{C3}{SEP}{Q}"
        full_ids, full_segs = split_and_tokenize(full_text, tok)

        req = MagicMock()
        req.pic_segments = full_segs
        req.get_fill_ids.return_value = full_ids

        from array import array as _array
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        dummy_key = RadixKey(token_ids=_array("q", []))
        result = cache.match_prefix(MatchPrefixParams(key=dummy_key, req=req))

        entries = result.pic_segment_entries
        self.assertEqual(len(entries), 5)  # SYS / C1 / C2 / C3 / Q

        # PIC 关键验证：C1 和 C3 命中，C2 未命中（非连续复用！）
        self.assertIsNotNone(entries[0], "SYS 段应命中缓存")
        self.assertIsNotNone(entries[1], "C1 段应命中缓存（warmup 1 填充）")
        self.assertIsNone(entries[2],    "C2 段不应命中（未 warmup）")
        self.assertIsNotNone(entries[3], "C3 段应命中缓存（warmup 3 填充，非连续复用！）")
        self.assertIsNone(entries[4],    "最后段 Q 永远为 None")

        # device_indices 始终为空
        self.assertEqual(result.device_indices.numel(), 0)

    def test_same_content_different_positions_same_hash(self):
        """相同 token ids 的段，无论出现在哪个位置，哈希相同（位置无关性）。"""
        ids1 = [10, 20, 30]
        ids2 = [10, 20, 30]  # 内容完全相同
        h1 = segment_hash(ids1)
        h2 = segment_hash(ids2)
        self.assertEqual(h1, h2, "相同内容的段应产生相同哈希")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
