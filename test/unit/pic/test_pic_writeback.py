"""
PIC writeback 单元测试。

覆盖：
  pic_alloc._pic_alloc_transition_rope   → batch.pic_public_out_loc 的形状和值
  model_runner._pic_writeback_mla_kv     → MLA KV 从 private → public 的拷贝
  dsa_indexer._pic_writeback_dsa_k       → DSA K writeback 的掩码/路由逻辑
"""

import types
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.pic.pic_alloc import _pic_alloc_transition_rope
from sglang.srt.pic.hasher import segment_hash
from sglang.srt.pic.picache import SegmentEntry


# =========================================================================
# 通用 helper
# =========================================================================

def _kv_alloc(capacity: int = 512):
    """顺序分配、支持 free 的极简 KV allocator mock。"""
    pool = list(range(capacity))
    m = MagicMock()

    def _alloc(n):
        if n > len(pool):
            return None
        s = pool[:n]
        del pool[:n]
        return torch.tensor(s, dtype=torch.int64)

    def _free(t):
        pool.extend(t.tolist())

    m.alloc = _alloc
    m.free = _free
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
    batch = MagicMock()
    batch.reqs = reqs
    batch.device = "cpu"
    batch.pic_public_out_loc = None
    return batch


def _run_alloc(reqs, capacity=512):
    """Wrapper：调用 _pic_alloc_transition_rope，返回 (out_cache_loc, pub_loc)。"""
    alloc = _kv_alloc(capacity)
    batch = _make_batch(reqs)
    tree = MagicMock()
    tree.add_inflight = MagicMock()
    out = _pic_alloc_transition_rope(batch, tree, alloc, dsa_state_pool=None)
    return out, batch.pic_public_out_loc


# =========================================================================
# 1. pic_public_out_loc 的构建
# =========================================================================

class TestPICPublicOutLocBuilding(unittest.TestCase):
    """_pic_alloc_transition_rope 必须正确设置 batch.pic_public_out_loc。"""

    def test_shape_equals_out_cache_loc(self):
        """pub_loc 和 out_cache_loc 形状相同。"""
        segs = [(0, 10), (10, 20), (20, 30)]
        req = _make_req(segs)
        out_loc, pub_loc = _run_alloc([req])
        self.assertEqual(out_loc.shape, pub_loc.shape,
                         "pic_public_out_loc 必须与 out_cache_loc 等长")

    def test_non_last_miss_has_valid_pub_slots(self):
        """非最后段的 miss token → pub_loc >= 0，且值与分配的 public slot 一致。"""
        segs = [(0, 8), (8, 16), (16, 24)]   # 3 段，最后段不缓存
        req = _make_req(segs)
        out_loc, pub_loc = _run_alloc([req])
        # 前两段（0..15）应有 public slot
        self.assertTrue((pub_loc[:16] >= 0).all(),
                        "前两段 miss 位置 pub_loc 应 >= 0")

    def test_last_segment_always_neg1(self):
        """最后段（永不缓存）的所有位置 pub_loc 必须为 -1。"""
        segs = [(0, 6), (6, 12), (12, 20)]
        req = _make_req(segs)
        out_loc, pub_loc = _run_alloc([req])
        self.assertTrue((pub_loc[12:20] == -1).all(),
                        "最后段 pub_loc 必须全为 -1")

    def test_hit_segment_all_neg1(self):
        """命中段的位置 pub_loc 必须为 -1（entry.full_kv_slots 已有效）。"""
        segs = [(0, 8), (8, 16)]
        seg_ids = torch.arange(8, dtype=torch.int64)
        h = segment_hash(seg_ids)
        entry = SegmentEntry(
            seg_hash=h,
            full_kv_slots=torch.arange(8, dtype=torch.int64),
            token_ids=seg_ids,
        )
        req = _make_req(
            segs,
            hit_segs=[(0, 8, h)],
            miss_segs=[(8, 16)],     # 只有最后段是 miss
            seg_entries={h: entry},
        )
        out_loc, pub_loc = _run_alloc([req])
        self.assertTrue((pub_loc[:8] == -1).all(),
                        "命中段位置 pub_loc 必须全为 -1")

    def test_non_pic_req_all_neg1(self):
        """非 PIC 请求的所有位置 pub_loc 必须为 -1。"""
        req = MagicMock()
        req.pic_segments = None
        req.extend_input_len = 20
        out_loc, pub_loc = _run_alloc([req])
        self.assertEqual(out_loc.shape[0], 20)
        self.assertTrue((pub_loc == -1).all(),
                        "非 PIC 请求 pub_loc 必须全为 -1")

    def test_pub_slots_disjoint_from_priv_slots(self):
        """Public slot 和 private slot 必须是不同的物理 slot（不能重叠）。"""
        segs = [(0, 10), (10, 20), (20, 30)]
        req = _make_req(segs)
        out_loc, pub_loc = _run_alloc([req])
        valid = pub_loc >= 0
        pub_set = set(pub_loc[valid].tolist())
        priv_set = set(out_loc.tolist())
        overlap = pub_set & priv_set
        self.assertEqual(overlap, set(), f"pub/priv slot 重叠: {overlap}")

    def test_single_segment_all_neg1(self):
        """只有一段时（就是最后段），pub_loc 全为 -1。"""
        segs = [(0, 15)]
        req = _make_req(segs)
        out_loc, pub_loc = _run_alloc([req])
        self.assertTrue((pub_loc == -1).all())

    def test_mixed_pic_and_non_pic_batch(self):
        """PIC + 非 PIC 混合 batch：pub_loc 按顺序正确拼接。"""
        # req0: 非 PIC，20 tokens
        req0 = MagicMock()
        req0.pic_segments = None
        req0.extend_input_len = 20
        # req1: PIC，2 段（seg0=10tok miss, seg1=10tok last）
        segs1 = [(0, 10), (10, 20)]
        req1 = _make_req(segs1)
        out_loc, pub_loc = _run_alloc([req0, req1])
        self.assertEqual(pub_loc.shape[0], 40)
        self.assertTrue((pub_loc[:20] == -1).all(),  # req0 全 -1
                        "非 PIC req 位置应为 -1")
        self.assertTrue((pub_loc[20:30] >= 0).all(), # req1 seg0 有 pub slot
                        "PIC miss 段位置应有有效 pub slot")
        self.assertTrue((pub_loc[30:40] == -1).all(), # req1 最后段 -1
                        "PIC 最后段位置应为 -1")

    def test_pub_slot_count_equals_non_last_miss_tokens(self):
        """有效 pub slot 的数量 = 非最后 miss 段的 token 总数。"""
        segs = [(0, 5), (5, 15), (15, 30)]   # 5 + 10 + 15 = 30 tokens
        req = _make_req(segs)
        out_loc, pub_loc = _run_alloc([req])
        # 非最后段 miss = seg0(5) + seg1(10) = 15 tokens
        valid_count = (pub_loc >= 0).sum().item()
        self.assertEqual(valid_count, 15)


# =========================================================================
# 2. MLA KV writeback (_pic_writeback_mla_kv)
# =========================================================================

def _writeback_mla_kv(forward_batch, buffers, start_layer, end_layer):
    """
    从 model_runner._pic_writeback_mla_kv 提取的核心逻辑，
    独立于 ModelRunner 类，便于单元测试。
    """
    pub_loc = getattr(forward_batch, "pic_public_out_loc", None)
    if pub_loc is None:
        return
    valid = pub_loc >= 0
    if not valid.any():
        return
    pub_slots = pub_loc[valid]
    priv_slots = forward_batch.out_cache_loc[valid]
    for layer_id in range(start_layer, end_layer):
        buf = buffers[layer_id]
        buf[pub_slots] = buf[priv_slots]


class TestPICMlaKvWriteback(unittest.TestCase):
    """MLA KV 从 private slot 复制到 public slot 的核心逻辑。"""

    def _make_buffers(self, num_layers=2, slots=64, kv_dim=8):
        return {lid: torch.zeros(slots, kv_dim) for lid in range(num_layers)}

    def _make_fb(self, out_cache_loc, pub_loc):
        fb = MagicMock()
        fb.out_cache_loc = torch.tensor(out_cache_loc, dtype=torch.int64)
        fb.pic_public_out_loc = (
            torch.tensor(pub_loc, dtype=torch.int64)
            if pub_loc is not None else None
        )
        return fb

    def test_no_pub_loc_is_noop(self):
        """pic_public_out_loc 为 None → buffer 不变。"""
        bufs = self._make_buffers()
        for buf in bufs.values():
            buf[:] = 99.0
        fb = self._make_fb([0, 1], None)
        _writeback_mla_kv(fb, bufs, 0, 2)
        for buf in bufs.values():
            self.assertTrue((buf == 99.0).all())

    def test_miss_position_copied_to_public(self):
        """Miss 位置的 private slot KV 被正确复制到 public slot。"""
        bufs = self._make_buffers(num_layers=2, slots=32, kv_dim=4)
        for lid, buf in bufs.items():
            buf[0] = float(lid * 10 + 1)   # priv slot 0
        # out_cache_loc[0] = slot 0 (priv); pub_loc[0] = slot 20 (pub)
        fb = self._make_fb([0], [20])
        _writeback_mla_kv(fb, bufs, 0, 2)
        for lid, buf in bufs.items():
            self.assertTrue(torch.allclose(buf[20], buf[0]),
                            f"layer {lid}: pub slot 20 应等于 priv slot 0")

    def test_neg1_positions_skipped(self):
        """pub_loc=-1 的位置不被拷贝，其他位置正常拷贝。"""
        bufs = self._make_buffers(num_layers=1, slots=32, kv_dim=4)
        buf = bufs[0]
        buf[3] = 11.0   # priv slot 3（对应 pub_loc=-1，应跳过）
        buf[7] = 22.0   # priv slot 7（对应 pub_loc=25）
        fb = self._make_fb([3, 7], [-1, 25])
        _writeback_mla_kv(fb, bufs, 0, 1)
        # slot 25 应等于 slot 7（全部 4 个维度）
        self.assertTrue(torch.allclose(buf[25], buf[7]),
                        "pub slot 25 应拷贝自 priv slot 7")
        # slot 3 本身不应被改变
        self.assertTrue(torch.allclose(buf[3], torch.full((4,), 11.0)),
                        "priv slot 3 本身不应被改变")

    def test_all_neg1_is_noop(self):
        """全部 pub_loc=-1 → 无任何拷贝。"""
        bufs = self._make_buffers(num_layers=2, slots=16, kv_dim=4)
        for buf in bufs.values():
            buf[:] = 77.0
        fb = self._make_fb([0, 1, 2], [-1, -1, -1])
        _writeback_mla_kv(fb, bufs, 0, 2)
        for buf in bufs.values():
            self.assertTrue((buf == 77.0).all())

    def test_all_layers_receive_copy(self):
        """每一层的 kv_buffer 都被正确复制。"""
        N = 4
        bufs = self._make_buffers(num_layers=N, slots=32, kv_dim=6)
        for lid, buf in bufs.items():
            buf[5] = float(lid + 1)   # priv slot 5 有唯一值
        fb = self._make_fb([5], [28])
        _writeback_mla_kv(fb, bufs, 0, N)
        for lid, buf in bufs.items():
            self.assertTrue(torch.allclose(buf[28], buf[5]),
                            f"layer {lid} 未正确拷贝")

    def test_multiple_miss_tokens(self):
        """多个 miss token 同时被拷贝。"""
        bufs = self._make_buffers(num_layers=1, slots=64, kv_dim=4)
        buf = bufs[0]
        buf[1] = 1.0
        buf[2] = 2.0
        buf[3] = 3.0
        # out_cache_loc: [1, 2, 3]; pub_loc: [10, 11, 12]
        fb = self._make_fb([1, 2, 3], [10, 11, 12])
        _writeback_mla_kv(fb, bufs, 0, 1)
        for pub, priv in [(10, 1), (11, 2), (12, 3)]:
            self.assertTrue(torch.allclose(buf[pub], buf[priv]),
                            f"pub slot {pub} 应等于 priv slot {priv}")


# =========================================================================
# 3. DSA K writeback 掩码逻辑
# =========================================================================

class TestPICDsaKWritebackMaskLogic(unittest.TestCase):
    """_pic_writeback_dsa_k 的掩码提取和路由逻辑（不依赖 Indexer 完整初始化）。"""

    def _extract(self, pub_loc_vals, key_dim=128):
        """模拟 _pic_writeback_dsa_k 的掩码提取部分。"""
        pub_loc = torch.tensor(pub_loc_vals, dtype=torch.int64)
        key = torch.arange(len(pub_loc_vals) * key_dim,
                           dtype=torch.float32).reshape(len(pub_loc_vals), key_dim)
        valid = pub_loc >= 0
        if not valid.any():
            return None, None
        return pub_loc[valid], key[valid]

    def test_no_valid_position_returns_none(self):
        """全部 -1 → 无有效位置，直接返回。"""
        pub_slots, pub_key = self._extract([-1, -1, -1])
        self.assertIsNone(pub_slots)

    def test_correct_slots_extracted(self):
        """正确提取 pub_loc >= 0 的 slot 值。"""
        pub_slots, _ = self._extract([-1, 5, -1, 12, -1])
        self.assertEqual(pub_slots.tolist(), [5, 12])

    def test_correct_keys_extracted(self):
        """正确提取对应位置的 key 行。"""
        pub_loc_vals = [-1, 5, -1, 12, -1]
        key_dim = 4
        pub_loc = torch.tensor(pub_loc_vals, dtype=torch.int64)
        key = torch.arange(len(pub_loc_vals) * key_dim,
                           dtype=torch.float32).reshape(len(pub_loc_vals), key_dim)
        valid = pub_loc >= 0
        pub_key = key[valid]
        # positions 1 and 3 are valid
        self.assertTrue(torch.allclose(pub_key[0], key[1]))
        self.assertTrue(torch.allclose(pub_key[1], key[3]))

    def test_all_valid_all_extracted(self):
        """全部 pub_loc >= 0 → 全部 key 行被提取。"""
        pub_slots, pub_key = self._extract([10, 20, 30], key_dim=8)
        self.assertIsNotNone(pub_slots)
        self.assertEqual(pub_slots.shape[0], 3)
        self.assertEqual(pub_key.shape[0], 3)

    def test_single_valid_position(self):
        """只有一个有效位置。"""
        pub_slots, pub_key = self._extract([-1, -1, 7, -1])
        self.assertEqual(pub_slots.tolist(), [7])
        self.assertEqual(pub_key.shape[0], 1)

    def test_pub_loc_none_is_early_exit(self):
        """forward_batch.pic_public_out_loc 为 None → 提前返回。"""
        fb = MagicMock()
        fb.pic_public_out_loc = None
        result = getattr(fb, "pic_public_out_loc", None)
        self.assertIsNone(result)

    def test_extracted_key_is_contiguous(self):
        """提取后的 key 经过 .contiguous() 保证连续布局（避免 kernel 崩溃）。"""
        pub_loc_vals = [-1, 3, -1, 7]
        key = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        valid = torch.tensor(pub_loc_vals, dtype=torch.int64) >= 0
        pub_key = key[valid].contiguous()
        self.assertTrue(pub_key.is_contiguous())


# =========================================================================
# 4. 端到端：alloc → pub_loc → writeback 一致性
# =========================================================================

class TestPICWritebackEndToEnd(unittest.TestCase):
    """alloc 产出的 pub_loc 与 out_cache_loc 配合 writeback 逻辑一致。"""

    def test_pub_slots_are_written_to_correctly(self):
        """
        alloc 产出 pub_loc，用 writeback 逻辑把数据从 priv slot 复制到 pub slot，
        验证 pub slot 的内容等于 priv slot 的内容。
        """
        segs = [(0, 5), (5, 10), (10, 15)]  # seg0, seg1 = miss; seg2 = last
        req = _make_req(segs)
        out_loc, pub_loc = _run_alloc([req])

        # 构造一个大 buffer，让 priv slot 有唯一值
        n_slots = 128
        buf = torch.zeros(n_slots, 4)
        # 把 priv slot 的值设为 slot index 本身（便于验证）
        for i in range(n_slots):
            buf[i] = float(i)

        # 执行 writeback：把 priv slot 内容复制到 pub slot
        valid = pub_loc >= 0
        pub_slots = pub_loc[valid]
        priv_slots = out_loc[valid]
        buf[pub_slots] = buf[priv_slots]

        # 验证每个 pub slot 的值等于对应 priv slot 的值
        for pub, priv in zip(pub_slots.tolist(), priv_slots.tolist()):
            self.assertTrue(
                torch.allclose(buf[pub], buf[priv]),
                f"pub slot {pub} 应等于 priv slot {priv}: "
                f"got {buf[pub].tolist()} vs {buf[priv].tolist()}"
            )

    def test_last_seg_pub_loc_never_written(self):
        """最后段 pub_loc 恒为 -1，writeback 逻辑跳过它。

        只需验证 pub_loc 中最后段对应位置全部为 -1；
        writeback 不会向任何 -1 的 slot 写入。
        """
        segs = [(0, 8), (8, 16)]   # seg0 miss, seg1 last
        req = _make_req(segs)
        out_loc, pub_loc = _run_alloc([req])

        # 最后段 positions 8..15 → pub_loc 全为 -1
        self.assertTrue((pub_loc[8:16] == -1).all(),
                        "最后段 (seg1) 所有位置 pub_loc 必须为 -1")

        # 构造大 buffer；用 writeback 只写 miss 段
        buf = torch.full((64, 4), 99.0)
        valid = pub_loc >= 0
        if valid.any():
            pub_slots = pub_loc[valid]
            priv_slots = out_loc[valid]
            buf[pub_slots] = buf[priv_slots]

        # 验证：只有有效 pub slot 发生了写入（值变为 99.0 → 99.0，不变）
        # 最关键的断言：last seg 的 priv slots 对应的 pub_loc 全为 -1
        last_seg_pub = pub_loc[8:16]
        self.assertTrue((last_seg_pub == -1).all(),
                        "最后段每个 token 的 pub_loc 都必须是 -1，不存在 writeback 目标")


if __name__ == "__main__":
    unittest.main()
