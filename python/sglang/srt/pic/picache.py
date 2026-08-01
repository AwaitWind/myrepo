"""
PICache — Position-Independent Cache (transition_rope mode, GLM5.2 / DSA).

In transition_rope mode each segment is cached with position-free *public*
K/V slots.  On a cache hit the model loads the public K and re-applies RoPE
for the current position (private slots), then uses the private slots for
attention.  Only miss segments require full Q/K/V computation from scratch.

Design note — empty device_indices
------------------------------------
match_prefix always returns empty device_indices so that prefix_len = 0 and
extend_input_len covers the full sequence.  This guarantees out_cache_loc
has seq_len elements, satisfying the DSA indexer kernel requirement:
  fused_store_index_k_cache expects num_tokens == seq_len, not miss_len.
PIC segment hits are tracked via pic_segment_entries in MatchResult only.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.pic.eviction import EvictionStrategy, LRUStrategy
from sglang.srt.pic.hasher import segment_hash

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.pic.pic_alloc import DSAStatePool

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SegmentEntry:
    """PIC segment cache entry (transition_rope mode).

    full_kv_slots: public (position-free) K/V slots stored in the KV pool.
    dsa_state_slot: DSA accumulator state slot index (0 = not used).
    start_pos: sequence position of the first token when this segment was cached.
               Used to compute the delta_pos for RoPE re-application on future hits.
    """

    seg_hash: bytes
    full_kv_slots: torch.Tensor
    token_ids: torch.Tensor
    start_pos: int = 0
    dsa_state_slot: int = 0
    lock_ref: int = 0
    last_access_time: float = 0.0
    creation_time: float = 0.0
    hit_count: int = 0
    priority: int = 0


class PICache(BasePrefixCache):
    """PIC KV cache — transition_rope mode, GLM5.2 / DSA only."""

    def __init__(
        self,
        req_to_token_pool: "ReqToTokenPool",
        token_to_kv_pool_allocator: "BaseTokenToKVPoolAllocator",
        dsa_state_pool: Optional["DSAStatePool"] = None,
        page_size: int = 1,
        disable: bool = False,
        enable_metrics: bool = False,
    ):
        super().__init__()
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.dsa_state_pool = dsa_state_pool
        self.page_size = page_size
        self.disable = disable

        self._entries: Dict[bytes, SegmentEntry] = {}
        self.eviction_strategy: EvictionStrategy = LRUStrategy()

        self._inflight_full_tokens: int = 0
        self._inflight_dsa_slots: int = 0

        if enable_metrics:
            self.init_metrics_collector()

        self._device = getattr(token_to_kv_pool_allocator, "device", torch.device("cpu"))

    @property
    def has_dsa(self) -> bool:
        return self.dsa_state_pool is not None

    # ------------------------------------------------------------------
    # BasePrefixCache interface
    # ------------------------------------------------------------------

    def supports_mamba(self) -> bool:
        return False

    def supports_swa(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def reset(self):
        for entry in list(self._entries.values()):
            self._evict_entry(entry)
        self._entries.clear()
        self._inflight_full_tokens = 0
        self._inflight_dsa_slots = 0

    # ------------------------------------------------------------------
    # match_prefix
    # ------------------------------------------------------------------

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        if self.disable:
            return self._empty_match_result()

        req = params.req
        if req is None or not req.pic_segments:
            # 诊断：match_prefix 提前返回（PIC 分段未生效）
            logger.warning(
                "[PIC-DBG] match_prefix EARLY-RETURN: req_is_none=%s pic_segments=%s entries_in_cache=%d",
                req is None,
                None if req is None else req.pic_segments,
                len(self._entries),
            )
            return self._empty_match_result()

        # 修复：不能用 req.get_fill_ids()。match_prefix 在 init_next_round_input 早期
        # 调用，此时 req.fill_len 尚为 0，get_fill_ids() = full_untruncated_fill_ids[:0]
        # 返回空，导致每段切片为空、hash 变成空哈希(e3b0c442)而永远 miss。
        # pic_segments 的坐标基于完整 prompt token，直接用 origin_input_ids，
        # 与 cache_unfinished_req 在 prefill 阶段取到的 fill_ids 坐标一致。
        fill_ids = req.origin_input_ids
        pic_segment_entries: List[Optional[SegmentEntry]] = []
        _dbg_segs = []  # 每段诊断: (idx, seg_len, hash前缀, 是否命中)

        for i, (start, end) in enumerate(req.pic_segments):
            if i == len(req.pic_segments) - 1:
                pic_segment_entries.append(None)  # last segment never cached
                _dbg_segs.append((i, end - start, "----", "last"))
                continue

            seg_ids = torch.tensor(fill_ids[start:end], dtype=torch.int64)
            seg_hash_val = segment_hash(seg_ids)
            entry = self._match_segment(seg_hash_val, seg_ids)

            _dbg_segs.append(
                (i, end - start, seg_hash_val[:4].hex(), "HIT" if entry is not None else "miss")
            )

            if entry is None:
                pic_segment_entries.append(None)
                continue

            entry.last_access_time = time.monotonic()
            entry.hit_count += 1
            pic_segment_entries.append(entry)

        # PIC 命中不通过 cached_tokens 上报（match_prefix 故意返回空 device_indices，
        # 见文件顶部设计注释）。这里显式打印命中统计，便于测试脚本/运维判断段级缓存
        # 是否真正命中。用 warning 级别以便在默认 --log-level warning 下也可见。
        num_hit = sum(1 for e in pic_segment_entries if e is not None)
        # 无条件诊断日志：打印段数/命中数/缓存里现有段数/各段命中详情。
        logger.warning(
            "[PIC-DBG] match_prefix rid=%s segments=%d num_hit=%d "
            "entries_in_cache=%d fill_ids_len=%d segs=%s",
            getattr(req, "rid", "?"),
            len(req.pic_segments),
            num_hit,
            len(self._entries),
            len(fill_ids),
            _dbg_segs,
        )
        if num_hit > 0:
            hit_tokens = sum(
                int(e.token_ids.numel()) for e in pic_segment_entries if e is not None
            )
            logger.warning(
                "[PIC-HIT] rid=%s hit_segments=%d/%d hit_tokens=%d",
                getattr(req, "rid", "?"),
                num_hit,
                len(req.pic_segments),
                hit_tokens,
            )

        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64, device=self._device),
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
            host_hit_length=0,
            mamba_branching_seqlen=None,
            cache_protected_len=None,
            pic_segment_entries=pic_segment_entries,
        )

    def _empty_match_result(self) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64, device=self._device),
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
            host_hit_length=0,
            mamba_branching_seqlen=None,
            cache_protected_len=None,
            pic_segment_entries=None,
        )

    def _match_segment(self, seg_hash: bytes, seg_ids: torch.Tensor) -> Optional[SegmentEntry]:
        entry = self._entries.get(seg_hash)
        if entry is None:
            return None
        if not torch.equal(entry.token_ids, seg_ids):
            return None
        return entry

    # ------------------------------------------------------------------
    # _insert_segment
    # ------------------------------------------------------------------

    def _insert_segment(
        self,
        seg_hash: bytes,
        seg_ids: torch.Tensor,
        kv_slots: torch.Tensor,
        dsa_slot: int = 0,
        start_pos: int = 0,
    ) -> SegmentEntry:
        now = time.monotonic()
        entry = SegmentEntry(
            seg_hash=seg_hash,
            full_kv_slots=kv_slots.clone(),
            token_ids=seg_ids.detach().clone(),
            start_pos=start_pos,
            dsa_state_slot=dsa_slot,
            creation_time=now,
            last_access_time=now,
        )
        self._entries[seg_hash] = entry
        return entry

    # ------------------------------------------------------------------
    # cache_unfinished_req
    # ------------------------------------------------------------------

    def cache_unfinished_req(self, req: "Req"):
        """Store completed prefill segments into the cache."""
        if self.disable or not req.pic_segments:
            return
        if not req.pic_miss_segment_slots:
            return

        inflight_tokens = inflight_dsa = 0
        fill_ids = req.get_fill_ids()

        for (start, end) in req.pic_miss_segments:
            if (start, end) == req.pic_segments[-1]:
                continue  # last segment is never cached
            if (start, end) in req._pic_cached_segments:
                continue

            seg_ids = torch.tensor(fill_ids[start:end], dtype=torch.int64)
            seg_hash_val = segment_hash(seg_ids)
            slot_tuple = req.pic_miss_segment_slots[(start, end)]

            # transition_rope slot tuple: (priv_slots, pub_slots[, dsa_slot])
            if self.has_dsa:
                _priv, public, dsa = slot_tuple
            else:
                _priv, public = slot_tuple
                dsa = 0
            kv_slots = public

            existing = self._match_segment(seg_hash_val, seg_ids)
            # 诊断：打印缓存写入的每段 hash 前缀，用于与 match_prefix 的查找 hash 对比。
            logger.warning(
                "[PIC-DBG] cache_write seg=[%d,%d) len=%d hash=%s -> %s",
                start, end, end - start, seg_hash_val[:4].hex(),
                "DEDUP(existing)" if existing is not None else "INSERT(new)",
            )
            if existing is not None:
                # Concurrent duplicate — free the redundant public slots.
                self.token_to_kv_pool_allocator.free(kv_slots)
                if dsa > 0 and self.has_dsa:
                    self.dsa_state_pool.free(  # type: ignore[union-attr]
                        torch.tensor([dsa], dtype=torch.int64, device=kv_slots.device)
                    )
                req.pic_segment_entries[seg_hash_val] = existing
            else:
                entry = self._insert_segment(
                    seg_hash_val, seg_ids, kv_slots, dsa,
                    start_pos=start,  # record canonical position for future delta-RoPE
                )
                entry.lock_ref += 1  # protect until request finishes
                req.pic_segment_entries[seg_hash_val] = entry

            inflight_tokens += kv_slots.numel()
            if dsa > 0:
                inflight_dsa += 1
            req._pic_cached_segments.add((start, end))

        if inflight_tokens or inflight_dsa:
            self.remove_inflight(inflight_tokens, inflight_dsa)

    # ------------------------------------------------------------------
    # cache_finished_req
    # ------------------------------------------------------------------

    def cache_finished_req(self, req: "Req", is_insert: bool = True):
        """Release all PIC-managed resources for a finished request."""
        if self.disable:
            return

        if is_insert:
            self.cache_unfinished_req(req)

        if not req.pic_miss_segment_slots:
            # Non-PIC request routed through PICache — free all KV directly.
            kv_len = getattr(req, "kv_committed_len", req.seqlen)
            if kv_len > 0:
                kv_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :kv_len  # type: ignore[index]
                ]
                self.token_to_kv_pool_allocator.free(kv_indices)
            self.dec_lock_ref(req)
            return

        # 1. Free last segment's private slots (last segment is never cached).
        last_seg = req.pic_segments[-1]  # type: ignore[index]
        last_tuple = req.pic_miss_segment_slots.get(last_seg)
        if last_tuple is not None:
            if self.has_dsa:
                last_priv, _pub, last_dsa = last_tuple
            else:
                last_priv, _pub = last_tuple
                last_dsa = 0
            self.token_to_kv_pool_allocator.free(last_priv)
            if last_dsa > 0 and self.has_dsa:
                self.dsa_state_pool.free(  # type: ignore[union-attr]
                    torch.tensor([last_dsa], dtype=torch.int64, device=self._device)
                )
            self.remove_inflight(0, 0)

        # 2. Free private slots for all non-last miss segments.
        for (start, end), slot_tuple in req.pic_miss_segment_slots.items():
            if (start, end) == req.pic_segments[-1]:  # type: ignore[index]
                continue
            self.token_to_kv_pool_allocator.free(slot_tuple[0])

        # 3. Free private slots for hit segments (used for RoPE re-encoding).
        for _seg, hit_info in req.pic_rope_hit_private_slots.items():
            priv_slots = hit_info[0]  # (priv_slots, pub_slots, old_start)
            self.token_to_kv_pool_allocator.free(priv_slots)

        # 4. Free decode KV slots (tokens generated after the prompt).
        prompt_end = len(req.origin_input_ids)
        kv_committed_len = getattr(req, "kv_committed_len", req.seqlen)
        if kv_committed_len > prompt_end:
            decode_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, prompt_end:kv_committed_len  # type: ignore[index]
            ]
            self.token_to_kv_pool_allocator.free(decode_slots)

        # 5. Free pic_a3 new-path scratch that has NO other owner.
        #    l01_scratch is already freed above — every segment's l01 slice is
        #    stored in pic_rope_hit_private_slots (hit) / pic_miss_segment_slots
        #    (miss) and freed in steps 1-3 — and each miss segment's "public"
        #    slots become the cached SegmentEntry. But two per-request slices had
        #    no free path and leaked ~(miss_len + max_imp_len) slots per pic_a3
        #    request, exhausting the KV pool after a few dozen requests:
        #      - pic_a3_l2plus_imp_slots_pool : recomputed imp-token KV at
        #        layer 2+, deliberately NOT written back to public ("discarded").
        #      - pic_a3_l2plus_miss_slots_per_seg : layer-2+ private miss KV.
        #    Both are read via req_to_token_pool through decode, so they are only
        #    safe to release here, at request completion. Idempotent: the attrs
        #    are cleared so a second cache_finished_req call is a no-op.
        _imp_pool = getattr(req, "pic_a3_l2plus_imp_slots_pool", None)
        if _imp_pool is not None and _imp_pool.numel() > 0:
            self.token_to_kv_pool_allocator.free(_imp_pool)
        req.pic_a3_l2plus_imp_slots_pool = None
        _l2plus_miss_per_seg = getattr(req, "pic_a3_l2plus_miss_slots_per_seg", None)
        if _l2plus_miss_per_seg:
            for _miss_slice in _l2plus_miss_per_seg.values():
                if _miss_slice is not None and _miss_slice.numel() > 0:
                    self.token_to_kv_pool_allocator.free(_miss_slice)
            req.pic_a3_l2plus_miss_slots_per_seg = {}

        self.dec_lock_ref(req)

    # ------------------------------------------------------------------
    # evict
    # ------------------------------------------------------------------

    def evict(self, params: EvictParams) -> EvictResult:
        need_tokens = params.num_tokens
        need_dsa = getattr(params, "dsa_num", 0)

        if need_tokens <= 0 and need_dsa <= 0:
            return EvictResult()

        candidates = [e for e in self._entries.values() if e.lock_ref == 0]
        candidates.sort(key=self.eviction_strategy.get_priority)

        n_tok = n_dsa = 0
        for entry in candidates:
            if n_tok >= need_tokens and n_dsa >= need_dsa:
                break
            self._evict_entry(entry)
            del self._entries[entry.seg_hash]
            n_tok += entry.full_kv_slots.numel()
            if self.has_dsa and entry.dsa_state_slot > 0:
                n_dsa += 1

        return EvictResult(num_tokens_evicted=n_tok, dsa_num_evicted=n_dsa)

    def _evict_entry(self, entry: SegmentEntry):
        self.token_to_kv_pool_allocator.free(entry.full_kv_slots)
        if self.has_dsa and entry.dsa_state_slot > 0:
            self.dsa_state_pool.free(  # type: ignore[union-attr]
                torch.tensor([entry.dsa_state_slot], dtype=torch.int64, device=self._device)
            )

    # ------------------------------------------------------------------
    # lock_ref protocol
    # ------------------------------------------------------------------

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        if not hasattr(node, "pic_segment_entries"):
            return IncLockRefResult(delta=0)
        delta = 0
        for entry in node.pic_segment_entries.values():
            entry.lock_ref += 1
            delta += 1
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if not hasattr(node, "pic_segment_entries"):
            return DecLockRefResult(delta=0)
        delta = 0
        for entry in node.pic_segment_entries.values():
            if entry.lock_ref > 0:
                entry.lock_ref -= 1
                delta += 1
        return DecLockRefResult(delta=delta)

    # ------------------------------------------------------------------
    # inflight tracking
    # ------------------------------------------------------------------

    def add_inflight(self, num_tokens: int, dsa_num: int = 0):
        self._inflight_full_tokens += num_tokens
        self._inflight_dsa_slots += dsa_num

    def remove_inflight(self, num_tokens: int, dsa_num: int = 0):
        self._inflight_full_tokens = max(0, self._inflight_full_tokens - num_tokens)
        self._inflight_dsa_slots = max(0, self._inflight_dsa_slots - dsa_num)

    # ------------------------------------------------------------------
    # size reporting
    # ------------------------------------------------------------------

    def evictable_size(self) -> int:  # type: ignore[override]
        return sum(e.full_kv_slots.numel() for e in self._entries.values() if e.lock_ref == 0)

    def protected_size(self) -> int:  # type: ignore[override]
        return sum(e.full_kv_slots.numel() for e in self._entries.values() if e.lock_ref > 0)

    def total_size(self) -> int:
        return sum(e.full_kv_slots.numel() for e in self._entries.values())

    def full_evictable_size(self) -> int:  # type: ignore[override]
        return self.evictable_size()

    def full_protected_size(self) -> int:  # type: ignore[override]
        return self.protected_size() + self._inflight_full_tokens

    def dsa_evictable_size(self) -> int:
        return sum(1 for e in self._entries.values() if e.lock_ref == 0 and e.dsa_state_slot > 0)

    def dsa_protected_size(self) -> int:
        return (
            sum(1 for e in self._entries.values() if e.lock_ref > 0 and e.dsa_state_slot > 0)
            + self._inflight_dsa_slots
        )

    def all_values_flatten(self) -> torch.Tensor:
        vals = [e.full_kv_slots for e in self._entries.values()]
        if not vals:
            return torch.empty((0,), dtype=torch.int64, device=self._device)
        return torch.cat(vals)

    def session_held_tokens(self, active_pool_idxs=None) -> int:
        return 0

    def session_held_full_tokens(self, active_pool_idxs=None) -> int:
        return 0

    def session_held_swa_tokens(self, active_pool_idxs=None) -> int:
        return 0

    def session_held_req_count(self, active_pool_idxs=None) -> int:
        return 0

    def session_held_mamba_slots(self, active_pool_idxs=None) -> int:
        return 0

    def pretty_print(self):
        logger.info(
            "PICache: %d entries, %d tokens, inflight(t=%d, d=%d)",
            len(self._entries), self.total_size(),
            self._inflight_full_tokens, self._inflight_dsa_slots,
        )

    def sanity_check(self):
        pass
