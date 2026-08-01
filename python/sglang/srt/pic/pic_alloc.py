"""
PIC allocation module — transition_rope mode, GLM5.2 / DSA only.

out_cache_loc covers ONLY miss-segment private slots, so its length equals the
total number of miss tokens across all PIC requests in the batch.  Hit-segment
private slots are pre-populated by _pic_prepopulate_hit_slots BEFORE the model
forward and are never passed through out_cache_loc.

pic_public_out_loc has the same length as out_cache_loc (miss tokens only).
For each miss-segment token it carries the public KV slot index to write back
to, or -1 if no writeback is needed (e.g. last segment which is never cached).

slot tuple format stored in req.pic_miss_segment_slots:
  without DSA:  (priv_slots, pub_slots_or_None)
  with DSA:     (priv_slots, pub_slots_or_None, dsa_slot)
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


# =========================================================================
# PIC allocation for pic_a3 / pic_cacheblend
#
# Under IMP_ONLY (the sole path): schedule-time static pick populates
# req.pic_hit_imp_flat with the last ceil(seg_len * recomp_ratio) tokens per
# hit segment (rounded up to 64). Only miss + imp positions enter the forward.
# Non-imp hit positions are served from _pic_prepopulate_hit_slots' pool fill,
# which mirrors the vanilla `pic` mode. See ragkv (yczhounlp/ragkv) for the
# source A³ paper method.
# =========================================================================


# =========================================================================
# DSAStatePool — GLM5.2 DSA hidden-state pool
# =========================================================================

class DSAStatePool:
    """Pool for GLM5.2 DSA (Discrete State Accumulator) per-segment states.

    Slot 0 is always dummy (zero-initialized, never allocated).
    Allocatable range: [1, pool_size].
    """

    def __init__(
        self,
        num_dsa_layers: int,
        pool_size: int,
        dsa_state_shape: Tuple[int, ...],
        dsa_dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_dsa_layers = num_dsa_layers
        self.pool_size = pool_size
        self.dsa_state_shape = dsa_state_shape
        self.dsa_dtype = dsa_dtype
        self.device = device

        self.dsa_state = torch.zeros(
            (num_dsa_layers, pool_size + 1, *dsa_state_shape),
            dtype=dsa_dtype,
            device=device,
        )
        self._free_slots: List[int] = list(range(1, pool_size + 1))

    def alloc(self, need_size: int = 1) -> Optional[torch.Tensor]:
        if need_size > len(self._free_slots):
            return None
        slots = self._free_slots[:need_size]
        self._free_slots = self._free_slots[need_size:]
        for slot in slots:
            self.dsa_state[:, slot, ...] = 0
        return torch.tensor(slots, dtype=torch.int64, device=self.device)

    def free(self, free_index: torch.Tensor) -> None:
        for idx in free_index.tolist():
            idx = int(idx)
            if idx >= 1:
                self._free_slots.append(idx)

    def get_state(self, slot: int) -> torch.Tensor:
        return self.dsa_state[:, slot, ...]

    def set_state(self, slot: int, state: torch.Tensor) -> None:
        self.dsa_state[:, slot, ...] = state

    def available_size(self) -> int:
        return len(self._free_slots)


# =========================================================================
# DSA slot allocator
# =========================================================================

def _alloc_one_dsa(dsa_state_pool: DSAStatePool, tree_cache) -> int:
    result = dsa_state_pool.alloc(1)
    if result is not None:
        return int(result[0])
    tree_cache.evict(EvictParams(num_tokens=0, dsa_num=1))
    result = dsa_state_pool.alloc(1)
    if result is None:
        raise RuntimeError("DSAStatePool exhausted after evict")
    return int(result[0])


# =========================================================================
# _pic_alloc_transition_rope
# =========================================================================

def _is_pic_a3_new_path_req(req) -> bool:
    """pic_a3/pic_cacheblend new path: layer 0-1 fresh recompute over
    full_len, layer 2+ only (miss+imp). See scripts/pic_a3_full_recompute_plan.md.

    Kept in sync with schedule_batch._build_input_ids_and_miss_positions's
    _is_pic_a3_new_path — they MUST agree, otherwise input_ids length
    (schedule_batch) and slot count (here) mismatch.
    """
    return bool(
        getattr(req, "pic_mode", None) in ("pic_a3", "pic_cacheblend")
        and getattr(req, "precomputed_kv_path", None) is None
        and getattr(req, "pic_hit_segments", None)
    )


def _pic_alloc_transition_rope(
    batch: "ScheduleBatch",
    tree_cache,
    token_to_kv_pool_allocator,
    dsa_state_pool: Optional[DSAStatePool] = None,
) -> torch.Tensor:
    """Allocate KV slots for transition_rope mode; return out_cache_loc.

    out_cache_loc contains private slots for ONLY miss segments (not hit
    segments).  Hit-segment private slots are written to req_to_token_pool
    directly and pre-populated by _pic_prepopulate_hit_slots before the model
    forward.

    req.pic_miss_segment_slots  — slot tuples for miss segments
    req.pic_rope_hit_private_slots — {(start,end): (priv_slots, pub_kv_slots, old_start)}

    Also sets batch.pic_public_out_loc: same length as out_cache_loc (miss
    tokens only), containing the public slot index for each miss-segment token
    (or -1 for last-segment tokens that are never cached).  The model runner
    uses this to write-back the computed K to the public KV pool slots so that
    future cache hits can re-use (and RoPE-transition) the stored K.

    === pic_a3 new-path (§3.3) ===
    When a request has _is_pic_a3_new_path_req(req)=True, we allocate THREE
    slot sets per request:
      - l01_scratch (full_len): layer 0-1 fresh KV write; freed at layer 2
        boundary by the deepseek_v2 hook (§3.6).
      - l2plus_miss (miss_len): layer 2+ miss token fresh KV; writes back to
        public via pic_public_out_loc at forward end (standard PIC path).
      - l2plus_imp (max_imp_len, over-allocated by ceil(hit_len*ratio)+miss_len):
        layer 2+ imp token fresh KV; NOT written back (discarded).

    out_cache_loc for pic_a3 new-path reqs is the l01_scratch flat (full_len
    per req), aligned with input_ids from schedule_batch. pic_public_out_loc
    is all -1 for those chunks (no writeback in layer 0-1). The
    deepseek_v2 hook swaps out_cache_loc + pic_public_out_loc to
    pic_a3_l2plus_out_cache_loc / pic_a3_l2plus_pub_out_loc at layer 2.
    """
    import math as _math

    device = token_to_kv_pool_allocator.device
    total_pub_inflight = 0
    total_dsa_alloc = 0

    all_priv_in_order: List[torch.Tensor] = []
    # Parallel list: public slot index for the same position, -1 if no writeback needed
    all_pub_in_order: List[torch.Tensor] = []

    # For pic_a3 new-path: collect the layer-0-1 scratch slot slices per req
    # so we can flatten them onto batch.pic_a3_l01_scratch_flat at the end
    # (used by the deepseek_v2 hook to free at layer 2 boundary, §3.6).
    _pic_a3_l01_scratch_pieces: List[torch.Tensor] = []
    _any_pic_a3_new_path = False

    _neg1 = lambda n: torch.full((n,), -1, dtype=torch.int64, device=device)

    for req in batch.reqs:
        if not hasattr(req, "pic_segments") or not req.pic_segments:
            # Non-PIC request — standard extend alloc.
            if req.extend_input_len > 0:
                kv_slots = token_to_kv_pool_allocator.alloc(req.extend_input_len)
                if kv_slots is None:
                    raise RuntimeError(
                        f"KV pool exhausted for non-PIC req "
                        f"(extend_input_len={req.extend_input_len})"
                    )
                all_priv_in_order.append(kv_slots)
                all_pub_in_order.append(_neg1(req.extend_input_len))
                # Non-PIC: write slots to req_to_token_pool will be handled by
                # standard write_cache_indices (these requests keep prefix_len=0
                # and extend covers all tokens).
            continue

        req.pic_miss_segment_slots = {}
        req.pic_rope_hit_private_slots = {}

        hit_lookup: dict = {(s, e): h for (s, e, h) in req.pic_hit_segments}
        miss_set: set = set(req.pic_miss_segments)

        # ── pic_a3 new-path branch (§3.3) ────────────────────────────────
        _is_a3 = _is_pic_a3_new_path_req(req)
        if _is_a3:
            _any_pic_a3_new_path = True
            full_len = sum(end - start for (start, end) in req.pic_segments)
            hit_len = sum(
                end - start for (start, end) in req.pic_segments
                if (start, end) in hit_lookup
            )
            miss_len_local = sum(
                end - start for (start, end) in req.pic_segments
                if (start, end) in miss_set
            )
            # Same "no last-seg pub" convention as legacy path.
            miss_non_last = sum(
                end - start for (start, end) in req.pic_segments
                if (start, end) in miss_set and (start, end) != req.pic_segments[-1]
            )
            recomp_ratio = float(getattr(req, "recomp_ratio", 0.15) or 0.15)
            # max_imp_len upper bound. Normal case: ⌈hit_len × recomp_ratio⌉ + miss_len.
            # For diagnostic (PIC_A3_FORCE_ALL_IMP=1) we may need up to full_len imp
            # slots. Also for the union-with-miss path where miss is added on top
            # of topk, we can safely bump — over-allocation only wastes slot pool.
            # Bump to full_len (worst case: every position is imp).
            _os_alloc = __import__("os")
            _force_all = _os_alloc.environ.get("PIC_A3_FORCE_ALL_IMP", "0") == "1"
            # Phase B keep-all-alive research probe: the l2plus_imp block is
            # repurposed as a per-token throwaway region for NON-imp fresh K
            # writes (one unique slot per position), so it must span full_len.
            _keep_alive = _os_alloc.environ.get("SGLANG_PIC_A3_KEEP_ALIVE", "0") == "1"
            # pic_a3_oracle now runs with keepalive on (Phase B windowed
            # re-selection), so _keep_alive already bumps max_imp_len to full_len
            # below — no separate oracle bump needed.
            _hit_only_env = _os_alloc.environ.get("SGLANG_PIC_A3_HIT_ONLY_IMP", "0")
            _hit_only = _hit_only_env not in ("0", "false", "False", "")
            if _force_all or _keep_alive:
                max_imp_len = full_len
            elif _hit_only:
                # HIT_ONLY picker: per hit-segment budget is
                # min(align_up_64(ceil(seg_len * ratio)), seg_len). The 64-align
                # can push each segment's budget WAY above the legacy
                # ceil(hit_len * ratio) global bound (e.g. 4 hit segs of
                # 4032 tokens at ratio=0.15 → 4*640=2560 vs legacy 1824).
                _PAGE = 64
                _per_seg_max = 0
                for (s, e) in hit_lookup.keys():
                    _seg_len = e - s
                    if _seg_len <= 0:
                        continue
                    _k = int(_math.ceil(_seg_len * recomp_ratio))
                    _k = ((_k + _PAGE - 1) // _PAGE) * _PAGE
                    _k = min(_k, _seg_len)
                    _per_seg_max += _k
                max_imp_len = _per_seg_max + miss_len_local
            else:
                max_imp_len = int(_math.ceil(hit_len * recomp_ratio)) + miss_len_local

            need = full_len + miss_len_local + miss_non_last + max_imp_len
            all_slots_a3 = token_to_kv_pool_allocator.alloc(need)
            if all_slots_a3 is None:
                raise RuntimeError(
                    f"KV pool exhausted for pic_a3 new-path req "
                    f"(need={need}: full_len={full_len} + miss={miss_len_local} + "
                    f"miss_pub={miss_non_last} + max_imp={max_imp_len})"
                )

            # Partition the alloc'd block:
            #   [0 : full_len)                     — l01_scratch  (layer 0-1 write)
            #   [full_len : full_len+miss_len)     — l2plus_miss  (layer 2+ miss)
            #   [.. : + miss_non_last)             — l2plus_miss_pub (writeback)
            #   [.. : + max_imp_len)               — l2plus_imp   (layer 2+ imp)
            _off = 0
            l01_scratch = all_slots_a3[_off : _off + full_len];       _off += full_len
            l2plus_miss = all_slots_a3[_off : _off + miss_len_local]; _off += miss_len_local
            l2plus_miss_pub = all_slots_a3[_off : _off + miss_non_last]; _off += miss_non_last
            l2plus_imp = all_slots_a3[_off : _off + max_imp_len];     _off += max_imp_len
            assert _off == need

            total_pub_inflight += miss_non_last

            # Stash for the deepseek_v2 hook (§3.6). All device tensors.
            req.pic_a3_l01_scratch_slots = l01_scratch.clone()
            req.pic_a3_l2plus_miss_slots = l2plus_miss.clone()
            req.pic_a3_l2plus_miss_pub_slots = l2plus_miss_pub.clone()
            req.pic_a3_l2plus_imp_slots_pool = l2plus_imp.clone()  # over-allocated pool
            req.pic_a3_max_imp_len = max_imp_len
            req.pic_a3_full_len = full_len
            req.pic_a3_miss_len = miss_len_local
            req.pic_a3_hit_len = hit_len

            # Walk segments in position order, slicing l01_scratch by (s, e).
            # This keeps req_to_token_pool[req_idx, s:e] filled with the l01
            # scratch slots for that segment (both hit and miss). Under
            # pic_a3 new-path the same slot layout is used until the layer 2
            # hook rewrites it (see §3.6).
            _miss_l2plus_offset = 0
            _miss_pub_offset = 0
            for (start, end) in req.pic_segments:
                seg_len = end - start
                is_last = (start, end) == req.pic_segments[-1]
                l01_slice = l01_scratch[start:end]  # position-indexed slice

                if (start, end) in hit_lookup:
                    seg_hash = hit_lookup[(start, end)]
                    entry = req.pic_segment_entries.get(seg_hash)
                    if entry is not None:
                        # Store l01_slice as the "private" slot so req_to_token_pool
                        # gets scratch layout; pub_kv_slots is the real public slot
                        # (used later for hit-non-imp position rewrite in layer 2).
                        req.pic_rope_hit_private_slots[(start, end)] = (
                            l01_slice.clone(),
                            entry.full_kv_slots.to(device),
                            entry.start_pos,
                        )
                    continue

                if (start, end) not in miss_set:
                    raise RuntimeError(
                        f"pic_a3 segment [{start},{end}) is neither hit nor miss"
                    )

                # Miss segment: layer 0-1 writes to l01_slice; layer 2+ will use
                # l2plus_miss (rewritten by the hook).
                l2plus_miss_slice = l2plus_miss[
                    _miss_l2plus_offset : _miss_l2plus_offset + seg_len
                ]
                _miss_l2plus_offset += seg_len
                if not is_last:
                    pub_slots = l2plus_miss_pub[
                        _miss_pub_offset : _miss_pub_offset + seg_len
                    ]
                    _miss_pub_offset += seg_len
                    pub_clone_a3: Optional[torch.Tensor] = pub_slots.clone()
                else:
                    pub_clone_a3 = None

                dsa_slot_a3 = 0
                if dsa_state_pool is not None and not is_last:
                    dsa_slot_a3 = _alloc_one_dsa(dsa_state_pool, tree_cache)
                    total_dsa_alloc += 1

                # For legacy compatibility: pic_miss_segment_slots keeps the
                # standard 2-tuple (priv, pub) or 3-tuple with dsa — picache.py
                # unpacks exactly that. The extra l2plus_miss_slice is stashed
                # in a SEPARATE per-req dict so the hook can look it up without
                # breaking picache's unpacking.
                if dsa_state_pool is not None:
                    req.pic_miss_segment_slots[(start, end)] = (
                        l01_slice.clone(), pub_clone_a3, dsa_slot_a3,
                    )
                else:
                    req.pic_miss_segment_slots[(start, end)] = (
                        l01_slice.clone(), pub_clone_a3,
                    )
                if not hasattr(req, "pic_a3_l2plus_miss_slots_per_seg"):
                    req.pic_a3_l2plus_miss_slots_per_seg = {}
                req.pic_a3_l2plus_miss_slots_per_seg[(start, end)] = (
                    l2plus_miss_slice.clone()
                )

            # Layer 0-1 out_cache_loc = full-length l01_scratch in position order
            all_priv_in_order.append(l01_scratch)
            # No writeback during layer 0-1
            all_pub_in_order.append(_neg1(full_len))
            # Stash the l01_scratch flat piece for the free-at-layer-2 op
            _pic_a3_l01_scratch_pieces.append(l01_scratch.clone())
            continue

        # ── Legacy PIC branch (pic mode, or pic_a3 with .pt) ─────────────
        # Count slots needed for this request.
        total_priv = sum(end - start for (start, end) in req.pic_segments)
        # Hit segments don't need pub allocation (they already have pub in
        # PICache entries); only miss segments allocate new pub slots.
        total_pub_miss = sum(
            end - start
            for (start, end) in req.pic_segments
            if (start, end) in miss_set and (start, end) != req.pic_segments[-1]
        )

        all_slots = token_to_kv_pool_allocator.alloc(total_priv + total_pub_miss)
        if all_slots is None:
            raise RuntimeError(
                f"KV pool exhausted for PIC request (need={total_priv + total_pub_miss})"
            )

        total_pub_inflight += total_pub_miss

        priv_pool = all_slots[:total_priv]
        pub_pool = all_slots[total_priv:]

        priv_offset = 0
        pub_offset = 0

        for (start, end) in req.pic_segments:
            seg_len = end - start
            is_last = (start, end) == req.pic_segments[-1]

            priv_slots = priv_pool[priv_offset : priv_offset + seg_len]
            priv_offset += seg_len

            if (start, end) in hit_lookup:
                # Hit segment: private slots pre-populated by _pic_prepopulate_hit_slots
                # BEFORE the model forward. NOT added to out_cache_loc (which is
                # miss-only).
                seg_hash = hit_lookup[(start, end)]
                entry = req.pic_segment_entries.get(seg_hash)
                if entry is not None:
                    req.pic_rope_hit_private_slots[(start, end)] = (
                        priv_slots.clone(),
                        entry.full_kv_slots.to(device),
                        entry.start_pos,
                    )
                # Slot will be written to req_to_token_pool directly (not via out_cache_loc)
                continue

            if (start, end) not in miss_set:
                raise RuntimeError(f"PIC segment [{start},{end}) is neither hit nor miss")

            # Miss segment: add to out_cache_loc (model forward computes K/V for these)
            all_priv_in_order.append(priv_slots)

            if not is_last:
                pub_slots = pub_pool[pub_offset : pub_offset + seg_len]
                pub_offset += seg_len
                pub_clone: Optional[torch.Tensor] = pub_slots.clone()
                all_pub_in_order.append(pub_slots.to(device, non_blocking=True))
            else:
                pub_clone = None
                all_pub_in_order.append(_neg1(seg_len))
                # Last segment is never cached — mark with -1 so writeback filter
                # (pub >= 0) drops it. Keep priv aligned so len(priv) == len(pub).

            dsa_slot = 0
            if dsa_state_pool is not None and not is_last:
                dsa_slot = _alloc_one_dsa(dsa_state_pool, tree_cache)
                total_dsa_alloc += 1

            if dsa_state_pool is not None:
                req.pic_miss_segment_slots[(start, end)] = (
                    priv_slots.clone(), pub_clone, dsa_slot,
                )
            else:
                req.pic_miss_segment_slots[(start, end)] = (
                    priv_slots.clone(), pub_clone,
                )

    tree_cache.add_inflight(total_pub_inflight, total_dsa_alloc)

    out_cache_loc = (
        torch.cat(all_priv_in_order) if all_priv_in_order
        else torch.empty((0,), dtype=torch.int64, device=device)
    )
    # pic_public_out_loc: same length as out_cache_loc.
    # Legacy: miss tokens only, each entry = public KV slot for writeback
    #   (or -1 for last segment / non-PIC).
    # pic_a3 new-path: chunk length = full_len per req, all -1 (no writeback
    #   during layer 0-1; hook rewrites at layer 2 boundary).
    batch.pic_public_out_loc = (
        torch.cat(all_pub_in_order) if all_pub_in_order
        else None
    )

    # pic_a3 new-path: expose the flat concat of layer-0-1 scratch slots
    # (used by the deepseek_v2 hook to free at layer 2 boundary, §3.6).
    # None if no pic_a3 new-path req in this batch.
    if _any_pic_a3_new_path and _pic_a3_l01_scratch_pieces:
        batch.pic_a3_l01_scratch_flat = torch.cat(_pic_a3_l01_scratch_pieces)
    else:
        batch.pic_a3_l01_scratch_flat = None

    return out_cache_loc


# =========================================================================
# pic_alloc_for_extend — unified entry point
# =========================================================================

def pic_alloc_for_extend(
    batch: "ScheduleBatch",
    tree_cache,
    token_to_kv_pool_allocator,
    dsa_state_pool: Optional[DSAStatePool] = None,
):
    """PIC extend allocation — replaces alloc_for_extend for PIC batches.

    Returns (out_cache_loc, req_pool_indices_device, req_pool_indices_cpu).

    Key design change vs original:
    - out_cache_loc covers ONLY miss segment private slots (not hit segments).
    - Hit segment private slots are pre-populated by _pic_prepopulate_hit_slots
      in model_runner BEFORE the model forward.
    - req_to_token_pool is written HERE with all slot indices (hit + miss) at
      their correct non-contiguous sequence positions.
    """
    from sglang.srt.mem_cache.common import alloc_req_slots

    batch.maybe_evict_swa()

    req_pool_indices = alloc_req_slots(batch.req_to_token_pool, batch.reqs, tree_cache)
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    out_cache_loc = _pic_alloc_transition_rope(
        batch, tree_cache, token_to_kv_pool_allocator, dsa_state_pool,
    )

    # Write ALL slot indices to req_to_token_pool at their correct sequence positions.
    # PIC segments are non-contiguous (hit and miss interleaved), so we can't use
    # the standard write_cache_indices which assumes contiguous prefix + extend layout.
    pool = batch.req_to_token_pool.req_to_token
    for i, req in enumerate(batch.reqs):
        req_idx = req_pool_indices_cpu[i].item()
        if not req.pic_segments:
            # Non-PIC request: all tokens are "miss", slots are in out_cache_loc
            # sequentially. Write them to positions [0, extend_len).
            n = req.extend_input_len
            # Find offset in out_cache_loc for this non-PIC request
            # (already handled by the sequential all_priv_in_order accumulation)
            continue  # will be written via the miss offset logic below

        # PIC request: write hit and miss slots at their actual positions
        for (s, e), hit_info in req.pic_rope_hit_private_slots.items():
            priv_slots = hit_info[0]
            pool[req_idx, s:e] = priv_slots
        for (s, e), slot_tuple in req.pic_miss_segment_slots.items():
            priv_slots = slot_tuple[0]
            pool[req_idx, s:e] = priv_slots

    # For non-PIC requests in the batch, write their slots using out_cache_loc offset.
    miss_offset = 0
    for i, req in enumerate(batch.reqs):
        req_idx = req_pool_indices_cpu[i].item()
        if req.pic_segments:
            # Already written above
            continue
        n = req.extend_input_len
        if n > 0:
            pool[req_idx, 0:n] = out_cache_loc[miss_offset : miss_offset + n]
            miss_offset += n

    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu
