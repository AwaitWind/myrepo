"""
PIC (Position-Independent Cache) — segment-level KV cache (transition_rope mode only).

Module structure:
  hasher.py    — segment_hash: SHA-256[:16], int32 little-endian serialization
  segmenter.py — split_and_tokenize: split text by <<PIC_SEP>> and tokenize
  eviction.py  — EvictionStrategy and LRU/LFU/FIFO/MRU/FILO/Priority/SLRU implementations
  picache.py   — SegmentEntry + PICache
  pic_alloc.py — DSAStatePool + pic_alloc_for_extend (_pic_alloc_transition_rope)
"""

from sglang.srt.pic.eviction import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)
from sglang.srt.pic.hasher import segment_hash
from sglang.srt.pic.pic_alloc import DSAStatePool, pic_alloc_for_extend
from sglang.srt.pic.picache import PICache, SegmentEntry
from sglang.srt.pic.segmenter import split_and_tokenize

__all__ = [
    # hash
    "segment_hash",
    # segmentation
    "split_and_tokenize",
    # data structures
    "SegmentEntry",
    "PICache",
    "DSAStatePool",
    # eviction strategies
    "EvictionStrategy",
    "LRUStrategy",
    "LFUStrategy",
    "FIFOStrategy",
    "MRUStrategy",
    "FILOStrategy",
    "PriorityStrategy",
    "SLRUStrategy",
    # allocation
    "pic_alloc_for_extend",
]
