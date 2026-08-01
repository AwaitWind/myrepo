"""Segment-warmup primitives — sync HTTP against sglang's /generate.

Ported from pic_bench.warmup (async / LMCache-flavoured) and simplified for
Phase 1 of the pic_bench_lite integration: one blocking HTTP request per
chunk, matching how quick_test_online.py already calls sglang. The wire
format follows pic_bench:

    system_prompt + separator + <chunk> + separator + warmup_query

so PIC's segment cache stores the same segment token-id sequence as a
subsequent measure request built with `prompt.build_prompt(chunks, query,
separator)`. Without the leading `sys + sep` wrap the boundary tokens
differ (BOS / sentence-start effects) and cache lookups miss.

PIC + DSA quirk — page alignment: DSA forces page_size=64, so every PIC
segment slot must be page-aligned. In dataset mode we pad `sys` / `chunk`
/ `query` to multiples of 64 tokens. **The warmup query has to be padded
too**, else PIC's `pic_alloc_for_extend` will see a non-64-aligned trailing
segment and crash with:

    RuntimeError: The expanded size of the tensor (N) must match the
    existing size (64) at non-singleton dimension 0.

Callers can override the default `"warmup"` via `warmup_query=`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from sglang.test.pic_bench_lite.prompt import _DEFAULT_SYSTEM_PROMPT

_WARMUP_QUERY = "warmup"


@dataclass
class WarmupReport:
    wall_time_sec: float
    n_unique_chunks_primed: int
    n_requests: int
    n_failed: int


# A `send_fn` is any callable that takes (prompt: str, max_new_tokens: int) and
# returns True on HTTP 2xx / False on any error. Deliberately narrow so callers
# don't need to plumb sglang-specific request/response types into this module.
SendFn = Callable[[str, int], bool]


def prime_one(
    send_fn: SendFn,
    chunk_text: str,
    separator: str,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    warmup_query: Optional[str] = None,
) -> bool:
    """Send a single-chunk warmup prompt. Returns True on success.

    The chunk MUST be wrapped in `separator` on both sides for the segment
    hash to match what a later `build_prompt(...)` will produce; see the
    module docstring for the reasoning.

    `warmup_query` overrides the default `"warmup"` — needed when the caller
    is doing PIC + DSA page-alignment padding and the trailing segment must
    also land on a 64-token boundary.
    """
    q = warmup_query if warmup_query is not None else _WARMUP_QUERY
    prompt = system_prompt + separator + chunk_text + separator + q
    try:
        return bool(send_fn(prompt, 1))
    except Exception:
        return False


def prime_sample(
    send_fn: SendFn,
    chunks,
    separator: str,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    warmup_query: Optional[str] = None,
) -> WarmupReport:
    """Prime every chunk of a single sample. Analogous to pic_bench's
    per-sample accuracy warmup — each chunk is primed via
    `sys<sep><chunk><sep>warmup` so the following measure request hits
    the segment cache for every chunk.
    """
    start = time.time()
    n_failed = 0
    for c in chunks:
        if not prime_one(
            send_fn, c, separator, system_prompt, warmup_query=warmup_query,
        ):
            n_failed += 1
    return WarmupReport(
        wall_time_sec=time.time() - start,
        n_unique_chunks_primed=len(chunks) - n_failed,
        n_requests=len(chunks),
        n_failed=n_failed,
    )
