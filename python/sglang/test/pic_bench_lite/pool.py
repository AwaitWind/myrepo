"""Zipf shared-document distractor pool (accuracy mode).

Faithful port of ``pic_bench/runner.py::_fill_distractor_pool`` (+ its
``_estimate_tokens`` helper and pool constants) into pic_bench_lite, so the
sglang ``quick_test_online.py --dataset`` path exercises the SAME multi-doc /
shared-cache regime as the reference pic_bench and produces comparable numbers.

Mechanism: pool-eligible datasets (hotpotqa, 2wikimqa, ...) whose own prompt is
below ``POOL_MIN_TOKENS`` get padded with distractor documents drawn from a
shared, Zipf-popularity-weighted pool of other samples' chunks, up to
8192-16384 tokens. Popular pool documents recur across many requests, modelling
realistic RAG / multi-tenant cache reuse.

Usage (once, before the per-mode measure loop so every mode sees identical
pooled prompts):

    fill_distractor_pool(samples, tokenizer, sep=SEP, seed=0)

then in the measure request assemble ``all_chunks = sample.meta["_pool_chunks"]
+ sample.chunks`` (shuffled) — see quick_test_online.run_mode_dataset.

Adapted from pic_bench: operates on ``ProcessedSample`` (sample_id / chunks /
ground_truth / meta) instead of ``RequestSpec`` (request_id).
"""
from __future__ import annotations

import random as _random
from collections import defaultdict
from typing import List, Optional

# ── pool config — kept identical to pic_bench/runner.py:238-255 ──────────────
POOL_DATASETS = {
    "hotpotqa", "2wikimqa", "musique", "longbench_qasper",
    "multinews", "longbench_narrativeqa", "longbench_gov_report",
}
POOL_MIN_TOKENS = 8192
POOL_MAX_TOKENS = 16384
POOL_MAX_TOKENS_OVERRIDE = {
    "multinews": 27648,
    "longbench_gov_report": 28672,
}
POOL_SIZE = 200
POOL_ZIPF_S = 1.0
# Summarization datasets need a larger pool to reach the multi-doc regime.
POOL_MIN_TOKENS_OVERRIDE = {
    "multinews": 16384,
    "longbench_gov_report": 16384,
}


def estimate_tokens(text: str, tokenizer) -> int:
    """Token count of `text` (no special tokens); char/4 fallback if no tokenizer."""
    if tokenizer is None:
        return len(text) // 4
    return len(tokenizer.encode(text, add_special_tokens=False))


def fill_distractor_pool(
    samples: List,
    tokenizer,
    sep: str,
    seed: int,
    min_tokens_override: Optional[int] = None,
    max_tokens_override: Optional[int] = None,
) -> None:
    """Assign a Zipf distractor pool to each sample's ``meta["_pool_chunks"]``.

    Mirrors pic_bench ``_fill_distractor_pool``:
      * Per dataset (``sample.meta["dataset"]``), build a pool of ``POOL_SIZE``
        records (uniform sample from the group), each with a fixed rank 0..N-1.
      * For each sample below the min-token target, pick pool records via
        Zipf-weighted sampling without replacement (Efraimidis-Spirakis) until
        the prompt reaches the target (capped at the max), then sort the picks
        by ascending rank before prepending so the rank-0 record is a stable
        leading prefix that prefix-cache backends can exploit across requests.

    Records are pooled *within* the loaded samples, so a larger ``--n-samples``
    yields a richer pool (pic_bench pools 200 from the full dataset).

    Ground truth: ``_pool_gts`` is stashed for parity with pic_bench, but F1
    datasets (hotpotqa) score against the sample's own ``ground_truth`` only —
    the caller does not combine pool GTs (pic_bench combines only for rouge_l).
    """
    by_dataset: dict = defaultdict(list)
    for s in samples:
        ds = (s.meta or {}).get("dataset", "")
        if ds in POOL_DATASETS:
            by_dataset[ds].append(s)

    for ds, group in by_dataset.items():
        pool_min_tokens = (
            min_tokens_override if min_tokens_override is not None
            else POOL_MIN_TOKENS_OVERRIDE.get(ds, POOL_MIN_TOKENS)
        )
        pool_max_tokens = (
            max_tokens_override if max_tokens_override is not None
            else POOL_MAX_TOKENS_OVERRIDE.get(ds, POOL_MAX_TOKENS)
        )

        def _gt_str(s) -> str:
            gt = s.ground_truth
            if gt is None:
                return ""
            if isinstance(gt, list):
                return " ".join(gt)
            return str(gt)

        # Pool record: (request_id, chunks_list, gt_str, total_tokens).
        # request_id = f"{ds}-{sample_id}" to match pic_bench's RequestSpec.
        # request_id (qa_f1.py:58) so pool selection + shuffle RNG draws align.
        all_records: list = []
        for s in group:
            if not s.chunks:
                continue
            n_tok = estimate_tokens(sep.join(s.chunks), tokenizer)
            all_records.append((f"{ds}-{s.sample_id}", list(s.chunks), _gt_str(s), n_tok))

        pool_rng = _random.Random(f"{seed}:{ds}:pool_build")
        pool_size = min(POOL_SIZE, len(all_records))
        pool = pool_rng.sample(all_records, pool_size)
        # pool[rank] = (sample_id, chunks, gt_str, n_tok); rank 0 most Zipf-popular
        zipf_w = [1.0 / (i + 1) ** POOL_ZIPF_S for i in range(pool_size)]

        for s in group:
            _rid = f"{ds}-{s.sample_id}"  # match pic_bench spec.request_id
            current_tokens = estimate_tokens(
                sep.join(s.chunks) + sep + s.query, tokenizer
            )
            if current_tokens >= pool_min_tokens:
                s.meta["_pool_chunks"] = []
                s.meta["_pool_gts"] = []
                continue

            candidates = [
                (rank, chunks, gt, n_tok)
                for rank, (sid, chunks, gt, n_tok) in enumerate(pool)
                if sid != _rid
            ]
            if not candidates:
                s.meta["_pool_chunks"] = []
                s.meta["_pool_gts"] = []
                continue

            # Efraimidis-Spirakis weighted shuffle: key = u^(1/w), sort desc.
            rng = _random.Random(f"{seed}:{_rid}:pool")
            keys = [rng.random() ** (1.0 / zipf_w[rank]) for rank, *_ in candidates]
            ordered = sorted(zip(keys, candidates), key=lambda x: -x[0])

            selected: list = []
            pad_tokens = 0
            needed = pool_min_tokens - current_tokens
            for _, (rank, chunks, gt, n_tok) in ordered:
                if current_tokens + pad_tokens + n_tok > pool_max_tokens:
                    continue
                selected.append((rank, chunks, gt))
                pad_tokens += n_tok
                if pad_tokens >= needed:
                    break

            # Sort by rank ascending → rank-0 record at head → stable prefix.
            selected.sort(key=lambda x: x[0])
            pad_chunks: list = []
            pad_gts: list = []
            for _, chunks, gt in selected:
                pad_chunks.extend(chunks)
                pad_gts.append(gt)
            s.meta["_pool_chunks"] = pad_chunks
            s.meta["_pool_gts"] = pad_gts
