"""LongBench (narrativeqa / qasper / gov_report) dataset loaders.

Ported from pic_bench.workloads.datasets_longbench. narrativeqa / gov_report
optionally chunk the long context with a tokenizer (set env
`PIC_BENCH_TOKENIZER=<model_path>` to enable token-accurate chunking; falls
back to a char-based approximation otherwise). Qasper keeps a single chunk.
"""
import os
from pathlib import Path
from typing import Iterable, Iterator

from sglang.test.pic_bench_lite.datasets.base import ProcessedSample


def _preprocess_longbench_iter(rows: Iterable, tag: str) -> Iterator[ProcessedSample]:
    for row in rows:
        ctx = row.get("context", "")
        if not ctx:
            continue
        answers = row.get("answers") or []
        gt = list(answers) if answers else None
        yield ProcessedSample(
            sample_id=str(row["_id"]), chunks=[ctx], query=row["input"],
            ground_truth=gt, meta={"dataset": tag},
        )


def _load_parquets_or_hf(raw_dir: Path, hf_config: str):
    import json
    raw_dir = Path(raw_dir)
    jsonls = sorted(raw_dir.glob("*.jsonl"))
    if jsonls:
        for p in jsonls:
            for line in p.read_text().splitlines():
                if line.strip():
                    yield json.loads(line)
        return
    parquets = sorted(raw_dir.glob("*.parquet"))
    if parquets:
        import pyarrow.parquet as pq
        for p in parquets:
            yield from pq.read_table(p).to_pylist()
        return
    from datasets import load_dataset
    ds = load_dataset("THUDM/LongBench", hf_config, split="test", cache_dir=str(raw_dir))
    yield from iter(ds)


def _chunk_text_by_tokens(text: str, chunk_tokens: int, tokenizer=None) -> list:
    """Split text into chunks of ~chunk_tokens tokens.

    Tries paragraph boundaries (\\n\\n) first; falls back to word-level slicing
    when the whole text is a single block (no paragraph breaks).
    """
    if tokenizer is None:
        chunk_chars = chunk_tokens * 4  # ~4 chars/token approximation
        if len(text) <= chunk_chars:
            return [text] if text.strip() else []
        chunks = []
        pos = 0
        while pos < len(text):
            chunks.append(text[pos:pos + chunk_chars])
            pos += chunk_chars
        return [c for c in chunks if c.strip()]

    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= chunk_tokens:
        return [text] if text.strip() else []

    paragraphs = text.split("\n\n")
    if len(paragraphs) > 1:
        chunks = []
        current_parts = []
        current_tokens = 0
        for para in paragraphs:
            para_tokens = len(tokenizer.encode(para, add_special_tokens=False))
            if current_tokens + para_tokens > chunk_tokens and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = [para]
                current_tokens = para_tokens
            else:
                current_parts.append(para)
                current_tokens += para_tokens
        if current_parts:
            chunks.append("\n\n".join(current_parts))
        if len(chunks) > 1:
            return [c for c in chunks if c.strip()]

    chunks = []
    for start in range(0, len(ids), chunk_tokens):
        chunk_ids = ids[start:start + chunk_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return [c for c in chunks if c.strip()]


def _try_load_tokenizer():
    tok_path = os.environ.get("PIC_BENCH_TOKENIZER", "")
    if not tok_path:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    except Exception:
        return None


class LongBenchNarrativeQA:
    name = "longbench_narrativeqa"
    _CHUNK_TOKENS = 16384

    def preprocess_for_pic(self, raw_dir: Path) -> Iterator[ProcessedSample]:
        tokenizer = _try_load_tokenizer()
        yield from self._preprocess_iter(_load_parquets_or_hf(raw_dir, "narrativeqa"), tokenizer)

    @classmethod
    def _preprocess_iter(cls, rows: Iterable, tokenizer=None) -> Iterator[ProcessedSample]:
        for row in rows:
            ctx = row.get("context", "")
            if not ctx:
                continue
            answers = row.get("answers") or []
            gt = list(answers) if answers else None
            chunks = _chunk_text_by_tokens(ctx, cls._CHUNK_TOKENS, tokenizer)
            if not chunks:
                continue
            yield ProcessedSample(
                sample_id=str(row["_id"]), chunks=chunks, query=row["input"],
                ground_truth=gt, meta={"dataset": "longbench_narrativeqa"},
            )


class LongBenchQasper:
    name = "longbench_qasper"

    def preprocess_for_pic(self, raw_dir: Path) -> Iterator[ProcessedSample]:
        yield from _preprocess_longbench_iter(
            _load_parquets_or_hf(raw_dir, "qasper"), tag="longbench_qasper"
        )


class LongBenchGovReport:
    name = "longbench_gov_report"
    _query = "Write a one-paragraph summary of the report above."
    _CHUNK_TOKENS = 4096

    def preprocess_for_pic(self, raw_dir: Path) -> Iterator[ProcessedSample]:
        tokenizer = _try_load_tokenizer()
        yield from self._preprocess_iter(
            _load_parquets_or_hf(raw_dir, "gov_report"), tokenizer,
        )

    @classmethod
    def _preprocess_iter(cls, rows: Iterable, tokenizer=None) -> Iterator[ProcessedSample]:
        for row in rows:
            ctx = row.get("context", "")
            if not ctx:
                continue
            answers = row.get("answers") or []
            gt = answers[0] if answers else None
            chunks = _chunk_text_by_tokens(ctx, cls._CHUNK_TOKENS, tokenizer)
            if not chunks:
                continue
            yield ProcessedSample(
                sample_id=str(row["_id"]),
                chunks=chunks,
                query=cls._query,
                ground_truth=gt,
                meta={"dataset": "longbench_gov_report"},
            )


# ── Multi-hop MULTI-DOC QA (2wikimqa / musique) ──────────────────────────────
# LongBench multi-hop QA: each sample's `context` is the SET OF PASSAGES provided
# for that question (supporting + retrieved distractor passages — all relevant to
# THIS query, i.e. a real retrieved multi-doc set, NOT a cross-sample stuffed
# pool). We chunk the context into multiple segments (~PIC_BENCH_CHUNK_TOKENS,
# default 1024) so PIC/oracle sees genuine multi-segment RELEVANT content. Score
# with F1 against `answers`. Run with --no-dataset-pool (the context already IS
# the multi-doc set). This is the fair "realistic RAG" test for pic_a3_oracle.
def _preprocess_multidoc(
    rows: Iterable, tag: str, chunk_tokens: int, tokenizer=None
) -> Iterator[ProcessedSample]:
    for row in rows:
        ctx = row.get("context", "")
        if not ctx:
            continue
        answers = row.get("answers") or []
        gt = list(answers) if answers else None
        chunks = _chunk_text_by_tokens(ctx, chunk_tokens, tokenizer)
        if not chunks:
            continue
        yield ProcessedSample(
            sample_id=str(row["_id"]), chunks=chunks, query=row["input"],
            ground_truth=gt, meta={"dataset": tag},
        )


class LongBench2WikiMQA:
    name = "longbench_2wikimqa"

    def preprocess_for_pic(self, raw_dir: Path) -> Iterator[ProcessedSample]:
        tok = _try_load_tokenizer()
        ck = int(os.environ.get("PIC_BENCH_CHUNK_TOKENS", "1024"))
        yield from _preprocess_multidoc(
            _load_parquets_or_hf(raw_dir, "2wikimqa"),
            "longbench_2wikimqa", ck, tok,
        )


class LongBenchMusique:
    name = "longbench_musique"

    def preprocess_for_pic(self, raw_dir: Path) -> Iterator[ProcessedSample]:
        tok = _try_load_tokenizer()
        ck = int(os.environ.get("PIC_BENCH_CHUNK_TOKENS", "1024"))
        yield from _preprocess_multidoc(
            _load_parquets_or_hf(raw_dir, "musique"),
            "longbench_musique", ck, tok,
        )
