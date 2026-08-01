"""HotpotQA (distractor) dataset loader.

Preprocessed sample layout: one merged chunk containing all `<title>\\n<body>`
documents joined by "\\n\\n". `ground_truth` is a list (may contain the single
gold answer). Meta carries `dataset="hotpotqa"`.

Direct-invoke entry point:

    python -m sglang.test.pic_bench_lite.datasets.hotpotqa \\
        --raw-dir data/raw/hotpotqa \\
        --out    data/processed/hotpotqa/processed.jsonl
"""
import argparse
from pathlib import Path
from typing import Iterable, Iterator

from sglang.test.pic_bench_lite.datasets.base import (
    ProcessedSample,
    load_or_build_processed,
)


class HotpotQA:
    name = "hotpotqa"

    def preprocess_for_pic(self, raw_dir: Path) -> Iterator[ProcessedSample]:
        parquets = sorted(Path(raw_dir).glob("*.parquet"))
        if parquets:
            import pyarrow.parquet as pq
            for p in parquets:
                rows = pq.read_table(p).to_pylist()
                yield from self._preprocess_iter(rows)
            return
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", "distractor", split="validation", cache_dir=str(raw_dir))
        yield from self._preprocess_iter(iter(ds))

    @staticmethod
    def _preprocess_iter(rows: Iterable) -> Iterator[ProcessedSample]:
        for row in rows:
            titles = row["context"]["title"]
            sentences = row["context"]["sentences"]
            all_docs: list = []
            for title, sents in zip(titles, sentences):
                body = "".join(sents).strip()
                if not body:
                    continue
                all_docs.append(f"{title}\n{body}")
            if not all_docs:
                continue
            ans = row["answer"]
            yield ProcessedSample(
                sample_id=str(row["id"]),
                chunks=["\n\n".join(all_docs)],
                query=row["question"],
                ground_truth=[ans] if isinstance(ans, str) else list(ans),
                meta={"dataset": "hotpotqa", "n_supporting": 0},
            )


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Preprocess HotpotQA into pic_bench_lite jsonl.")
    ap.add_argument("--raw-dir", required=True, help="Directory with *.parquet files or HF cache root.")
    ap.add_argument("--out", required=True, help="Output jsonl path (auto-created).")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of samples written.")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out)
    ds = HotpotQA()
    it = ds.preprocess_for_pic(raw)
    if args.limit > 0:
        def _limited():
            for i, s in enumerate(it):
                if i >= args.limit:
                    break
                yield s
        it_final = _limited()
    else:
        it_final = it

    n = 0
    for _ in load_or_build_processed(out, lambda: it_final):
        n += 1
    print(f"HotpotQA preprocess done: wrote {n} samples -> {out}")


if __name__ == "__main__":
    _cli()
