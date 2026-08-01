"""Dataset base classes and processed-sample cache helpers.

Ported from pic_bench.workloads.datasets. Kept minimal on purpose: Phase 1
only needs ProcessedSample + load_or_build_processed. Dataset-specific
classes live in sibling modules (hotpotqa.py, longbench.py).
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Protocol, Union


@dataclass
class ProcessedSample:
    sample_id: str
    chunks: list
    query: str
    ground_truth: Union[str, list, None]
    meta: dict = field(default_factory=dict)


class Dataset(Protocol):
    name: str

    def preprocess_for_pic(self, raw_dir: Path) -> Iterator[ProcessedSample]: ...


def load_or_build_processed(
    cache_path: Path,
    builder: Callable[[], Iterator[ProcessedSample]],
    enrich: Union[Callable[[ProcessedSample], None], None] = None,
) -> Iterator[ProcessedSample]:
    """Load cached processed samples from jsonl, or build + cache then yield.

    If `enrich` is provided, it is called on each freshly built sample before
    the sample is written to the cache. Warm-cache reads do not invoke it.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        with cache_path.open() as fp:
            for line in fp:
                if not line.strip():
                    continue
                obj = json.loads(line)
                yield ProcessedSample(**obj)
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("w") as fp:
        for sample in builder():
            # Wrap every chunk with "\n\n" on both ends so segment cache
            # sees a stable, content-addressed boundary independent of the
            # adjacent chunk's tokenization. Applied uniformly across all
            # datasets at build time so the cache embeds the wrapping.
            sample.chunks = [f"\n\n{c}\n\n" for c in sample.chunks]
            if enrich is not None:
                enrich(sample)
            fp.write(json.dumps(asdict(sample)) + "\n")
            yield sample
    tmp.rename(cache_path)
