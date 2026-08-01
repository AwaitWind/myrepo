"""Dataset registry — populated lazily to avoid pulling in optional deps."""
from sglang.test.pic_bench_lite.datasets.base import (
    Dataset,
    ProcessedSample,
    load_or_build_processed,
)
from sglang.test.pic_bench_lite.datasets.hotpotqa import HotpotQA
from sglang.test.pic_bench_lite.datasets.longbench import (
    LongBench2WikiMQA,
    LongBenchGovReport,
    LongBenchMusique,
    LongBenchNarrativeQA,
    LongBenchQasper,
)

__all__ = [
    "Dataset",
    "ProcessedSample",
    "load_or_build_processed",
    "HotpotQA",
    "LongBench2WikiMQA",
    "LongBenchGovReport",
    "LongBenchMusique",
    "LongBenchNarrativeQA",
    "LongBenchQasper",
    "DATASETS",
    "get_dataset",
]

DATASETS: dict = {
    "hotpotqa": HotpotQA,
    "longbench_2wikimqa": LongBench2WikiMQA,
    "longbench_musique": LongBenchMusique,
    "longbench_narrativeqa": LongBenchNarrativeQA,
    "longbench_qasper": LongBenchQasper,
    "longbench_gov_report": LongBenchGovReport,
}


def get_dataset(name: str):
    """Look up a dataset class by name. Raises ValueError if unknown."""
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}"
        )
    return DATASETS[name]()
