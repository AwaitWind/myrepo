"""Scorer registry — populated by submodules.

Import lazily so a missing optional dependency (e.g. `rouge_score`) doesn't
break f1/substring users.
"""
from sglang.test.pic_bench_lite.scorers.f1 import f1_score
from sglang.test.pic_bench_lite.scorers.substring import substring_score

SCORERS: dict = {
    "f1": f1_score,
    "substring": substring_score,
}

try:
    from sglang.test.pic_bench_lite.scorers.rouge import rouge_l_score
    SCORERS["rouge_l"] = rouge_l_score
except ImportError:
    pass
