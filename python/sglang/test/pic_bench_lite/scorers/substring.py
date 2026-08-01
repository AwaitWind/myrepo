"""LongBench-style substring containment scorer."""
from sglang.test.pic_bench_lite.scorers.extract import extract_final_answer
from sglang.test.pic_bench_lite.scorers.normalize import normalize_answer


def substring_score(prediction: str, reference: str) -> float:
    """1.0 if normalize(reference) is a substring of
    normalize(extract_final_answer(prediction)[0]), else 0.0.

    Empty reference + empty (post-extract) prediction -> 1.0.
    Empty reference + non-empty prediction -> 0.0.
    """
    pred_text, _ = extract_final_answer(prediction)
    p = normalize_answer(pred_text)
    r = normalize_answer(reference)
    if not r:
        return 1.0 if not p else 0.0
    return 1.0 if r in p else 0.0
