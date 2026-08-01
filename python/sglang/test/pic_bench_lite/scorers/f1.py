"""Token-level F1 scorer (HotpotQA / SQuAD style)."""
from collections import Counter

from sglang.test.pic_bench_lite.scorers.extract import extract_final_answer
from sglang.test.pic_bench_lite.scorers.normalize import normalize_answer


def f1_score(prediction: str, reference) -> float:
    """Token-level F1 (HotpotQA / SQuAD style). Strips </think> reasoning prefix first.

    When reference is a list of candidate ground truths, return the max F1 over them.
    """
    if isinstance(reference, list):
        if not reference:
            return 0.0
        return max(f1_score(prediction, r) for r in reference)
    pred_text, _ = extract_final_answer(prediction)
    pred_tokens = normalize_answer(pred_text).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
