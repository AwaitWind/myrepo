"""ROUGE-1 / ROUGE-2 / ROUGE-L scorer. Requires the `rouge_score` package."""
from sglang.test.pic_bench_lite.scorers.extract import extract_final_answer
from sglang.test.pic_bench_lite.scorers.normalize import normalize_answer

try:
    from rouge_score import rouge_scorer
    _SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
except ImportError:  # rouge_score is optional
    _SCORER = None


def rouge_scores(prediction: str, reference: str) -> dict:
    """ROUGE-1/2/L F-measures on normalized text. Strips </think> prefix first.
    Both empty -> all 1.0; either empty (xor) -> all 0.0.
    """
    if _SCORER is None:
        raise ImportError(
            "rouge_score is not installed. `pip install rouge-score` to use ROUGE scorers."
        )
    pred_text, _ = extract_final_answer(prediction)
    p = normalize_answer(pred_text)
    r = normalize_answer(reference)
    if not p and not r:
        return {"rouge_1": 1.0, "rouge_2": 1.0, "rouge_l": 1.0}
    if not p or not r:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    s = _SCORER.score(r, p)
    return {
        "rouge_1": s["rouge1"].fmeasure,
        "rouge_2": s["rouge2"].fmeasure,
        "rouge_l": s["rougeL"].fmeasure,
    }


def rouge_l_score(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure (back-compat wrapper around rouge_scores)."""
    return rouge_scores(prediction, reference)["rouge_l"]
