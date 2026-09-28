"""评测 OCR 质量。对齐论文 Table II 的 CER / WER / ACC。

**必须避开的循环论证**:
  伪标签里的框是"因为 OCR 认对了才被保留"的，拿那批框去算 CER 必然是 0。
  所以本脚本不用伪标签做评测集，改用两个不循环的指标：

  1. GT 文本召回率（主指标）
     一张图的 GT 文本集合里，有多少被 OCR 原样认出来了。
     分母是 GT，与 OCR 表现无关，因此不循环。

  2. 近似匹配下的 CER / WER（诊断用）
     对没被精确召回的 GT 文本，在 OCR 的全部输出里找编辑距离最近的一条配对，
     算字符错误率。它回答的是"认错的时候错成什么样"，
     用来定位易混字符（Ω/O/0、1/l/I、μ/u）。
     注意这是**乐观估计** —— 最近邻配对本身就假设了 OCR 至少认了个八九不离十。

评分口径提醒：赛题是严格字符串匹配，10k / 10K / 10 k 判为不同。
所以主指标只认原样相等，不做任何归一化。

用法:
    python3 paddleocr_model/eval_ocr.py --limit 20
    python3 paddleocr_model/eval_ocr.py --limit 20 --backend easyocr --report out/ocr_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "yolo_model"))
import engine as eng  # noqa: E402
from common import cases as cases_mod  # noqa: E402


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def nearest(target: str, pool):
    best, bestd = None, None
    for cand in pool:
        d = edit_distance(target, cand)
        if bestd is None or d < bestd:
            best, bestd = cand, d
            if d == 0:
                break
    return best, bestd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", default=None)
    ap.add_argument("--harvest", default="")
    ap.add_argument("--backend", default="paddleocr", choices=sorted(eng.BACKENDS))
    ap.add_argument("--lang", default="ch")
    ap.add_argument("--ocr-model-dir", default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--report", default="out/ocr_eval.json")
    ap.add_argument("--hash-cache", default="out/.image_hash_cache.json")
    a = ap.parse_args()

    roots = cases_mod.roots_from_args(a.official, a.harvest)
    all_cases, _, _ = cases_mod.discover(roots, hash_cache=a.hash_cache)
    if a.limit:
        all_cases = all_cases[: a.limit]
    backend = eng.build(a.backend, a.lang, a.ocr_model_dir, a.gpu)

    tot = Counter()
    confus = Counter()
    rows = []
    for i, case in enumerate(all_cases, 1):
        try:
            boxes = backend.detect_and_read(case.image_path)
        except Exception as err:
            print(f"[eval] {case.case_id} 失败: {type(err).__name__}: {err}")
            tot["cases_failed"] += 1
            continue
        preds = [t for *_, t, c in boxes if c >= a.min_conf]
        pred_set = set(preds)
        gt = cases_mod.gt_texts(cases_mod.load_target(case.json_path))

        hit = sum(1 for g in gt if g in pred_set)
        miss = [g for g in gt if g not in pred_set]
        tot["gt_unique"] += len(gt)
        tot["gt_hit"] += hit
        tot["ocr_outputs"] += len(preds)

        chars = errs = words = werr = 0
        for g in miss:
            cand, d = nearest(g, pred_set)
            if cand is None:
                continue
            chars += len(g); errs += d
            words += 1; werr += 1
            if d <= 2 and len(g) == len(cand):
                for x, y in zip(g, cand):
                    if x != y:
                        confus[f"{x}->{y}"] += 1
        # 精确命中的部分计入分母，否则 CER 会被严重高估
        for g in gt:
            if g in pred_set:
                chars += len(g); words += 1
        tot["chars"] += chars; tot["char_err"] += errs
        tot["words"] += words; tot["word_err"] += werr

        rows.append({"case_id": case.case_id, "gt_unique": len(gt), "gt_hit": hit,
                     "ocr_outputs": len(preds)})
        if i % 5 == 0 or i == len(all_cases):
            print(f"[eval] {i}/{len(all_cases)}  GT召回 "
                  f"{tot['gt_hit']}/{tot['gt_unique']}", flush=True)

    recall = tot["gt_hit"] / tot["gt_unique"] if tot["gt_unique"] else 0.0
    cer = tot["char_err"] / tot["chars"] if tot["chars"] else 0.0
    wer = tot["word_err"] / tot["words"] if tot["words"] else 0.0
    rep = {"args": vars(a), "totals": dict(tot),
           "gt_text_recall": recall, "cer": cer, "wer": wer,
           "top_confusions": confus.most_common(25), "cases": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.report)) or ".", exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)

    print(f"\n[eval] 后端 {backend.name}  样本 {len(rows)}")
    print(f"[eval] GT 文本召回率(主指标)  {tot['gt_hit']}/{tot['gt_unique']} = {recall*100:.1f}%")
    print(f"[eval] CER {cer*100:.2f}%   WER {wer*100:.2f}%   (近似匹配，偏乐观)")
    if confus:
        print("[eval] 高频混淆: " + "  ".join(f"{k}({v})" for k, v in confus.most_common(10)))
    print(f"[eval] 报告 {a.report}")
    print("\n参照论文 Table II: 整图 OCR 的 CER 21.35%，YOLO 框好再识别 6.33%。"
          "本脚本跑的是整图模式，所以这里的数字对应前者；"
          "等 YOLO #2 训好之后走 read_crop，应当明显更好。")


if __name__ == "__main__":
    main()
