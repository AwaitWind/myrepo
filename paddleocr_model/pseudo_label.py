"""用 OCR 造 YOLO #2 的训练标签。

为什么必须造:
  官方 200 份 target.json 的顶层键只有 {components, nets, pins}（已核对 200/200），
  没有任何文本区域标注。论文自己的数据集标了 10 万个 text region，
  但赛题放出来的那份不含这一层。所以 YOLO #2 一个现成标签都没有。

做法（位置信任 OCR，内容归 GT 判定）:
  1. OCR 整图检测+识别，得到候选框和字符串；
  2. 把 GT 的 Name / value / pinname 汇总成白名单；
  3. 命中白名单 -> 保留该框作训练标签；
  4. 命中不了 -> 丢弃，大概率是符号、导线或图框误检。

坐标: OCR 直接在图上跑，输出已是图像坐标（左上原点），**不做 y 翻转**。
      只有 GT 的 bbox / point 才是左下原点需要翻。

两条已知局限，别当成完美标签:
  - GND / VCC / 1 / 2 这类字符串一张图里重复几十次，命中白名单只能说明
    "这是文本"，定位不到具体元件。所以 comp_keys 在重复文本上是多义的，
    **只可用于文本检测训练，不可当文本-元件配对的监督**。
  - 白名单只覆盖 GT 记录过的文本。图上的注释、标题栏、页码不在其中，会被
    判为误检丢掉。这会让 YOLO #2 学到"只检测有用文本"，是有意的取舍。

用法:
    python3 paddleocr_model/pseudo_label.py --out out/text_pseudo --limit 5
    python3 paddleocr_model/pseudo_label.py --out out/text_pseudo --backend easyocr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "yolo_model"))
import engine as eng  # noqa: E402
from common import cases as cases_mod  # noqa: E402


def norm_key(text: str) -> str:
    """匹配用的归一化。只用于判断"这框里是不是 GT 里的某个文本"。

    **绝不能**拿归一化结果去填输出字段。评分是严格字符串匹配，
    10k / 10K / 10 k 是三个不同答案，GT 里光 r/c 的值就有 652 种写法并存。
    这里放宽只为多保住一些框。
    """
    return "".join(unicodedata.normalize("NFKC", text).split()).casefold()


def match_case(case, boxes, min_conf: float, relaxed: bool):
    target = cases_mod.load_target(case.json_path)
    table = cases_mod.gt_texts(target)
    loose: dict[str, list] = {}
    if relaxed:
        for text, refs in table.items():
            loose.setdefault(norm_key(text), []).append((text, refs))

    kept, st = [], Counter()
    recovered = set()
    for x1, y1, x2, y2, text, conf in boxes:
        st["ocr_boxes"] += 1
        if conf < min_conf:
            st["dropped_low_conf"] += 1
            continue
        if text in table:
            gt_text, refs, mode = text, table[text], "exact"
        elif relaxed and norm_key(text) in loose:
            gt_text, refs = loose[norm_key(text)][0]
            mode = "relaxed"
        else:
            st["dropped_not_in_gt"] += 1
            continue
        st[f"kept_{mode}"] += 1
        recovered.add(gt_text)
        kept.append({
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "ocr_text": text, "gt_text": gt_text, "match": mode,
            "conf": round(conf, 4),
            "fields": sorted({f for f, _ in refs}),
            "comp_keys": sorted({k for _, k in refs})[:8],
            "ambiguous": len({k for _, k in refs}) > 1,
        })
    st["gt_text_unique"] = len(table)
    st["gt_text_recovered"] = len(recovered)
    return kept, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/text_pseudo")
    ap.add_argument("--official", default=None, help="不传=自动探测")
    ap.add_argument("--harvest", default=None, help="不传=自动探测")
    ap.add_argument("--backend", default="paddleocr", choices=sorted(eng.BACKENDS))
    ap.add_argument("--lang", default="ch")
    ap.add_argument("--ocr-model-dir", default=None, help="预下载的 OCR 模型目录")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--no-relaxed", action="store_true",
                    help="只接受严格相等，不做 NFKC/大小写/空格归一化匹配")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个样本")
    ap.add_argument("--hash-cache", default="out/.image_hash_cache.json")
    a = ap.parse_args()

    roots = cases_mod.roots_from_args(a.official, a.harvest)
    all_cases, _, _ = cases_mod.discover(roots, hash_cache=a.hash_cache)
    if a.limit:
        all_cases = all_cases[: a.limit]

    backend = eng.build(a.backend, a.lang, a.ocr_model_dir, a.gpu)
    os.makedirs(a.out, exist_ok=True)
    total = Counter()
    index = []

    for i, case in enumerate(all_cases, 1):
        try:
            boxes = backend.detect_and_read(case.image_path)
        except Exception as err:   # OCR 后端异常类型不可预知，单例失败不该中断全批
            print(f"[pseudo] {case.case_id} OCR 失败: {type(err).__name__}: {err}")
            total["cases_failed"] += 1
            continue
        kept, st = match_case(case, boxes, a.min_conf, not a.no_relaxed)
        total.update(st)
        total["cases_done"] += 1
        with open(os.path.join(a.out, case.case_id + ".json"), "w", encoding="utf-8") as fh:
            json.dump({"case_id": case.case_id, "image": case.image_path,
                       "json": case.json_path, "source": case.source,
                       "dataset": case.dataset, "group": case.group,
                       "backend": backend.name, "boxes": kept,
                       "stats": dict(st)}, fh, ensure_ascii=False, indent=1)
        index.append({"case_id": case.case_id, "boxes": len(kept),
                      "source": case.source})
        if i % 20 == 0 or i == len(all_cases):
            print(f"[pseudo] {i}/{len(all_cases)}  累计保留框 "
                  f"{total['kept_exact'] + total['kept_relaxed']}", flush=True)

    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"args": vars(a), "totals": dict(total), "cases": index},
                  fh, ensure_ascii=False, indent=2)

    kept_n = total["kept_exact"] + total["kept_relaxed"]
    uniq, rec = total["gt_text_unique"], total["gt_text_recovered"]
    print(f"\n[pseudo] 样本 成功/失败      {total['cases_done']} / {total['cases_failed']}")
    print(f"[pseudo] OCR 候选框          {total['ocr_boxes']}")
    print(f"[pseudo] 保留 严格/宽松      {total['kept_exact']} / {total['kept_relaxed']}")
    print(f"[pseudo] 丢弃 低置信/不在GT  {total['dropped_low_conf']} / {total['dropped_not_in_gt']}")
    if total["ocr_boxes"]:
        print(f"[pseudo] 候选框留存率        {kept_n / total['ocr_boxes'] * 100:.1f}%")
    if uniq:
        print(f"[pseudo] **GT 文本召回率**   {rec}/{uniq} = {rec / uniq * 100:.1f}%")
    print(f"[pseudo] 输出                {a.out}")
    print("\n判断 OCR 质量看 **GT 文本召回率**，不是候选框留存率 —— 留存率低往往只是"
          "因为图上的注释、标题栏、页码不在 GT 白名单里，被正常丢弃了。")


if __name__ == "__main__":
    main()
