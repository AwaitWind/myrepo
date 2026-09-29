"""YOLO #2 文本检测的伪标签生成：OCR 出候选框，GT 文本当白名单筛选。

为什么要造伪标签:
  官方 200 份 target.json 的顶层键**只有** {components, nets, pins}，没有任何
  文本区域标注（已核对 200/200）。论文的数据集标了 10 万个 text region，
  但赛题放出来的那份没带这一层。所以 YOLO #2 一个标签都没有，必须自己造。

做法（位置信任 OCR，内容归 GT 判定）:
  1. OCR 检测器给出候选框，每个框识别出一个字符串；
  2. 把 GT 的 Name / value / pinname 汇总成白名单；
  3. OCR 结果命中白名单 -> 保留该框作为训练标签；
  4. 命中不了 -> 丢弃，大概率是符号、导线或图框误检。

坐标说明 —— 和 comp_det 不同，这里**不做 y 翻转**:
  OCR 是直接在图像上跑的，输出已经是图像坐标（左上原点）。
  只有 GT 的 bbox / point 才是左下原点需要翻转。

已知局限，不要当成完美标签:
  - `GND`/`VCC`/`1`/`2` 这类字符串在一张图里重复出现几十次，命中白名单只能
    说明"这是文本"，无法定位到具体哪个元件。所以本脚本产出的 comp_keys 字段
    在重复文本上是多义的，**只可用于文本检测训练，不可当文本-元件配对的监督**。
  - 白名单只覆盖 GT 记录过的文本。图上的注释、标题栏、页码等不在 GT 里，
    会被判为误检丢掉。这会让 YOLO #2 学到"只检测有用文本"，是有意的取舍。

OCR 后端在本机不可用（paddleocr/easyocr/doctr 均未安装，且 PaddleOCR 权重的
`bj.bcebos.com` 不可达），本脚本的 OCR 部分**未在本机实测**，需在能联网的机器上跑。
产出是纯 JSON，拷回离线机即可供 export.py 使用。

用法（联网机器）:
    pip install paddleocr
    python3 yolo_model/text_det/pseudo_label.py --out out/text_pseudo
    python3 yolo_model/text_det/pseudo_label.py --out out/text_pseudo --backend easyocr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import cases as cases_mod  # noqa: E402


def norm_key(text: str) -> str:
    """匹配用的归一化。只用于判断"这个框里是不是 GT 里的某个文本"。

    注意：**绝不能**用归一化后的结果去填输出字段。评分是严格字符串匹配，
    `10k` / `10K` / `10 k` 是三个不同答案，GT 里 364 种写法并存是真实分布。
    这里归一化只为放宽匹配、多保住一些框。
    """
    t = unicodedata.normalize("NFKC", text)
    return "".join(t.split()).casefold()


class PaddleBackend:
    name = "paddleocr"

    def __init__(self, lang: str = "ch", model_dir: str | None = None):
        from paddleocr import PaddleOCR

        kwargs = {"lang": lang}
        if model_dir:
            kwargs["det_model_dir"] = os.path.join(model_dir, "det")
            kwargs["rec_model_dir"] = os.path.join(model_dir, "rec")
        self.ocr = PaddleOCR(**kwargs)

    def run(self, image_path: str):
        """返回 [(x1, y1, x2, y2, text, conf)]。"""
        raw = self.ocr.ocr(image_path)
        out = []
        for page in raw or []:
            for item in page or []:
                try:
                    poly, (text, conf) = item[0], item[1]
                except (TypeError, ValueError, IndexError):
                    continue
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                out.append((min(xs), min(ys), max(xs), max(ys), str(text), float(conf)))
        return out


class EasyOCRBackend:
    name = "easyocr"

    def __init__(self, lang: str = "en", model_dir: str | None = None):
        import easyocr

        langs = ["ch_sim", "en"] if lang == "ch" else ["en"]
        self.reader = easyocr.Reader(langs, model_storage_directory=model_dir)

    def run(self, image_path: str):
        out = []
        for poly, text, conf in self.reader.readtext(image_path):
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            out.append((min(xs), min(ys), max(xs), max(ys), str(text), float(conf)))
        return out


BACKENDS = {"paddleocr": PaddleBackend, "easyocr": EasyOCRBackend}


def make_backend(name: str, lang: str, model_dir: str | None):
    if name not in BACKENDS:
        raise SystemExit(f"未知后端 {name}，可选: {', '.join(BACKENDS)}")
    try:
        return BACKENDS[name](lang=lang, model_dir=model_dir)
    except ImportError as err:
        raise SystemExit(
            f"后端 {name} 不可用: {err}\n"
            f"在能联网的机器上先 `pip install {name}`。"
            "PaddleOCR 的权重站 bj.bcebos.com 在部分网络下不可达，"
            "此时可改用 --backend easyocr，或用 --ocr-model-dir 指向预下载的模型目录。"
        ) from err


def match_case(case, boxes, min_conf: float, relaxed: bool):
    """用 GT 文本白名单筛选 OCR 候选框。"""
    target = cases_mod.load_target(case.json_path)
    table = cases_mod.gt_texts(target)
    exact = dict(table)
    loose: dict[str, list] = {}
    if relaxed:
        for text, refs in table.items():
            loose.setdefault(norm_key(text), []).append((text, refs))

    kept, stats = [], Counter()
    for x1, y1, x2, y2, text, conf in boxes:
        stats["ocr_boxes"] += 1
        if conf < min_conf:
            stats["dropped_low_conf"] += 1
            continue
        if text in exact:
            gt_text, refs, mode = text, exact[text], "exact"
        elif relaxed and norm_key(text) in loose:
            cands = loose[norm_key(text)]
            gt_text, refs = cands[0]
            mode = "relaxed"
        else:
            stats["dropped_not_in_gt"] += 1
            continue
        stats[f"kept_{mode}"] += 1
        kept.append({
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "ocr_text": text,
            "gt_text": gt_text,
            "match": mode,
            "conf": round(conf, 4),
            "fields": sorted({f for f, _ in refs}),
            # 重复文本下多义，仅供排查，不可当配对监督
            "comp_keys": sorted({k for _, k in refs})[:8],
            "ambiguous": len({k for _, k in refs}) > 1,
        })
    stats["gt_text_entries"] = sum(len(v) for v in table.values())
    stats["gt_text_unique"] = len(table)
    return kept, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/text_pseudo")
    ap.add_argument("--official", default=None, help="不传=自动探测")
    ap.add_argument("--harvest", default=None, help="不传=自动探测")
    ap.add_argument("--backend", default="paddleocr", choices=sorted(BACKENDS))
    ap.add_argument("--lang", default="ch")
    ap.add_argument("--ocr-model-dir", default=None,
                    help="预下载的 OCR 模型目录，离线机器用")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--no-relaxed", action="store_true",
                    help="只接受严格相等的文本，不做 NFKC/大小写/空格归一化匹配")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个样本，调试用")
    ap.add_argument("--hash-cache", default="out/.image_hash_cache.json")
    a = ap.parse_args()

    roots = cases_mod.roots_from_args(a.official, a.harvest)
    all_cases, _, _ = cases_mod.discover(roots, hash_cache=a.hash_cache)
    if a.limit:
        all_cases = all_cases[: a.limit]

    backend = make_backend(a.backend, a.lang, a.ocr_model_dir)
    os.makedirs(a.out, exist_ok=True)
    total = Counter()
    index = []

    for i, case in enumerate(all_cases, 1):
        try:
            boxes = backend.run(case.image_path)
        except Exception as err:  # OCR 后端的异常类型不可预知，单例失败不该中断全批
            print(f"[text_pseudo] {case.case_id} OCR 失败: {type(err).__name__}: {err}")
            total["cases_failed"] += 1
            continue
        kept, stats = match_case(case, boxes, a.min_conf, not a.no_relaxed)
        total.update(stats)
        total["cases_done"] += 1
        rec = {
            "case_id": case.case_id,
            "image": case.image_path,
            "json": case.json_path,
            "source": case.source,
            "dataset": case.dataset,
            "group": case.group,
            "backend": backend.name,
            "boxes": kept,
            "stats": dict(stats),
        }
        with open(os.path.join(a.out, case.case_id + ".json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=1)
        index.append({"case_id": case.case_id, "boxes": len(kept), "source": case.source})
        if i % 20 == 0 or i == len(all_cases):
            print(f"[text_pseudo] {i}/{len(all_cases)}  累计保留框 "
                  f"{total['kept_exact'] + total['kept_relaxed']}")

    summary = {"args": vars(a), "totals": dict(total), "cases": index}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    kept_n = total["kept_exact"] + total["kept_relaxed"]
    print(f"\n[text_pseudo] 样本 成功/失败      {total['cases_done']} / {total['cases_failed']}")
    print(f"[text_pseudo] OCR 候选框          {total['ocr_boxes']}")
    print(f"[text_pseudo] 保留 严格/宽松      {total['kept_exact']} / {total['kept_relaxed']}")
    print(f"[text_pseudo] 丢弃 低置信/不在GT  {total['dropped_low_conf']} / {total['dropped_not_in_gt']}")
    if total["ocr_boxes"]:
        print(f"[text_pseudo] 白名单命中率        {kept_n / total['ocr_boxes'] * 100:.1f}%")
    print(f"[text_pseudo] GT 文本条目(可命中上限) {total['gt_text_entries']}")
    print(f"[text_pseudo] 输出                {a.out}")
    print("\n注意: 命中率偏低不一定是 OCR 差 —— 图上的注释、标题栏、页码不在 GT 白名单里，"
          "会被正常丢弃。要判断 OCR 质量，看的是 GT 文本条目的召回，不是候选框的留存率。")


if __name__ == "__main__":
    main()
