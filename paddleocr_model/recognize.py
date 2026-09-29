"""正式流程的识别环节：YOLO #2 给框 -> 裁剪放大 -> OCR 只做识别。

这是训练完成后真正要用的那条路，和 pseudo_label.py 的整图模式不同。
论文 Table II：整图直接 OCR 的 CER 是 21.35%，先框后认降到 6.33%。

输入框的两个来源:
  --boxes-from yolo    用训好的 YOLO #2 权重现场检测（切片推理 + 跨片 NMS）
  --boxes-from pseudo  从伪标签 JSON 读框，用来在没训完 YOLO #2 时先打通链路

输出 {case_id: [{bbox, text, conf}]}，供后续的文本-元件配对使用。

绝对不要归一化识别结果：评分是严格字符串匹配，10k / 10K / 10 k 是不同答案。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "yolo_model"))
import engine as eng  # noqa: E402
from common import cases as cases_mod  # noqa: E402
from common import tiling  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def nms(boxes, iou_thr=0.3):
    """跨片合并。阈值取低，因为文本框之间本来重叠就少。"""
    if not boxes:
        return []
    arr = np.array([b[:4] for b in boxes], dtype=float)
    scores = np.array([b[4] for b in boxes], dtype=float)
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(1e-6, areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thr]
    return [boxes[i] for i in keep]


def yolo_boxes(model, image_path, window, overlap, conf):
    """切片推理再把框映射回全图坐标，最后跨片 NMS。"""
    with Image.open(image_path) as raw:
        im = raw.convert("RGB")
        W, H = im.size
        stride = tiling.stride_of(window, overlap)
        out = []
        for oy in tiling.tile_origins(H, window, stride):
            for ox in tiling.tile_origins(W, window, stride):
                tile = im.crop((ox, oy, ox + window, oy + window))
                for r in model.predict(tile, conf=conf, verbose=False):
                    for b in r.boxes:
                        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                        out.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy,
                                    float(b.conf[0])))
    return nms(out)


def pseudo_boxes(pseudo_dir, case_id):
    p = os.path.join(pseudo_dir, case_id + ".json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        rec = json.load(fh)
    return [(*b["bbox"], b.get("conf", 1.0)) for b in rec.get("boxes", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/ocr_text.json")
    ap.add_argument("--boxes-from", choices=["yolo", "pseudo"], default="yolo")
    ap.add_argument("--weights", default="out/yolo_runs/text_det/weights/best.pt")
    ap.add_argument("--pseudo", default="out/text_pseudo")
    ap.add_argument("--official", default=None)
    ap.add_argument("--harvest", default=None)
    ap.add_argument("--backend", default="paddleocr", choices=sorted(eng.BACKENDS))
    ap.add_argument("--lang", default="ch")
    ap.add_argument("--ocr-model-dir", default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--window", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hash-cache", default="out/.image_hash_cache.json")
    a = ap.parse_args()

    roots = cases_mod.roots_from_args(a.official, a.harvest)
    all_cases, _, _ = cases_mod.discover(roots, hash_cache=a.hash_cache)
    if a.limit:
        all_cases = all_cases[: a.limit]

    model = None
    if a.boxes_from == "yolo":
        if not os.path.exists(a.weights):
            raise SystemExit(
                f"找不到 YOLO #2 权重 {a.weights}。\n"
                "  先训练: bash yolo_model/run.sh --what text --device 0\n"
                "  或先用伪标签的框打通链路: --boxes-from pseudo")
        from ultralytics import YOLO
        model = YOLO(a.weights)

    backend = eng.build(a.backend, a.lang, a.ocr_model_dir, a.gpu)
    result = {}
    n_box = n_txt = 0
    for i, case in enumerate(all_cases, 1):
        boxes = (yolo_boxes(model, case.image_path, a.window, a.overlap, a.conf)
                 if model else pseudo_boxes(a.pseudo, case.case_id))
        n_box += len(boxes)
        with Image.open(case.image_path) as raw:
            img = np.asarray(raw.convert("RGB"))
        items = []
        for x1, y1, x2, y2, bconf in boxes:
            crop = eng.prepare_crop(img, (x1, y1, x2, y2))
            if crop is None:
                continue
            try:
                res = backend.read_crop_rot(crop)
            except Exception as err:
                print(f"[ocr] {case.case_id} 识别失败: {type(err).__name__}: {err}")
                continue
            if not res or not res[0]:
                continue
            text, conf = res
            items.append({"bbox": [round(x1, 2), round(y1, 2),
                                   round(x2, 2), round(y2, 2)],
                          "text": text, "conf": round(float(conf), 4),
                          "box_conf": round(float(bconf), 4)})
        n_txt += len(items)
        result[case.case_id] = items
        if i % 10 == 0 or i == len(all_cases):
            print(f"[ocr] {i}/{len(all_cases)}  框 {n_box}  识别出 {n_txt}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(f"\n[ocr] 样本 {len(result)}  文本框 {n_box}  识别出文本 {n_txt}")
    print(f"[ocr] 输出 {a.out}")
    print("[ocr] 下一步是文本-元件配对（最近邻 + 类别一致性/几何对齐/局部唯一性"
          "三条约束），那部分尚未实现。")


if __name__ == "__main__":
    main()
