"""单张图片跑 OCR，看识别结果。摸底用，不参与训练流程。

和另外三个脚本的区别：它们都按数据集目录批量跑，这个只吃一张图。

两种模式:
  --boxes whole   整图交给 OCR 自带检测器（对应论文 Table II 的 21.35% CER 那一档）
  --boxes yolo    用 YOLO #2 权重先框文本，再逐块裁剪放大识别（6.33% 那一档）

产出:
  <out>.json  每个框的 bbox / text / conf
  <out>.png   把框和识别结果画在原图上，直接看效果
  若图片旁边有 *_target.json，额外报一个 GT 文本召回率（严格相等，不做归一化）

用法（在仓库根目录执行，即能看见 赛题六公开数据集/ 或 data/ 的地方）:
    python3 paddleocr_model/test/ocr_one.py 赛题六公开数据集/200_train_cases/0006/0006-jlc.png
    python3 paddleocr_model/test/ocr_one.py 图.png --boxes yolo \
        --weights out/yolo_runs/text_det/weights/best.pt
    python3 paddleocr_model/test/ocr_one.py 图.png --backend stub   # 只验管道，不需要权重
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))     # paddleocr_model/test
_PKG = os.path.dirname(_HERE)                          # paddleocr_model
_ROOT = os.path.dirname(_PKG)                          # 仓库根
sys.path.insert(0, _PKG)                               # engine / recognize
sys.path.insert(0, os.path.join(_ROOT, "yolo_model"))  # common.cases / common.tiling
import engine as eng  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


class Paddle3(eng._Base):
    """PaddleOCR 3.x。两处非默认设置:

    1. 关掉文档方向/矫正子模型 —— 对原理图裁剪块无用，
       但默认开启会多下载三个模型、显著拖慢首次启动。
    2. 默认关掉 oneDNN(mkldnn)。paddlepaddle 3.3.1 的 oneDNN 后端在跑文本检测时
       会崩在 `ConvertPirAttribute2RuntimeAttribute not support
       [pir::ArrayAttribute<pir::DoubleAttribute>]`（onednn_instruction.cc:116），
       某个属性类型的 PIR 转换尚未实现。关掉它走通用 CPU 算子即可绕过，
       代价是 CPU 推理变慢。确认自己的 paddle 版本没这个 bug 可以加 --mkldnn 开回来。

    老版本不认这些参数：PaddleOCR 自身抛 TypeError，通用参数走 paddleocr 的
    parse_common_args 校验、抛的是 ValueError，两种都要接。
    """

    name = "paddleocr"
    _OFF = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }

    def __init__(self, lang="en", model_dir=None, mkldnn=False, device=None):
        from paddleocr import PaddleOCR

        kw = {"lang": lang, **self._OFF}
        if device:
            kw["device"] = device
        # oneDNN 是 CPU 专用加速库，GPU 上这条开关无意义，就不传了
        if not mkldnn and not (device or "").startswith("gpu"):
            kw["enable_mkldnn"] = False
        if model_dir:
            det, rec = os.path.join(model_dir, "det"), os.path.join(model_dir, "rec")
            if os.path.isdir(det):
                kw["text_detection_model_dir"] = det
            if os.path.isdir(rec):
                kw["text_recognition_model_dir"] = rec
        while True:
            try:
                self.ocr = PaddleOCR(**kw)
                break
            except (TypeError, ValueError) as err:
                drop = next((k for k in list(kw) if k != "lang" and k in str(err)), None)
                if drop is None:
                    raise
                kw.pop(drop)
                print(f"[ocr] 当前 PaddleOCR 不支持 {drop}，已忽略")

    def _run(self, target):
        fn = getattr(self.ocr, "predict", None) or self.ocr.ocr
        return fn(target)

    def detect_and_read(self, image_path: str):
        out = []
        for poly, text, conf in eng.PaddleBackend._iter_items(self._run(image_path)):
            if poly is None:
                continue
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            out.append((min(xs), min(ys), max(xs), max(ys), text, conf))
        return out

    def read_crop(self, crop: np.ndarray):
        best = None
        for _, text, conf in eng.PaddleBackend._iter_items(self._run(crop)):
            if best is None or conf > best[1]:
                best = (text, conf)
        return best


def build_backend(name: str, lang: str, model_dir: str | None, gpu: bool,
                  mkldnn: bool = False):
    if name == "paddleocr":
        try:
            return Paddle3(lang=lang, model_dir=model_dir, mkldnn=mkldnn,
                           device="gpu" if gpu else None)
        except ImportError as err:
            raise SystemExit(
                f"paddleocr 不可用: {err}\n"
                "  两个包都要装: pip install paddlepaddle paddleocr\n"
                "  （paddleocr 的依赖里不含 paddlepaddle，必须单独装；"
                "GPU 版是 paddlepaddle-gpu，注意它可能和 torch 的 CUDA 打架）\n"
                "  只想验证管道接线: --backend stub"
            ) from err
    return eng.build(name, lang, model_dir, gpu)


def yolo_boxes(weights: str, image_path: str, window: int, overlap: float, conf: float):
    """切片推理 + 跨片 NMS。复用 recognize.py 的实现，保持与正式流程一致。"""
    import recognize as rec
    from ultralytics import YOLO

    return rec.yolo_boxes(YOLO(weights), image_path, window, overlap, conf)


def gt_recall(image_path: str, texts):
    """图片旁边有 GT 就报召回率。严格相等，不做任何归一化（评分口径如此）。"""
    d = os.path.dirname(os.path.abspath(image_path))
    cand = [f for f in os.listdir(d) if f.endswith("_target.json")
            and not f.startswith("._")]
    if not cand:
        return None
    from common import cases as cases_mod

    gt = cases_mod.gt_texts(cases_mod.load_target(os.path.join(d, cand[0])))
    got = set(texts)
    hit = sorted(g for g in gt if g in got)
    miss = sorted(g for g in gt if g not in got)
    return {"gt_file": cand[0], "gt_unique": len(gt), "hit": len(hit),
            "recall": round(len(hit) / len(gt), 4) if gt else 0.0,
            "missed_sample": miss[:30]}


def draw(image_path: str, items, out_png: str):
    with Image.open(image_path) as raw:
        im = raw.convert("RGB")
    dr = ImageDraw.Draw(im)
    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        dr.text((x1, max(0, y1 - 11)), it["text"], fill=(0, 0, 255))
    im.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="图片路径")
    ap.add_argument("--boxes", choices=["whole", "yolo"], default="whole",
                    help="whole=OCR 自带检测器整图跑；yolo=用 YOLO #2 先框再逐块识别")
    ap.add_argument("--weights", default="out/yolo_runs/text_det/weights/best.pt")
    ap.add_argument("--backend", default="paddleocr",
                    choices=["paddleocr", "easyocr", "stub"])
    ap.add_argument("--lang", default="en",
                    help="原理图文本基本是 ASCII，用 en 比 ch 更准也更快")
    ap.add_argument("--ocr-model-dir", default=None, help="预下载的模型目录，离线机用")
    ap.add_argument("--gpu", action="store_true",
                    help="用 GPU 推理。需要 paddlepaddle-gpu；GPU 上不走 oneDNN，"
                         "所以也顺带绕开了那个 crash")
    ap.add_argument("--mkldnn", action="store_true",
                    help="开回 oneDNN 加速。默认关闭 —— paddlepaddle 3.3.1 的 oneDNN "
                         "后端跑文本检测会崩在 ConvertPirAttribute2RuntimeAttribute")
    ap.add_argument("--window", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO 框的置信度下限")
    ap.add_argument("--min-conf", type=float, default=0.0, help="OCR 识别置信度下限")
    ap.add_argument("--out", default=None, help="默认 out/ocr_one/<图名>")
    a = ap.parse_args()

    if not os.path.exists(a.image):
        raise SystemExit(f"找不到图片: {a.image}")
    stem = os.path.splitext(os.path.basename(a.image))[0]
    out_base = a.out or os.path.join("out", "ocr_one", stem)
    os.makedirs(os.path.dirname(os.path.abspath(out_base)) or ".", exist_ok=True)

    with Image.open(a.image) as raw:
        W, H = raw.size
        img = np.asarray(raw.convert("RGB"))
    print(f"[ocr] 图片 {a.image}  {W}x{H}  模式 {a.boxes}  后端 {a.backend}")

    backend = build_backend(a.backend, a.lang, a.ocr_model_dir, a.gpu, a.mkldnn)

    items = []
    if a.boxes == "whole":
        for x1, y1, x2, y2, text, conf in backend.detect_and_read(a.image):
            if conf < a.min_conf:
                continue
            items.append({"bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                          "text": text, "conf": round(float(conf), 4)})
    else:
        if not os.path.exists(a.weights):
            raise SystemExit(
                f"找不到 YOLO #2 权重 {a.weights}。\n"
                "  它还没训（缺伪标签）。先看整图模式的效果: --boxes whole")
        boxes = yolo_boxes(a.weights, a.image, a.window, a.overlap, a.conf)
        print(f"[ocr] YOLO 框出 {len(boxes)} 个文本区域")
        for x1, y1, x2, y2, bconf in boxes:
            crop = eng.prepare_crop(img, (x1, y1, x2, y2))
            if crop is None:
                continue
            res = backend.read_crop_rot(crop)
            if not res or not res[0] or res[1] < a.min_conf:
                continue
            items.append({"bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                          "text": res[0], "conf": round(float(res[1]), 4),
                          "box_conf": round(float(bconf), 4)})

    rec = gt_recall(a.image, [it["text"] for it in items])
    payload = {"image": a.image, "size": [W, H], "mode": a.boxes,
               "backend": backend.name, "n_text": len(items),
               "gt_check": rec, "items": items}
    with open(out_base + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    draw(a.image, items, out_base + ".png")

    print(f"\n[ocr] 识别出 {len(items)} 条文本")
    for it in items[:25]:
        print(f"  {it['conf']:.3f}  {it['text']!r}  @ {it['bbox']}")
    if len(items) > 25:
        print(f"  ...（共 {len(items)} 条，全部见 JSON）")
    if rec:
        print(f"\n[ocr] GT 文本召回 {rec['hit']}/{rec['gt_unique']} = {rec['recall']*100:.1f}%"
              f"  (对照 {rec['gt_file']}，严格相等)")
        if rec["missed_sample"]:
            print("[ocr] 未召回样例: " + "  ".join(repr(t) for t in rec["missed_sample"][:12]))
    print(f"\n[ocr] JSON  {out_base}.json")
    print(f"[ocr] 可视化 {out_base}.png  （红框=位置，蓝字=识别结果）")
    if a.boxes == "whole":
        print("[ocr] 这是整图模式，对应论文 Table II 的 CER 21.35% 那一档。"
              "YOLO #2 训好后走 --boxes yolo 应明显更好。")


if __name__ == "__main__":
    main()
