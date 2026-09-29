"""OCR 引擎封装：统一 paddleocr / easyocr / stub 三个后端的接口。

两种用法:
  detect_and_read(image)  整图检测+识别。只在造伪标签时用（那时还没有 YOLO #2）。
  read_crop(crop)         只识别，不检测。有了 YOLO #2 之后走这条。

为什么要区分:
  论文 Table II —— 整图直接 OCR 的 CER 是 21.35%，先用 YOLO 框出文本区域
  再识别降到 6.33%，差三倍多。所以正式流程一定走 read_crop。
  detect_and_read 只是"先有鸡还是先有蛋"的破局手段：没有 YOLO #2 就没有框，
  没有框就造不出 YOLO #2 的训练标签，所以第一轮得靠 OCR 自带的检测器。

PaddleOCR 的 API 在 2.x / 3.x 之间变过好几次，这里做了多种返回结构的兼容，
并在 rec-only 不可用时退回"在裁剪图上跑完整 ocr 取最高置信度"。
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

# 原理图上的文字只有 8~12px 高，而 OCR 识别器一般期望 32~48px。
# 不放大直接喂，识别率会暴跌。这条论文没写，但不做肯定不行。
TARGET_TEXT_HEIGHT = 48
MAX_UPSCALE = 6.0
CROP_PAD = 3          # 紧贴框裁会切掉字符边缘，上下左右各留几像素


def prepare_crop(img: np.ndarray, box, pad: int = CROP_PAD,
                 target_h: int = TARGET_TEXT_HEIGHT):
    """从整图裁出文本块，留边并放大到识别器友好的高度。"""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - pad); y1 = max(0, int(y1) - pad)
    x2 = min(W, int(x2) + pad); y2 = min(H, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    h = crop.shape[0]
    if h <= 0:
        return None
    scale = min(MAX_UPSCALE, max(1.0, target_h / h))
    if scale > 1.01:
        im = Image.fromarray(crop).resize(
            (max(1, int(crop.shape[1] * scale)), max(1, int(h * scale))),
            Image.BICUBIC,
        )
        crop = np.asarray(im)
    return crop


class _Base:
    name = "base"

    def detect_and_read(self, image_path: str):
        raise NotImplementedError

    def read_crop(self, crop: np.ndarray):
        """返回 (text, conf)。"""
        raise NotImplementedError

    def read_crop_rot(self, crop: np.ndarray):
        """竖排处理：原图和旋转 90 度的版本都跑，取置信度高的那个。

        论文原文就是这么做的（"OCR is applied to both the original crop and
        its 90-degree-rotated version, and the higher-confidence result is selected"）。
        """
        a = self.read_crop(crop)
        if crop.shape[0] <= crop.shape[1]:
            return a                      # 明显是横排，不必再转
        b = self.read_crop(np.ascontiguousarray(np.rot90(crop)))
        return b if (b and b[1] > (a[1] if a else -1)) else a


class PaddleBackend(_Base):
    name = "paddleocr"

    def __init__(self, lang="ch", model_dir=None, use_gpu=False):
        from paddleocr import PaddleOCR

        self._PaddleOCR = PaddleOCR
        kw = {"lang": lang}
        if model_dir:
            det = os.path.join(model_dir, "det")
            rec = os.path.join(model_dir, "rec")
            if os.path.isdir(det):
                kw["det_model_dir"] = det
            if os.path.isdir(rec):
                kw["rec_model_dir"] = rec
        self.kw = kw
        self.full = self._make(kw)
        # rec-only 实例；部分版本不支持 det=False，失败就退回 full
        try:
            self.rec_only = self._make(dict(kw, det=False))
        except Exception:
            self.rec_only = None

    def _make(self, kw):
        try:
            return self._PaddleOCR(**kw)
        except TypeError:
            kw = {k: v for k, v in kw.items() if k != "use_angle_cls"}
            return self._PaddleOCR(**kw)

    @staticmethod
    def _iter_items(raw):
        """兼容 2.x 的 [[ [poly,(text,conf)], ... ]] 和 3.x 的 dict 结构。"""
        if raw is None:
            return
        if isinstance(raw, dict):
            texts = raw.get("rec_texts") or []
            scores = raw.get("rec_scores") or []
            polys = raw.get("rec_polys") or raw.get("dt_polys") or [None] * len(texts)
            for t, s, p in zip(texts, scores, polys):
                yield p, str(t), float(s)
            return
        for page in raw:
            if isinstance(page, dict):
                yield from PaddleBackend._iter_items(page)
                continue
            for item in page or []:
                try:
                    poly, (text, conf) = item[0], item[1]
                    yield poly, str(text), float(conf)
                except (TypeError, ValueError, IndexError):
                    continue

    def _run(self, obj, target):
        for meth in ("predict", "ocr"):
            fn = getattr(obj, meth, None)
            if fn is None:
                continue
            try:
                return fn(target)
            except TypeError:
                try:
                    return fn(target, cls=False)
                except Exception:
                    continue
        return None

    def detect_and_read(self, image_path: str):
        out = []
        for poly, text, conf in self._iter_items(self._run(self.full, image_path)):
            if poly is None:
                continue
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            out.append((min(xs), min(ys), max(xs), max(ys), text, conf))
        return out

    def read_crop(self, crop: np.ndarray):
        obj = self.rec_only or self.full
        best = None
        for _, text, conf in self._iter_items(self._run(obj, crop)):
            if best is None or conf > best[1]:
                best = (text, conf)
        return best


class EasyOCRBackend(_Base):
    name = "easyocr"

    def __init__(self, lang="ch", model_dir=None, use_gpu=False):
        import easyocr

        langs = ["ch_sim", "en"] if lang == "ch" else ["en"]
        self.reader = easyocr.Reader(langs, gpu=use_gpu,
                                     model_storage_directory=model_dir)

    def detect_and_read(self, image_path: str):
        out = []
        for poly, text, conf in self.reader.readtext(image_path):
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            out.append((min(xs), min(ys), max(xs), max(ys), str(text), float(conf)))
        return out

    def read_crop(self, crop: np.ndarray):
        res = self.reader.readtext(crop, detail=1)
        if not res:
            return None
        best = max(res, key=lambda r: r[2])
        return (str(best[1]), float(best[2]))


class StubBackend(_Base):
    """没装任何 OCR 时用来验证管道接线，不产生有意义的识别结果。

    它只按几何规则回一个占位串，所以**不能**用它造真实伪标签 ——
    白名单会把这些占位串全部过滤掉，结果是空的。
    """

    name = "stub"

    def __init__(self, lang="ch", model_dir=None, use_gpu=False):
        pass

    def detect_and_read(self, image_path: str):
        with Image.open(image_path) as im:
            W, H = im.size
        return [(x * W / 8, y * H / 8, x * W / 8 + 40, y * H / 8 + 12,
                 f"STUB{y}{x}", 0.9)
                for y in range(8) for x in range(8)]

    def read_crop(self, crop: np.ndarray):
        return (f"STUB{crop.shape[0]}x{crop.shape[1]}", 0.9)


BACKENDS = {"paddleocr": PaddleBackend, "easyocr": EasyOCRBackend, "stub": StubBackend}


def build(name: str, lang: str = "ch", model_dir: str | None = None,
          use_gpu: bool = False):
    if name not in BACKENDS:
        raise SystemExit(f"未知后端 {name}，可选: {', '.join(BACKENDS)}")
    try:
        return BACKENDS[name](lang=lang, model_dir=model_dir, use_gpu=use_gpu)
    except ImportError as err:
        hint = ("pip install paddlepaddle paddleocr" if name == "paddleocr"
                else f"pip install {name}")
        raise SystemExit(
            f"后端 {name} 不可用: {err}\n"
            f"  安装: {hint}\n"
            "  建议装 CPU 版（paddlepaddle 不带 -gpu），避免和 torch 的 CUDA 打架。\n"
            "  权重准备: python3 paddleocr_model/weights.py\n"
            "  只想验证管道接线可以用 --backend stub（不产生有意义的识别结果）"
        ) from err
