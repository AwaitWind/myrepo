"""训练 YOLO #2 文本检测器。

与 comp_det 的增强差异（都是有理由的，不要统一）:
  fliplr / flipud = 0   文字镜像翻转后不再是文字，会教模型学错形状先验
  degrees = 0           竖排标签靠数据本身覆盖，不靠旋转增强伪造
  hsv_h = 0             同 comp_det：jlc 图靠颜色区分导线和符号
  imgsz = 640           文本是细长行，窗口和输入都比元件检测小一档

用法:
    python3 yolo_model/text_det/train.py --data out/text_det/data.yaml --epochs 2 --device cpu
    python3 yolo_model/text_det/train.py --data out/text_det/data.yaml --epochs 120 \
        --model yolo11m.pt --device 0 --batch 16
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import weights  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/text_det/data.yaml")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--weights-dir", default=weights.DEFAULT_CACHE)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="out/yolo_runs")
    ap.add_argument("--name", default="text_det")
    ap.add_argument("--patience", type=int, default=30)
    a = ap.parse_args()

    if not os.path.exists(a.data):
        raise SystemExit(
            f"找不到 {a.data}。顺序是: "
            "pseudo_label.py（联网机器）-> export.py -> 本脚本"
        )

    model_path = weights.ensure(
        a.model, cache_dir=a.weights_dir, allow_download=not a.no_download
    )
    print(f"[text_det] 权重: {model_path}"
          + ("  (随机初始化)" if model_path.endswith(".yaml") else ""))

    from ultralytics import YOLO

    YOLO(model_path).train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        project=a.project,
        name=a.name,
        exist_ok=True,
        patience=a.patience,
        degrees=0.0,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.3,
        hsv_v=0.3,
        erasing=0.0,
        iou=0.3,
        plots=True,
    )
    print("\n[text_det] 训练结束。这个检测器的价值参照论文 Table II: "
          "整图直接 OCR 的 CER 是 21.35%，先用 YOLO 框出文本区域再识别降到 6.33%。"
          "所以评价它要看下游 OCR 的 CER 改善，而不只看框的 mAP。")


if __name__ == "__main__":
    main()
