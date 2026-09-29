"""训练 YOLO #1 元件检测器。

CPU / GPU 同一份代码，靠 --device 区分。权重缺失时自动下载（断点续传），
也可以先跑 `python3 yolo_model/common/weights.py yolo11m.pt` 预下载到 weights/。

增强参数与 COCO 默认不同的几处及理由（依据 §M1a 与实测）:
  mosaic=0      拼接造出不存在的版式，且破坏小目标尺度一致性
  hsv_h=0       jlc 图靠颜色区分导线和符号，实测主绘图色 RGB(207,127,127)，动色相会毁掉这个线索
  degrees=2     原理图基本正交，大角度旋转不符合版式先验
  shear/persp=0 会畸变长宽比，破坏 8x20 这类细长符号的形状先验
  iou=0.3       小目标重叠少，NMS 阈值低于通用 0.5

用法:
    python3 yolo_model/comp_det/train.py --data out/comp_det/data.yaml --epochs 2 --device cpu
    python3 yolo_model/comp_det/train.py --data out/comp_det/data.yaml --epochs 150 \
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
    ap.add_argument("--data", default="out/comp_det/data.yaml")
    ap.add_argument("--model", default="yolo11m.pt",
                    help="权重名（自动下载）、本地 .pt 路径，或 .yaml 表示随机初始化")
    ap.add_argument("--weights-dir", default=weights.DEFAULT_CACHE)
    ap.add_argument("--no-download", action="store_true",
                    help="禁止联网，权重必须已在本地")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="out/yolo_runs")
    ap.add_argument("--name", default="comp_det")
    ap.add_argument("--patience", type=int, default=30)
    a = ap.parse_args()

    if not os.path.exists(a.data):
        raise SystemExit(f"找不到 {a.data}，先跑 yolo_model/comp_det/export.py")

    model_path = weights.ensure(
        a.model, cache_dir=a.weights_dir, allow_download=not a.no_download
    )
    print(f"[comp_det] 权重: {model_path}"
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
        degrees=2.0,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.3,
        hsv_v=0.3,
        erasing=0.0,
        iou=0.3,
        plots=True,
    )
    print("\n[comp_det] 训练结束。看召回率(R)而不是 mAP —— "
          "20px 判正阈值很宽松，定位精度不是瓶颈，漏检才是硬损失；"
          "置信度阈值要按 F1 最优点重调，不要用默认 0.25。")


if __name__ == "__main__":
    main()
