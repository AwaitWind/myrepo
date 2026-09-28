"""YOLO #2 文本检测：伪标签 -> YOLO 切片数据集。

输入是 text_det/pseudo_label.py 的产出目录。伪标签的 bbox 已经是图像坐标
（OCR 直接在图上跑的），所以**这里不做 y 翻转** —— 这是与 comp_det 的关键区别。

切片参数与 comp_det 有意不同:
  文本是细长的行（高度可能只有 8~12px，宽度几十像素），不是 8~60px 的方形符号。
  默认窗口小一档（640），让同样大小的文字在输入里占更多特征格点。

用法:
    python3 yolo_model/text_det/export.py --pseudo out/text_pseudo --out out/text_det
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import cases as cases_mod  # noqa: E402
from common import tiling  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def load_pseudo(pseudo_dir: str, min_conf: float, allow_relaxed: bool):
    files = sorted(
        p for p in glob.glob(os.path.join(pseudo_dir, "*.json"))
        if os.path.basename(p) != "summary.json"
    )
    if not files:
        raise SystemExit(
            f"{pseudo_dir} 里没有伪标签。先在能联网的机器上跑 "
            "yolo_model/text_det/pseudo_label.py"
        )
    recs = []
    skipped = Counter()
    for path in files:
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if not os.path.exists(rec.get("image", "")):
            skipped["image_missing"] += 1
            continue
        boxes = []
        for b in rec.get("boxes", []):
            if b.get("conf", 1.0) < min_conf:
                skipped["low_conf"] += 1
                continue
            if b.get("match") == "relaxed" and not allow_relaxed:
                skipped["relaxed_excluded"] += 1
                continue
            x1, y1, x2, y2 = b["bbox"]
            if x2 <= x1 or y2 <= y1:
                skipped["degenerate"] += 1
                continue
            boxes.append((float(x1), float(y1), float(x2), float(y2)))
        if not boxes:
            skipped["no_boxes"] += 1
            continue
        rec["_boxes"] = boxes
        recs.append(rec)
    return recs, skipped


def export(args):
    recs, skipped = load_pseudo(args.pseudo, args.min_conf, not args.exclude_relaxed)

    # 复用 cases 的分组划分逻辑，保证与 comp_det 的 split 口径一致
    stub = [
        cases_mod.Case(
            case_id=r["case_id"], image_path=r["image"], json_path=r.get("json", ""),
            source=r.get("source", "unknown"), dataset=r.get("dataset", "unknown"),
            group=r.get("group") or r["case_id"],
        )
        for r in recs
    ]
    train, val = cases_mod.split(stub, n_val=args.n_val, val_ratio=args.val_ratio, seed=args.seed)
    side = {c.case_id: "train" for c in train}
    side.update({c.case_id: "val" for c in val})

    window, stride = args.window, tiling.stride_of(args.window, args.overlap)
    for split_name in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.out, sub, split_name), exist_ok=True)

    stats = Counter()
    per_split = Counter()
    for rec in recs:
        split_name = side[rec["case_id"]]
        boxes = rec["_boxes"]
        stats["pseudo_boxes"] += len(boxes)
        with Image.open(rec["image"]) as raw:
            im = raw.convert("RGB")
            W, H = im.size
            for oy in tiling.tile_origins(H, window, stride):
                for ox in tiling.tile_origins(W, window, stride):
                    labels, n_drop = tiling.clip_to_tile(
                        boxes, ox, oy, window, keep_ratio=args.keep_ratio
                    )
                    stats["dropped_truncated"] += n_drop
                    if not labels:
                        stats["tiles_empty_skipped"] += 1
                        continue
                    lines = [tiling.format_label(0, cx, cy, w, h)
                             for cx, cy, w, h, _ in labels]
                    name = f"{rec['case_id']}_x{ox}_y{oy}"
                    im.crop((ox, oy, ox + window, oy + window)).save(
                        os.path.join(args.out, "images", split_name, name + ".png")
                    )
                    with open(os.path.join(args.out, "labels", split_name, name + ".txt"),
                              "w", encoding="utf-8") as fh:
                        fh.write("\n".join(lines) + "\n")
                    stats[f"tiles_{split_name}"] += 1
                    per_split[split_name] += len(lines)

    yaml_path = tiling.write_data_yaml(args.out, {0: "text"})
    with open(os.path.join(args.out, "export_report.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "args": vars(args),
            "skipped": dict(skipped),
            "split": {"train_cases": len(train), "val_cases": len(val)},
            "tiles": {"train": stats["tiles_train"], "val": stats["tiles_val"]},
            "labels": {"train": per_split["train"], "val": per_split["val"]},
            "pseudo_boxes": stats["pseudo_boxes"],
            "dropped_truncated": stats["dropped_truncated"],
            "tiles_empty_skipped": stats["tiles_empty_skipped"],
        }, fh, ensure_ascii=False, indent=2)

    print(f"\n[text_det] 输出 {args.out}  window={window} stride={stride}")
    print(f"[text_det] 样本 train/val      {len(train)} / {len(val)}")
    print(f"[text_det] 切片 train/val      {stats['tiles_train']} / {stats['tiles_val']}")
    print(f"[text_det] 切片标签数 train/val {per_split['train']} / {per_split['val']}"
          "  (含重叠切片的重复计数)")
    print(f"[text_det] 伪标签框总数         {stats['pseudo_boxes']}")
    print(f"[text_det] 跳过统计             {dict(skipped)}")
    print(f"[text_det] data.yaml            {yaml_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo", default="out/text_pseudo")
    ap.add_argument("--out", default="out/text_det")
    ap.add_argument("--window", type=int, default=640,
                    help="文本是细长行，窗口比元件检测小一档，让文字占更多特征格点")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--keep-ratio", type=float, default=0.5)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--exclude-relaxed", action="store_true",
                    help="只用严格命中 GT 的框，丢掉归一化后才命中的")
    ap.add_argument("--n-val", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    export(ap.parse_args())


if __name__ == "__main__":
    main()
