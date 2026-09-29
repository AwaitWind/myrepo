"""YOLO #1 元件检测：GT bbox -> YOLO 切片数据集。

为什么单类（类别无关）:
  类别口径目前是烂账 —— data/ 19 类、官方 200_train 41 类、10_GTcase 20 类，
  而且直接冲突（200_train 用 `mosfet`/`bjt`，另两个用 `mosfet_npn`/`bjt_npn`）。
  单类检测把这摊事整个绕开，只判"这里有没有元件"，分类留给下一步。
  这也正好是 data/ 唯一干净的部分：bbox 就是 bbox，不受匿名元件 pin 口径、
  空 edges、类别词表任何一个问题影响。
  需要多类时传 --classes multi，会读 GT 的 type 原值建词表（词表统一前不建议用）。

用法:
    python3 yolo_model/comp_det/export.py --out out/comp_det
    python3 yolo_model/comp_det/export.py --out out/comp_det_official --harvest ""
    python3 yolo_model/comp_det/export.py --out out/comp_det_s2 --scale 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import cases as cases_mod  # noqa: E402
from common import tiling  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def build_class_map(cases, mode: str):
    if mode == "single":
        return {"": 0}, {0: "component"}
    counter = Counter()
    for c in cases:
        target = cases_mod.load_target(c.json_path)
        for comp in (target.get("components") or {}).values():
            counter[(comp.get("type") or "unknown")] += 1
    names = sorted(counter)
    return {t: i for i, t in enumerate(names)}, {i: t for i, t in enumerate(names)}


def export(args):
    roots = cases_mod.roots_from_args(args.official, args.harvest)
    all_cases, disc_stats, dropped = cases_mod.discover(
        roots, hash_cache=args.hash_cache, min_components=args.min_components
    )
    if not all_cases:
        raise SystemExit("没有可用样本")

    train_cases, val_cases = cases_mod.split(
        all_cases, n_val=args.n_val, val_ratio=args.val_ratio, seed=args.seed
    )
    side = {c.case_id: "train" for c in train_cases}
    side.update({c.case_id: "val" for c in val_cases})

    cls_map, cls_names = build_class_map(all_cases, args.classes)
    window, stride = args.window, tiling.stride_of(args.window, args.overlap)

    for split_name in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.out, sub, split_name), exist_ok=True)

    stats = Counter()
    per_split_inst = Counter()
    manifest = []
    n_cases = len(all_cases)

    for idx, case in enumerate(all_cases, 1):
        split_name = side[case.case_id]
        target = cases_mod.load_target(case.json_path)
        with Image.open(case.image_path) as raw:
            im = raw.convert("RGB")
            W0, H0 = im.size
            # y 翻转在 image_boxes 里完成，scale 一并应用
            boxes = cases_mod.image_boxes(target, H0, scale=args.scale)
            stats["gt_instances"] += len(boxes)
            if args.scale != 1:
                im = im.resize((int(W0 * args.scale), int(H0 * args.scale)), Image.BICUBIC)
            W, H = im.size

            n_tiles = 0
            for oy in tiling.tile_origins(H, window, stride):
                for ox in tiling.tile_origins(W, window, stride):
                    labels, n_drop = tiling.clip_to_tile(
                        boxes, ox, oy, window, keep_ratio=args.keep_ratio
                    )
                    stats["dropped_truncated"] += n_drop
                    if not labels:
                        stats["tiles_empty_skipped"] += 1
                        continue
                    lines = []
                    for cx, cy, w, h, rest in labels:
                        ctype = rest[1] if len(rest) > 1 else ""
                        cid = 0 if args.classes == "single" else cls_map.get(ctype, 0)
                        lines.append(tiling.format_label(cid, cx, cy, w, h))
                    name = f"{case.case_id}_x{ox}_y{oy}"
                    im.crop((ox, oy, ox + window, oy + window)).save(
                        os.path.join(args.out, "images", split_name, name + ".png")
                    )
                    with open(
                        os.path.join(args.out, "labels", split_name, name + ".txt"),
                        "w", encoding="utf-8",
                    ) as fh:
                        fh.write("\n".join(lines) + "\n")
                    stats[f"tiles_{split_name}"] += 1
                    per_split_inst[split_name] += len(lines)
                    n_tiles += 1

        if idx % 25 == 0 or idx == n_cases:
            done = stats["tiles_train"] + stats["tiles_val"]
            print(f"[comp_det] {idx}/{n_cases} 样本  已写切片 {done}", flush=True)

        manifest.append({
            "case_id": case.case_id,
            "split": split_name,
            "source": case.source,
            "dataset": case.dataset,
            "group": case.group,
            "image": case.image_path,
            "json": case.json_path,
            "size": [W0, H0],
            "gt_components": case.n_components,
            "tiles": n_tiles,
        })

    yaml_path = tiling.write_data_yaml(args.out, cls_names)
    report = {
        "args": vars(args),
        "discover": disc_stats,
        "dropped": dropped,
        "split": {"train_cases": len(train_cases), "val_cases": len(val_cases)},
        "tiles": {"train": stats["tiles_train"], "val": stats["tiles_val"]},
        "instances": {"train": per_split_inst["train"], "val": per_split_inst["val"]},
        "gt_instances": stats["gt_instances"],
        "dropped_truncated": stats["dropped_truncated"],
        "tiles_empty_skipped": stats["tiles_empty_skipped"],
        "classes": cls_names,
        "cases": manifest,
    }
    with open(os.path.join(args.out, "export_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"\n[comp_det] 输出 {args.out}  window={window} stride={stride} scale={args.scale}")
    print(f"[comp_det] 样本 train/val      {len(train_cases)} / {len(val_cases)}")
    print(f"[comp_det] 切片 train/val      {stats['tiles_train']} / {stats['tiles_val']}")
    print(f"[comp_det] 切片标签数 train/val {per_split_inst['train']} / {per_split_inst['val']}"
          "  (含重叠切片的重复计数)")
    print(f"[comp_det] GT 实例总数          {stats['gt_instances']}")
    print(f"[comp_det] 丢弃截断>{int(args.keep_ratio*100)}% 实例   {stats['dropped_truncated']}")
    print(f"[comp_det] 跳过空白切片          {stats['tiles_empty_skipped']}")
    print(f"[comp_det] 类别数                {len(cls_names)}")
    print(f"[comp_det] data.yaml             {yaml_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/comp_det")
    ap.add_argument("--official", default=None,
                    help="官方样本根目录。不传=自动探测，传空字符串=不用该数据集")
    ap.add_argument("--harvest", default=None,
                    help="自采样本根目录。不传=自动探测，传空字符串=不用该数据集")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="导出前整图放大倍数；2 对应 §M1a 的 2x 上采样对照（切片数约翻 4 倍）")
    ap.add_argument("--keep-ratio", type=float, default=0.5,
                    help="实例落在切片内的面积占比下限")
    ap.add_argument("--classes", choices=["single", "multi"], default="single")
    ap.add_argument("--n-val", type=int, default=0, help="验证集样本数；给 0 则用 --val-ratio")
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="对齐论文的 design-level 8:1:1，取 0.1")
    ap.add_argument("--min-components", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hash-cache", default="out/.image_hash_cache.json")
    export(ap.parse_args())


if __name__ == "__main__":
    main()
