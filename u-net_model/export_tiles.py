"""把原图和 mask 预切片落盘，供 U-Net 训练直接读小图。

为什么必须预切片（这是实测踩出来的）:
  原先 dataset.py 做在线切片，只缓存 6 张原图。40 个样本时没问题，但 286 个
  样本配 shuffle=True 时，一个 batch 的 8 个切片来自 8 张不同原图，几乎每次
  缓存未命中，每一步都要重新解码多张大图。原图中位数 1MP、最大 17MP、
  合计 102 MB，于是训练退化成纯 I/O 瓶颈 —— 实测 97 秒连第一个 epoch 的
  日志都出不来。GPU 也救不了这个。

  两个 YOLO 的 export.py 本来就是预切片落盘的，这里保持一致。

用法:
    python3 u-net_mode/render_mask.py --out out/wire_mask
    python3 u-net_mode/export_tiles.py --mask-dir out/wire_mask --out out/wire_tiles
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dataset as ds_mod  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "yolo_model"))
from common import tiling  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def export(a):
    rows = ds_mod.load_usable(a.mask_dir, a.audit)
    tr_rows, va_rows = ds_mod.split_rows(rows, a.val_ratio, a.seed)
    side = {r["case_id"]: "train" for r in tr_rows}
    side.update({r["case_id"]: "val" for r in va_rows})
    print(f"[tiles] 样本 train/val {len(tr_rows)}/{len(va_rows)}")

    for split in ("train", "val"):
        for sub in ("images", "masks"):
            os.makedirs(os.path.join(a.out, sub, split), exist_ok=True)

    stride = tiling.stride_of(a.window, a.overlap)
    st = Counter()
    pos_sum = 0.0
    index = []
    for i, r in enumerate(rows, 1):
        split = side[r["case_id"]]
        with Image.open(r["image"]) as im:
            img = np.asarray(im.convert("RGB"))
        with Image.open(r["mask"]) as m:
            msk = np.asarray(m.convert("L"))
        if img.shape[:2] != msk.shape[:2]:
            st["shape_mismatch"] += 1
            continue
        H, W = msk.shape
        w = a.window
        for oy in tiling.tile_origins(H, w, stride):
            for ox in tiling.tile_origins(W, w, stride):
                lab = msk[oy:oy + w, ox:ox + w]
                n_pos = int((lab > 127).sum())
                if n_pos < a.min_pos:
                    st["skipped_low_pos"] += 1
                    continue
                tile = img[oy:oy + w, ox:ox + w]
                if tile.shape[0] != w or tile.shape[1] != w:
                    pad = np.full((w, w, 3), 255, dtype=tile.dtype)
                    pad[: tile.shape[0], : tile.shape[1]] = tile
                    tile = pad
                    pl = np.zeros((w, w), dtype=lab.dtype)
                    pl[: lab.shape[0], : lab.shape[1]] = lab
                    lab = pl
                name = f"{r['case_id']}_x{ox}_y{oy}.png"
                Image.fromarray(tile).save(os.path.join(a.out, "images", split, name))
                Image.fromarray((lab > 127).astype(np.uint8) * 255).save(
                    os.path.join(a.out, "masks", split, name)
                )
                st[f"tiles_{split}"] += 1
                pos_sum += n_pos / (w * w)
                index.append({"name": name, "split": split,
                              "case_id": r["case_id"], "source": r.get("source", "?"),
                              "dataset": r.get("dataset", "?"), "pos": n_pos})
        if i % 50 == 0 or i == len(rows):
            print(f"[tiles] {i}/{len(rows)} 样本  切片 "
                  f"{st['tiles_train']}+{st['tiles_val']}")

    n = st["tiles_train"] + st["tiles_val"]
    meta = {"args": vars(a), "stats": dict(st),
            "pos_ratio": pos_sum / n if n else 0.0, "tiles": index}
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)

    print(f"\n[tiles] 切片 train/val  {st['tiles_train']} / {st['tiles_val']}")
    print(f"[tiles] 跳过前景不足     {st['skipped_low_pos']}")
    if st["shape_mismatch"]:
        print(f"[tiles] 尺寸不匹配剔除   {st['shape_mismatch']}")
    print(f"[tiles] 正样本像素占比   {meta['pos_ratio']*100:.3f}%")
    print(f"[tiles] 输出             {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-dir", default="out/wire_mask")
    ap.add_argument("--audit", default=None, help="默认取 <mask-dir>/mask_audit.json")
    ap.add_argument("--out", default="out/wire_tiles")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-pos", type=int, default=64,
                    help="切片内导线像素数下限。原理图大片空白，不过滤会把正样本稀释到学不动")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    export(ap.parse_args())


if __name__ == "__main__":
    main()
