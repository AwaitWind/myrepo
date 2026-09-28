"""导线分割数据集：原图 + 渲染 mask，按切片取样。

输入是 render_mask.py 的产出（mask PNG + mask_audit.json）。
切片按 audit 里 usable=true 的样本取，复用 yolo_model/common 的分组划分，
保证与检测那边的 split 口径一致（内容重复图不跨 split）。

预处理关键点 —— 判前景必须用"非白"判据:
  jlc 来源的图是彩色的，实测主绘图色 RGB(207,127,127) 灰度 151、
  RGB(127,195,127) 灰度 167，比灰度 128 更亮。任何"二值化取暗像素"的
  预处理在 jlc 上会完全失效，而 jlc 占官方训练集 99/200。
  所以这里保留 RGB 三通道输入，不做灰度化，也不做基于灰度阈值的二值化。
"""

from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "yolo_model"))
from common import cases as cases_mod  # noqa: E402
from common import tiling  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_usable(mask_dir: str, audit_path: str | None = None):
    audit_path = audit_path or os.path.join(mask_dir, "mask_audit.json")
    if not os.path.exists(audit_path):
        raise SystemExit(f"找不到 {audit_path}，先跑 u-net_mode/render_mask.py")
    with open(audit_path, encoding="utf-8") as fh:
        rows = json.load(fh)["cases"]
    out = []
    for r in rows:
        if not r.get("usable"):
            continue
        mp = os.path.join(mask_dir, r["case_id"] + "_mask.png")
        if not os.path.exists(mp) or not os.path.exists(r["image"]):
            continue
        r["mask"] = mp
        out.append(r)
    if not out:
        raise SystemExit(f"{mask_dir} 里没有可用样本")
    return out


def split_rows(rows, val_ratio: float, seed: int):
    stub = [
        cases_mod.Case(case_id=r["case_id"], image_path=r["image"], json_path=r["json"],
                       source=r.get("source", "?"), dataset=r.get("dataset", "?"),
                       group=r.get("group") or r["case_id"])
        for r in rows
    ]
    train, val = cases_mod.split(stub, val_ratio=val_ratio, seed=seed)
    tr_ids = {c.case_id for c in train}
    return ([r for r in rows if r["case_id"] in tr_ids],
            [r for r in rows if r["case_id"] not in tr_ids])


def build_index(rows, window: int, overlap: float, min_pos: int):
    """枚举所有切片位置。丢弃 mask 里前景像素过少的切片。

    min_pos 存在的理由：原理图大片空白，不过滤的话绝大多数切片是全零标签，
    正样本被稀释到学不动。
    """
    stride = tiling.stride_of(window, overlap)
    index = []
    for ri, r in enumerate(rows):
        with Image.open(r["mask"]) as m:
            mask = np.asarray(m.convert("L"))
        H, W = mask.shape
        for oy in tiling.tile_origins(H, window, stride):
            for ox in tiling.tile_origins(W, window, stride):
                if (mask[oy:oy + window, ox:ox + window] > 127).sum() >= min_pos:
                    index.append((ri, ox, oy))
    return index


class WireTiles(Dataset):
    def __init__(self, rows, window: int = 512, overlap: float = 0.2,
                 min_pos: int = 64, augment: bool = False, seed: int = 0):
        self.rows = rows
        self.window = window
        self.augment = augment
        self.index = build_index(rows, window, overlap, min_pos)
        self.rng = random.Random(seed)
        self._cache: dict[int, tuple] = {}

    def __len__(self):
        return len(self.index)

    def _load(self, ri: int):
        if ri not in self._cache:
            if len(self._cache) > 6:          # 原图很大，只缓存少量
                self._cache.pop(next(iter(self._cache)))
            r = self.rows[ri]
            with Image.open(r["image"]) as im:
                img = np.asarray(im.convert("RGB"))
            with Image.open(r["mask"]) as m:
                msk = np.asarray(m.convert("L"))
            self._cache[ri] = (img, msk)
        return self._cache[ri]

    def __getitem__(self, i):
        ri, ox, oy = self.index[i]
        img, msk = self._load(ri)
        w = self.window
        tile = img[oy:oy + w, ox:ox + w]
        lab = msk[oy:oy + w, ox:ox + w]
        if tile.shape[0] != w or tile.shape[1] != w:   # 贴边不足时补白
            pad = np.full((w, w, 3), 255, dtype=tile.dtype)
            pad[: tile.shape[0], : tile.shape[1]] = tile
            tile = pad
            pl = np.zeros((w, w), dtype=lab.dtype)
            pl[: lab.shape[0], : lab.shape[1]] = lab
            lab = pl
        if self.augment:
            k = self.rng.randrange(4)
            if k:
                tile = np.rot90(tile, k)
                lab = np.rot90(lab, k)
            if self.rng.random() < 0.5:
                tile = tile[:, ::-1]
                lab = lab[:, ::-1]
        x = np.ascontiguousarray(tile).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        y = (np.ascontiguousarray(lab) > 127).astype(np.float32)
        return (torch.from_numpy(x).permute(2, 0, 1),
                torch.from_numpy(y).unsqueeze(0))


class TileFiles(Dataset):
    """读 export_tiles.py 预切好的小图。训练走这条路，不要用在线切片的 WireTiles。

    在线切片在样本多时会因为反复解码大原图退化成 I/O 瓶颈（实测 286 个样本时
    一个 epoch 都跑不出来），预切片之后每次只读一张 window×window 的小图。
    """

    def __init__(self, tiles_dir: str, split: str, augment: bool = False, seed: int = 0):
        self.img_dir = os.path.join(tiles_dir, "images", split)
        self.msk_dir = os.path.join(tiles_dir, "masks", split)
        if not os.path.isdir(self.img_dir):
            raise SystemExit(f"找不到 {self.img_dir}，先跑 u-net_mode/export_tiles.py")
        self.names = sorted(os.listdir(self.img_dir))
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        with Image.open(os.path.join(self.img_dir, name)) as im:
            tile = np.asarray(im.convert("RGB"))
        with Image.open(os.path.join(self.msk_dir, name)) as m:
            lab = np.asarray(m.convert("L"))
        if self.augment:
            k = self.rng.randrange(4)
            if k:
                tile, lab = np.rot90(tile, k), np.rot90(lab, k)
            if self.rng.random() < 0.5:
                tile, lab = tile[:, ::-1], lab[:, ::-1]
        x = np.ascontiguousarray(tile).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        y = (np.ascontiguousarray(lab) > 127).astype(np.float32)
        return (torch.from_numpy(x).permute(2, 0, 1),
                torch.from_numpy(y).unsqueeze(0))


def tiles_pos_ratio(tiles_dir: str, default: float = 0.01) -> float:
    """从 export_tiles.py 写的 index.json 读实测正样本占比，省得再扫一遍。"""
    p = os.path.join(tiles_dir, "index.json")
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as fh:
        return json.load(fh).get("pos_ratio", default) or default


def positive_ratio(ds: WireTiles, n: int = 20) -> float:
    """抽样估计正样本像素占比，用来定 pos_weight。"""
    if not len(ds):
        return 0.0
    step = max(1, len(ds) // n)
    vals = [ds[i][1].mean().item() for i in range(0, len(ds), step)][:n]
    return float(np.mean(vals)) if vals else 0.0
