"""对照实验：渲染线宽对导线分割的影响。

为什么需要一个固定的评测基准:
  不同线宽训出来的模型，如果各自对着自己的 mask 评 IoU，那是循环论证 ——
  线宽 5 的模型对线宽 5 的 mask 当然吻合，但它抠出来的线比真实导线粗一圈。
  所以两个臂统一对**线宽 1 的 mask**评，那才是真实导线位置
  （实测线宽 1 时 mask 命中前景 98.8~100%，即 GT 端点与图上导线像素基本完全吻合）。

两个与标签定义无关的指标:
  recall  真实导线像素里，有多少落在预测的 1px 邻域内 —— 漏没漏线
  spill   预测像素里，有多少离任何真实导线都超过 2px —— 有没有糊到非导线区域

  只看 recall 会奖励"全判正"，只看 spill 会奖励"什么都不预测"，必须一起看。
  按"宁可过分割不要断线"的原则，recall 的权重高于 spill。

用法:
    python3 u-net_mode/exp_linewidth.py --widths 1 3 --epochs 6
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dataset as ds_mod  # noqa: E402
import losses as loss_mod  # noqa: E402
import model as model_mod  # noqa: E402
import render_mask as rm  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "yolo_model"))
from common import cases as cases_mod  # noqa: E402

from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def build_masks(cases, width: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for c in cases:
        target = cases_mod.load_target(c.json_path)
        with Image.open(c.image_path) as im:
            W, H = im.size
        segs, st = rm.edge_segments(target, H)
        if st["edges_ok"] < 10 or (st["nets_empty_edges"] / max(st["nets_total"], 1)) > 0.05:
            continue
        mp = os.path.join(out_dir, c.case_id + "_mask.png")
        if not os.path.exists(mp):
            rm.render(segs, (W, H), width).save(mp)
        rows.append({"case_id": c.case_id, "image": c.image_path, "json": c.json_path,
                     "source": c.source, "dataset": c.dataset, "group": c.group,
                     "mask": mp, "usable": True})
    return rows


@torch.no_grad()
def evaluate(net, rows_eval, ref_dir, window, thr, max_tiles):
    """对线宽 1 的真值评 recall / spill。ref_dir 是线宽 1 的 mask 目录。"""
    ref_rows = [dict(r, mask=os.path.join(ref_dir, r["case_id"] + "_mask.png"))
                for r in rows_eval]
    ds = ds_mod.WireTiles(ref_rows, window, 0.2, min_pos=32, augment=False)
    net.eval()
    tp_r = tot_r = sp = tot_p = 0
    n = min(len(ds), max_tiles)
    for i in range(n):
        x, y = ds[i]
        pred = (torch.sigmoid(net(x.unsqueeze(0)))[0, 0] > thr).numpy()
        true = y[0].numpy() > 0.5
        if not true.any():
            continue
        pred_d = ndimage.binary_dilation(pred, iterations=1)
        true_d = ndimage.binary_dilation(true, iterations=2)
        tp_r += (pred_d & true).sum(); tot_r += true.sum()
        sp += (pred & ~true_d).sum(); tot_p += pred.sum()
    return {
        "recall": tp_r / tot_r if tot_r else 0.0,
        "spill": sp / tot_p if tot_p else 0.0,
        "tiles": n,
    }


def run_arm(width, cases, ref_dir, a):
    mdir = os.path.join(a.work, f"w{width}")
    rows = build_masks(cases, width, mdir)
    tr_rows, va_rows = ds_mod.split_rows(rows, a.val_ratio, a.seed)
    tr = ds_mod.WireTiles(tr_rows, a.window, 0.2, a.min_pos, augment=True, seed=a.seed)
    if a.max_train_tiles and len(tr.index) > a.max_train_tiles:
        rng = np.random.default_rng(a.seed)
        tr.index = [tr.index[i] for i in
                    rng.choice(len(tr.index), a.max_train_tiles, replace=False)]
    pos = ds_mod.positive_ratio(tr)
    pw = min(50.0, (1 - pos) / pos) if pos > 0 else 10.0

    torch.manual_seed(a.seed)
    net = model_mod.build("unet", base=a.base)
    crit = loss_mod.build("tversky", pw, a.beta)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=0)

    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        tot = 0.0
        for x, y in dl:
            loss = crit(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"  [w{width}] ep{ep}/{a.epochs} loss {tot/max(len(dl),1):.4f}"
              f"  ({time.time()-t0:.0f}s)")
    m = evaluate(net, va_rows, ref_dir, a.window, a.thr, a.eval_tiles)
    m.update({"width": width, "pos_ratio": pos, "pos_weight": pw,
              "train_tiles": len(tr), "val_cases": len(va_rows),
              "secs": round(time.time() - t0)})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", default=None, help="不传=自动探测")
    ap.add_argument("--harvest", default="", help="默认只用官方数据做线宽对照")
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--work", default="out/exp_linewidth")
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--min-pos", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--max-train-tiles", type=int, default=600)
    ap.add_argument("--eval-tiles", type=int, default=80)
    ap.add_argument("--limit-cases", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hash-cache", default="out/.image_hash_cache.json")
    a = ap.parse_args()

    roots = cases_mod.roots_from_args(a.official, a.harvest)
    cases, _, _ = cases_mod.discover(roots, hash_cache=a.hash_cache)
    if a.limit_cases:
        cases = cases[: a.limit_cases]
    print(f"参与实验的样本 {len(cases)}")

    ref_dir = os.path.join(a.work, "w1")
    build_masks(cases, 1, ref_dir)   # 评测基准，无论训练用什么线宽

    res = []
    for w in a.widths:
        print(f"\n=== 线宽 {w} ===")
        res.append(run_arm(w, cases, ref_dir, a))

    print(f"\n{'线宽':>4}{'正样本占比':>11}{'训练切片':>9}{'recall':>9}{'spill':>8}{'耗时':>7}")
    for m in res:
        print(f"{m['width']:>4}{m['pos_ratio']*100:>10.3f}%{m['train_tiles']:>9}"
              f"{m['recall']*100:>8.1f}%{m['spill']*100:>7.1f}%{m['secs']:>6}s")
    print("\n统一对线宽 1 的 mask 评测。recall 高=没漏线，spill 低=没糊到非导线区域。"
          "按'宁可过分割不要断线'，recall 优先。")
    print("注意: 这是小规模试点（base=16、少量切片、少量 epoch），"
          "用来定线宽这个超参的方向，不是最终性能。")


if __name__ == "__main__":
    main()
