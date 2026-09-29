"""训练 U-Net 导线分割。

用法:
    python3 u-net_mode/render_mask.py --out out/wire_mask
    python3 u-net_mode/train.py --mask-dir out/wire_mask --epochs 2 --device cpu --limit-steps 4
    python3 u-net_mode/train.py --mask-dir out/wire_mask --epochs 80 --device 0 \
        --arch resnet18_unet --batch 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as ds_mod  # noqa: E402
import losses as loss_mod  # noqa: E402
import model as model_mod  # noqa: E402


def run_epoch(net, loader, crit, opt, device, thr, limit=0):
    train = opt is not None
    net.train(train)
    agg = {"loss": 0.0, "iou": 0.0, "recall": 0.0, "precision": 0.0}
    n = 0
    for step, (x, y) in enumerate(loader):
        if limit and step >= limit:
            break
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            out = net(x)
            loss = crit(out, y)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        m = loss_mod.seg_metrics(out, y, thr)
        agg["loss"] += loss.item()
        for k in ("iou", "recall", "precision"):
            agg[k] += m[k]
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-dir", default="out/wire_mask")
    ap.add_argument("--tiles-dir", default=None,
                    help="export_tiles.py 的预切片目录。**推荐用这个** —— "
                         "在线切片在样本多时会因反复解码大原图退化成 I/O 瓶颈")
    ap.add_argument("--arch", default="unet", choices=["unet", "resnet18_unet"])
    ap.add_argument("--base", type=int, default=64, help="原版 U-Net 的基础通道数")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="resnet18_unet 不加载 ImageNet 权重（不联网）")
    ap.add_argument("--loss", default="tversky", choices=["dice_bce", "tversky"])
    ap.add_argument("--beta", type=float, default=0.7, help="Tversky 的 beta，>0.5 偏召回")
    ap.add_argument("--pos-weight", type=float, default=0.0,
                    help="0 表示按实测正样本占比自动估计")
    ap.add_argument("--thr", type=float, default=0.4, help="二值化阈值，偏召回所以低于 0.5")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-pos", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/unet_runs/wire")
    ap.add_argument("--limit-steps", type=int, default=0, help="每轮最多几步，冒烟用")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    if a.tiles_dir:
        tr = ds_mod.TileFiles(a.tiles_dir, "train", augment=True, seed=a.seed)
        va = ds_mod.TileFiles(a.tiles_dir, "val", augment=False, seed=a.seed)
        pos = ds_mod.tiles_pos_ratio(a.tiles_dir)
        print(f"[unet] 预切片 {a.tiles_dir}  切片 train/val {len(tr)}/{len(va)}")
    else:
        rows = ds_mod.load_usable(a.mask_dir)
        tr_rows, va_rows = ds_mod.split_rows(rows, a.val_ratio, a.seed)
        tr = ds_mod.WireTiles(tr_rows, a.window, a.overlap, a.min_pos,
                              augment=True, seed=a.seed)
        va = ds_mod.WireTiles(va_rows, a.window, a.overlap, a.min_pos,
                              augment=False, seed=a.seed)
        pos = ds_mod.positive_ratio(tr)
        print(f"[unet] 在线切片（样本多时会 I/O 瓶颈，建议改用 --tiles-dir）  "
              f"样本 {len(tr_rows)}/{len(va_rows)}  切片 {len(tr)}/{len(va)}")

    pw = a.pos_weight or (min(50.0, (1 - pos) / pos) if pos > 0 else 10.0)
    print(f"[unet] 正样本像素占比 {pos*100:.3f}%  ->  pos_weight {pw:.1f}")

    device = torch.device("cpu" if a.device in ("cpu", "") else f"cuda:{a.device}")
    net = model_mod.build(a.arch, base=a.base, pretrained=not a.no_pretrained).to(device)
    n_par = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"[unet] 架构 {a.arch}  参数 {n_par:.1f}M  设备 {device}")

    crit = loss_mod.build(a.loss, pw, a.beta)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs, 1))

    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=False)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers)

    os.makedirs(a.out, exist_ok=True)
    best, hist = -1.0, []
    for ep in range(1, a.epochs + 1):
        trm, ntr = run_epoch(net, tl, crit, opt, device, a.thr, a.limit_steps)
        vam, nva = run_epoch(net, vl, crit, None, device, a.thr, a.limit_steps)
        sched.step()
        hist.append({"epoch": ep, "train": trm, "val": vam})
        print(f"[unet] ep{ep:3d}  train loss {trm['loss']:.4f} iou {trm['iou']:.4f}"
              f" | val loss {vam['loss']:.4f} iou {vam['iou']:.4f}"
              f" recall {vam['recall']:.4f} prec {vam['precision']:.4f}")
        if vam["iou"] > best:
            best = vam["iou"]
            torch.save({"model": net.state_dict(), "arch": a.arch, "base": a.base,
                        "epoch": ep, "val_iou": best, "args": vars(a)},
                       os.path.join(a.out, "best.pt"))
    with open(os.path.join(a.out, "history.json"), "w", encoding="utf-8") as fh:
        json.dump({"args": vars(a), "history": hist}, fh, ensure_ascii=False, indent=2)
    print(f"\n[unet] 最佳 val IoU {best:.4f}  -> {a.out}/best.pt")
    print("[unet] 看召回率优先于 IoU —— 断一条线会把一个 net 拆成两个，"
          "匈牙利匹配下一次错误同时损伤两个 net。")


if __name__ == "__main__":
    main()
