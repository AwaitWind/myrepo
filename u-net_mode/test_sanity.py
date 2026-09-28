"""接线自检：改完模型或损失后跑一次，确认没把网络改坏。

比看长跑的 loss 曲线快得多。三项检查:

  1. 形状与数值   两个架构各做一次前向，输出形状必须等于标签形状；
                  两种损失在该输出上必须是有限值。
  2. 梯度         反向一次，所有可训练参数必须拿到非 None 且有限的梯度。
  3. 单批次过拟合 这是决定性的一项。一个接线错误（跳连接错、上采样尺寸没对齐、
                  损失里标签和预测调换）的分割网络**过拟合不了单批次**。
                  IoU 必须在限定步数内越过阈值。

实测参考（base=16, 8x256x256, 正样本占比 2.9%, lr 3e-3, CPU）:
    step  20  iou 0.036   <- 退化解，全判正
    step  60  iou 0.853   <- 走出退化解
    step 100  iou 0.975
  前 40 步停在 recall 1.0 / precision 0.04 是正常的：pos_weight 大时网络
  先学会"全判正"，要 50~60 步才走出来。所以阈值判定要给足步数，
  在前 40 步就断言失败会误报。

数据来源:
  有 --mask-dir 就用真实切片；没有就合成一批画了线的图。合成数据在这里只用于
  验证代码接线，不是训练样本，不受赛题"禁止合成数据"的约束。

用法:
    python3 u-net_mode/test_sanity.py
    python3 u-net_mode/test_sanity.py --mask-dir out/wire_mask --arch resnet18_unet
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as ds_mod  # noqa: E402
import losses as loss_mod  # noqa: E402
import model as model_mod  # noqa: E402

ARCHS = {"unet": {"base": 16}, "resnet18_unet": {"pretrained": False}}


def synth_batch(n: int, size: int, seed: int = 0):
    """合成一批画了正交线段的图，仅用于验证接线。"""
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        img = Image.new("RGB", (size, size), (255, 255, 255))
        mask = Image.new("L", (size, size), 0)
        di, dm = ImageDraw.Draw(img), ImageDraw.Draw(mask)
        for _ in range(rng.randint(8, 14)):
            x, y = rng.randrange(size), rng.randrange(size)
            if rng.random() < 0.5:
                seg = [x, y, min(size - 1, x + rng.randint(30, 120)), y]
            else:
                seg = [x, y, x, min(size - 1, y + rng.randint(30, 120))]
            di.line(seg, fill=(127, 195, 127), width=3)  # 用 jlc 的绿色，不是纯黑
            dm.line(seg, fill=255, width=3)
        a = np.asarray(img).astype(np.float32) / 255.0
        a = (a - ds_mod.IMAGENET_MEAN) / ds_mod.IMAGENET_STD
        xs.append(torch.from_numpy(a).permute(2, 0, 1))
        ys.append(torch.from_numpy((np.asarray(mask) > 127).astype(np.float32)).unsqueeze(0))
    return torch.stack(xs), torch.stack(ys)


def real_batch(mask_dir: str, n: int, size: int, min_pos: int):
    rows = ds_mod.load_usable(mask_dir)
    ds = ds_mod.WireTiles(rows, size, 0.2, min_pos=min_pos, augment=False)
    if len(ds) < n:
        raise SystemExit(f"{mask_dir} 只有 {len(ds)} 个合格切片，不足 {n}")
    xs, ys = zip(*[ds[i] for i in range(n)])
    return torch.stack(xs), torch.stack(ys)


def check_shapes_and_losses(size: int) -> list[str]:
    fails = []
    y = (torch.rand(1, 1, size, size) > 0.97).float()
    x = torch.randn(1, 3, size, size)
    for arch, kw in ARCHS.items():
        net = model_mod.build(arch, **kw)
        out = net(x)
        if out.shape != y.shape:
            fails.append(f"[形状] {arch} 输出 {tuple(out.shape)} != 标签 {tuple(y.shape)}")
            continue
        for ln in ("dice_bce", "tversky"):
            v = loss_mod.build(ln, 10.0, 0.7)(out, y)
            if not torch.isfinite(v):
                fails.append(f"[数值] {arch} + {ln} 损失非有限: {v.item()}")
        print(f"  形状/损失 {arch:16} 输出{tuple(out.shape)} "
              f"参数{sum(p.numel() for p in net.parameters())/1e6:.1f}M  OK")
    return fails


def check_gradients(arch: str, x, y) -> list[str]:
    net = model_mod.build(arch, **ARCHS[arch])
    loss = loss_mod.build("tversky", 10.0, 0.7)(net(x[:2]), y[:2])
    loss.backward()
    missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
    nonfinite = [n for n, p in net.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    fails = []
    if missing:
        fails.append(f"[梯度] {len(missing)} 个参数无梯度，例: {missing[:3]}")
    if nonfinite:
        fails.append(f"[梯度] {len(nonfinite)} 个参数梯度非有限，例: {nonfinite[:3]}")
    if not fails:
        n_par = sum(1 for _, p in net.named_parameters() if p.requires_grad)
        print(f"  梯度回传 {arch:16} {n_par} 个参数全部拿到有限梯度  OK")
    return fails


def check_overfit(arch: str, x, y, steps: int, lr: float, min_iou: float,
                  thr: float) -> list[str]:
    torch.manual_seed(0)
    net = model_mod.build(arch, **ARCHS[arch])
    crit = loss_mod.build("tversky", 10.0, 0.7)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    net.train()
    best, trace = 0.0, []
    for s in range(1, steps + 1):
        loss = crit(net(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % max(1, steps // 6) == 0 or s == steps:
            net.eval()
            with torch.no_grad():
                m = loss_mod.seg_metrics(net(x), y, thr)
            net.train()
            best = max(best, m["iou"])
            trace.append((s, loss.item(), m))
            print(f"  过拟合 {arch:16} step {s:4d}  loss {loss.item():.4f}  "
                  f"iou {m['iou']:.4f}  recall {m['recall']:.4f}  prec {m['precision']:.4f}")
            if best >= min_iou:
                break
    if best < min_iou:
        return [f"[过拟合] {arch} {steps} 步内最高 IoU 仅 {best:.4f} < {min_iou}，"
                "网络可能接线错误（跳连、上采样尺寸、或损失里预测与标签调换）"]
    print(f"  过拟合 {arch:16} 通过，最高 IoU {best:.4f}")
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-dir", default=None,
                    help="真实 mask 目录；不给则用合成批次")
    ap.add_argument("--arch", default="unet", choices=sorted(ARCHS),
                    help="做过拟合和梯度检查的架构；形状检查始终覆盖全部架构")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min-iou", type=float, default=0.7)
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--min-pos", type=int, default=800)
    a = ap.parse_args()

    torch.manual_seed(0)
    if a.mask_dir:
        x, y = real_batch(a.mask_dir, a.batch, a.size, a.min_pos)
        src = f"真实切片 {a.mask_dir}"
    else:
        x, y = synth_batch(a.batch, a.size)
        src = "合成批次（仅验证接线）"
    print(f"数据: {src}  批次 {tuple(x.shape)}  正样本占比 {y.mean().item()*100:.2f}%\n")

    fails = []
    print("[1/3] 形状与数值")
    fails += check_shapes_and_losses(a.size)
    print("\n[2/3] 梯度回传")
    fails += check_gradients(a.arch, x, y)
    print("\n[3/3] 单批次过拟合")
    fails += check_overfit(a.arch, x, y, a.steps, a.lr, a.min_iou, a.thr)

    print()
    if fails:
        print(f"自检失败 {len(fails)} 项:")
        for f in fails:
            print("  " + f)
        raise SystemExit(1)
    print("自检全部通过。注意: 这只证明接线正确和有容量，"
          "**不证明泛化能力** —— 那需要 286 个样本的完整长跑。")


if __name__ == "__main__":
    main()
