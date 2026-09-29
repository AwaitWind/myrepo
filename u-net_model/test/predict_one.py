"""单个案例跑训练好的 U-Net，看导线分割效果。摸底用，不参与训练流程。

和 test_sanity.py 的区别：那个在训练**前**验证接线（单批次过拟合），
这个在训练**后**验证效果（整图推理 + 对 GT 评测）。

产出:
  <out>_mask.png     预测的导线 mask（白=导线）
  <out>_overlay.png  原图上叠预测，绿=命中 GT、红=漏检、蓝=多预测，直接看错在哪
  <out>.json         指标与参数
  图片旁边有 GT json 时，额外报 IoU / recall / spill

用法（在仓库根目录执行，即能看见 200_train_cases/ 的地方）:
    python3 u-net_mode/test/predict_one.py --ckpt out/unet_runs/wire/best.pt \\
        200_train_cases/0006/0006-jlc.png
    python3 u-net_mode/test/predict_one.py --ckpt out/unet_runs/wire/best.pt \\
        --case 0006                    # 按案例号找图，省得写全路径
    python3 u-net_mode/test/predict_one.py --ckpt ... 图.png --device 0

为什么整图要切片推理:
  训练是在 512 切片上做的，直接喂 17MP 的整图既爆显存又与训练分布不符。
  这里按训练时的窗口滑窗，重叠区**取概率均值**而不是直接覆盖 ——
  边缘格点的感受野不完整、预测偏弱，直接覆盖会在拼缝处留下断线，
  而断线正是这个任务最贵的错误（一条断线在匈牙利匹配下同时损伤两个 net）。

评测线宽的坑（与 exp_linewidth.py 同一个道理）:
  拿训练时的线宽渲染 GT 来评自己，是循环论证 —— 线宽 3 训出来的模型
  对线宽 3 的 mask 当然吻合，但它抠出的线比真实导线粗一圈。
  所以默认对**线宽 1** 的 mask 评测，那才是真实导线位置
  （实测线宽 1 时 mask 命中前景 98.8~100%）。用 --gt-line-width 可改。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))        # <u-net 目录>/test
_PKG = os.path.dirname(_HERE)                             # <u-net 目录>
_ROOT = os.path.dirname(_PKG)                             # 仓库根
# 目录名在不同分支上叫 u-net_mode / u-net_model，所以从 __file__ 推导，不写死
sys.path.insert(0, _PKG)
sys.path.insert(0, os.path.join(_ROOT, "yolo_model"))
import dataset as ds_mod  # noqa: E402
import model as model_mod  # noqa: E402
import render_mask as rm  # noqa: E402

from common import tiling  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def load_model(ckpt_path: str, device):
    """从 checkpoint 自己读 arch/base，不让调用方猜 —— 猜错了权重加载会报形状不匹配。"""
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"找不到权重 {ckpt_path}。\n"
            "  先训练: bash u-net_mode/run.sh --device 0\n"
            "  训练产出在 out/unet_runs/<name>/best.pt"
        )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "unet")
    base = ck.get("base", 64)
    # resnet18_unet 走这条时不要联网取 ImageNet 权重，紧接着就被 state_dict 覆盖了
    net = model_mod.build(arch, base=base, pretrained=False)
    missing, unexpected = net.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"[test] 警告: state_dict 不完全匹配  缺失 {len(missing)}  多余 {len(unexpected)}")
    net.to(device).eval()
    trained_args = ck.get("args") or {}
    print(f"[test] 权重 {ckpt_path}")
    print(f"[test] 架构 {arch}  base {base}  训练至 epoch {ck.get('epoch')}  "
          f"val_iou {ck.get('val_iou')}")
    return net, trained_args


@torch.no_grad()
def predict_full(net, img: np.ndarray, window: int, overlap: float, device):
    """滑窗推理整图，重叠区取概率均值。返回 float32 概率图，与原图同尺寸。"""
    H, W = img.shape[:2]
    prob = np.zeros((H, W), dtype=np.float32)
    hits = np.zeros((H, W), dtype=np.float32)
    stride = tiling.stride_of(window, overlap)
    n = 0
    for oy in tiling.tile_origins(H, window, stride):
        for ox in tiling.tile_origins(W, window, stride):
            tile = img[oy:oy + window, ox:ox + window]
            th, tw = tile.shape[:2]
            if th != window or tw != window:      # 贴边不足补白，与训练时一致
                pad = np.full((window, window, 3), 255, dtype=tile.dtype)
                pad[:th, :tw] = tile
                tile = pad
            x = tile.astype(np.float32) / 255.0
            x = (x - ds_mod.IMAGENET_MEAN) / ds_mod.IMAGENET_STD
            t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
            out = net(t.unsqueeze(0).to(device))
            p = torch.sigmoid(out)[0, 0].cpu().numpy()
            prob[oy:oy + th, ox:ox + tw] += p[:th, :tw]
            hits[oy:oy + th, ox:ox + tw] += 1.0
            n += 1
    return prob / np.maximum(hits, 1e-6), n


def find_gt(image_path: str):
    """图片旁边的 GT json。官方是 <stem>_target.json，自采是 <目录名>.json。"""
    d = os.path.dirname(os.path.abspath(image_path))
    names = [f for f in os.listdir(d) if not f.startswith("._")]
    target = [f for f in names if f.endswith("_target.json")]
    if target:
        return os.path.join(d, target[0])
    same = [f for f in names if f == os.path.basename(d) + ".json"]
    if same:
        return os.path.join(d, same[0])
    js = [f for f in names if f.endswith(".json")]
    return os.path.join(d, js[0]) if len(js) == 1 else None


def gt_mask(gt_json: str, size, line_width: int):
    """渲染 GT 导线 mask。y 翻转在 render_mask.edge_segments 里完成。"""
    import json as _json

    with open(gt_json, encoding="utf-8") as fh:
        target = _json.load(fh)
    W, H = size
    segs, st = rm.edge_segments(target, H)
    mask = np.asarray(rm.render(segs, (W, H), line_width).convert("L")) > 127
    return mask, st


def metrics(pred: np.ndarray, true: np.ndarray):
    """IoU / recall / spill。

    recall 和 spill 带容差，理由同 exp_linewidth.py：预测线与真值线差一两个
    像素不算错，但完全糊到非导线区域要算错。严格 IoU 也一并给出做参照。
    """
    from scipy import ndimage

    inter = int((pred & true).sum())
    union = int((pred | true).sum())
    tp = int((ndimage.binary_dilation(pred, iterations=1) & true).sum())
    spill = int((pred & ~ndimage.binary_dilation(true, iterations=2)).sum())
    return {
        "iou_strict": inter / union if union else 1.0,
        "recall": tp / int(true.sum()) if true.sum() else 1.0,
        "spill": spill / int(pred.sum()) if pred.sum() else 0.0,
        "pred_px": int(pred.sum()),
        "true_px": int(true.sum()),
    }


def overlay(img: np.ndarray, pred: np.ndarray, true=None):
    """可视化。有 GT 时按命中/漏检/多预测三色，没有 GT 就单色画预测。"""
    vis = img.copy()
    if true is None:
        vis[pred] = (255, 0, 0)
        return Image.fromarray(vis)
    vis[true & ~pred] = (255, 0, 0)     # 漏检，最贵的错误
    vis[pred & ~true] = (0, 0, 255)     # 多预测
    vis[pred & true] = (0, 200, 0)      # 命中
    return Image.fromarray(vis)


def resolve_image(a) -> str:
    if a.image:
        if not os.path.exists(a.image):
            raise SystemExit(f"找不到图片: {a.image}")
        return a.image
    # --case 按案例号在候选数据根里找
    from common import cases as cases_mod

    roots = [r for r, _ in cases_mod.default_roots()]
    for root in roots:
        for d in glob.glob(os.path.join(root, a.case)):
            pngs = [p for p in glob.glob(os.path.join(d, "*.png"))
                    if not os.path.basename(p).startswith("._")]
            if len(pngs) == 1:
                return pngs[0]
    raise SystemExit(f"在 {roots} 里找不到案例 {a.case}")


def main():
    ap = argparse.ArgumentParser(description="单个案例跑训练好的 U-Net 导线分割")
    ap.add_argument("image", nargs="?", default=None, help="图片路径")
    ap.add_argument("--case", default=None, help="案例号（如 0006），替代直接给图片路径")
    ap.add_argument("--ckpt", default="out/unet_runs/wire/best.pt")
    ap.add_argument("--thr", type=float, default=None,
                    help="二值化阈值。默认沿用训练时的值（偏召回，通常 0.4）")
    ap.add_argument("--window", type=int, default=None,
                    help="滑窗大小。默认沿用训练时的值")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--gt-line-width", type=int, default=1,
                    help="渲染 GT 用的线宽。默认 1 = 真实导线位置，避免自己评自己")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="默认 out/unet_test/<图名>")
    a = ap.parse_args()

    if not a.image and not a.case:
        raise SystemExit("给一个图片路径，或用 --case 0006 指定案例号")

    image_path = resolve_image(a)
    device = torch.device("cpu" if a.device in ("cpu", "") else f"cuda:{a.device}")
    net, targs = load_model(a.ckpt, device)

    # 阈值和窗口默认跟训练时一致，避免推理口径与训练口径不一致
    thr = a.thr if a.thr is not None else float(targs.get("thr") or 0.4)
    window = a.window or int(targs.get("window") or 512)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_base = a.out or os.path.join("out", "unet_test", stem)
    os.makedirs(os.path.dirname(os.path.abspath(out_base)) or ".", exist_ok=True)

    with Image.open(image_path) as raw:
        img = np.asarray(raw.convert("RGB"))
    H, W = img.shape[:2]
    print(f"[test] 图片 {image_path}  {W}x{H}")
    print(f"[test] 滑窗 {window}  重叠 {a.overlap}  阈值 {thr}  设备 {device}")

    prob, n_tiles = predict_full(net, img, window, a.overlap, device)
    pred = prob > thr
    print(f"[test] 切片 {n_tiles} 个  预测导线像素 {int(pred.sum())} "
          f"({pred.mean()*100:.3f}%)")

    Image.fromarray((pred * 255).astype(np.uint8)).save(out_base + "_mask.png")

    gt_json = find_gt(image_path)
    m = None
    true = None
    if gt_json:
        true, st = gt_mask(gt_json, (W, H), a.gt_line_width)
        m = metrics(pred, true)
        m["gt_json"] = os.path.basename(gt_json)
        m["gt_line_width"] = a.gt_line_width
        m["gt_nets"] = st["nets_total"]
        m["gt_nets_empty_edges"] = st["nets_empty_edges"]
        m["gt_edges"] = st["edges_ok"]

    overlay(img, pred, true).save(out_base + "_overlay.png")

    payload = {"image": image_path, "size": [W, H], "ckpt": a.ckpt,
               "arch": targs.get("arch"), "thr": thr, "window": window,
               "overlap": a.overlap, "tiles": n_tiles,
               "pred_px": int(pred.sum()), "metrics": m}
    with open(out_base + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    if m:
        print(f"\n[test] 对照 {m['gt_json']}（线宽 {a.gt_line_width}）")
        print(f"[test] GT 网络 {m['gt_nets']}  空 edges {m['gt_nets_empty_edges']}"
              f"  线段 {m['gt_edges']}")
        print(f"[test] recall     {m['recall']*100:.1f}%   (真实导线像素有多少被抓到，1px 容差)")
        print(f"[test] spill      {m['spill']*100:.1f}%   (预测像素有多少糊到非导线区，2px 容差)")
        print(f"[test] IoU(严格)  {m['iou_strict']*100:.1f}%  (无容差，受线宽差异影响，仅作参照)")
        if m["gt_nets_empty_edges"]:
            print("[test] 注意: 该样本有空 edges 的网络，GT mask 本身缺线，"
                  "recall 会被低估、spill 会被高估")
    else:
        print("\n[test] 图片旁边没找到 GT json，只产出 mask 和可视化，不评指标")

    print(f"\n[test] mask     {out_base}_mask.png")
    print(f"[test] 可视化   {out_base}_overlay.png"
          + ("  (绿=命中 红=漏检 蓝=多预测)" if m else "  (红=预测导线)"))
    print(f"[test] JSON     {out_base}.json")
    print("\n[test] 看 recall 优先于 IoU —— 断一条线会把一个 net 拆成两个，"
          "匈牙利匹配下一次错误同时损伤两个 net。")
    print("[test] 提醒: mask 不是最终答案，到 nets.hyperGraph 还需要"
          "骨架化 → 打断 → 并查集 → 引脚吸附，那条链尚未实现。")


if __name__ == "__main__":
    main()
