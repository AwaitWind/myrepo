"""从 GT 的 nets[*].edges 渲染导线 mask，并审计 edges 完整度。

为什么这是免费的监督信号:
  GT 没有提供导线 mask，但 `nets[*].edges` 里有全部线段的两个端点。
  按 2~3px 线宽画出来就是像素级准确的标签。
  这不是合成数据 —— 图像仍是真实原图，渲染的只是标注层。

坐标:
  edges 的端点和 bbox 一样是**左下原点**，必须 `y_img = H - y_gt`。
  这里复用 yolo_model/common/cases.py 的同一套约定。

为什么必须先审计:
  data/ 的 594 个样本里有 31.7% 的网络 `edges` 是空的。空 edges 渲染出来
  就是缺线的 mask，等于喂假阴性。而"宁可过分割也不要断线"是这一步的铁律：
  断一条线会把一个 net 拆成两个，在匈牙利匹配下一次错误同时损伤两个 net。
  所以本脚本默认按 --max-empty-ratio 过滤，不达标的样本直接不产出 mask。

用法:
    # 只审计不产出，先看数据能用多少
    python3 u-net_mode/render_mask.py --audit-only
    # 产出 mask（默认只要 edges 完整度达标的样本）
    python3 u-net_mode/render_mask.py --out out/wire_mask
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "yolo_model"))
from common import cases as cases_mod  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def edge_segments(target: dict, height: int):
    """GT nets -> 图像坐标系线段。完成 y 翻转。

    返回 (segments, stats)；segments 为 [(x1, y1, x2, y2)]，左上原点。
    """
    segs = []
    st = Counter()
    for net in (target.get("nets") or {}).values():
        if not isinstance(net, dict):
            continue
        edges = net.get("edges") or {}
        st["nets_total"] += 1
        if not edges:
            st["nets_empty_edges"] += 1
            continue
        for pts in edges.values():
            if not isinstance(pts, list) or len(pts) != 2:
                st["edges_malformed"] += 1
                continue
            try:
                x1 = float(pts[0]["x"]); y1 = float(pts[0]["y"])
                x2 = float(pts[1]["x"]); y2 = float(pts[1]["y"])
            except (KeyError, TypeError, ValueError):
                st["edges_malformed"] += 1
                continue
            # 左下原点 -> 左上原点
            segs.append((x1, height - y1, x2, height - y2))
            st["edges_ok"] += 1
            if x1 == x2 and y1 == y2:
                st["edges_zero_length"] += 1
    return segs, st


def render(segs, size, width: int, dot_radius: int = 1):
    """画 mask。零长度线段画成一个点，否则会在 mask 上凭空消失。"""
    mask = Image.new("L", size, 0)
    dr = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in segs:
        if x1 == x2 and y1 == y2:
            dr.ellipse(
                [x1 - dot_radius, y1 - dot_radius, x1 + dot_radius, y1 + dot_radius],
                fill=255,
            )
        else:
            dr.line([x1, y1, x2, y2], fill=255, width=width)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/wire_mask")
    ap.add_argument("--official", default="赛题六公开数据集/200_train_cases")
    ap.add_argument("--harvest", default="data")
    ap.add_argument("--line-width", type=int, default=3,
                    help="渲染线宽（像素）。偏宽一点，配合偏召回的损失")
    ap.add_argument("--max-empty-ratio", type=float, default=0.05,
                    help="样本内 edges 为空的网络占比上限，超过则剔除。"
                         "官方数据实测为 0，data/ 整体 31.7%")
    ap.add_argument("--min-edges", type=int, default=10, help="样本最少线段数")
    ap.add_argument("--audit-only", action="store_true", help="只统计不写文件")
    ap.add_argument("--hash-cache", default="out/.image_hash_cache.json")
    a = ap.parse_args()

    roots = cases_mod.roots_from_args(a.official or None, a.harvest or None)
    all_cases, _, _ = cases_mod.discover(roots, hash_cache=a.hash_cache)

    if not a.audit_only:
        os.makedirs(a.out, exist_ok=True)

    per_dataset = {}
    rows = []
    kept = 0
    for case in all_cases:
        target = cases_mod.load_target(case.json_path)
        with Image.open(case.image_path) as im:
            W, H = im.size
        segs, st = edge_segments(target, H)
        n_nets = st["nets_total"]
        empty_ratio = st["nets_empty_edges"] / n_nets if n_nets else 1.0
        ok = empty_ratio <= a.max_empty_ratio and st["edges_ok"] >= a.min_edges

        d = per_dataset.setdefault(case.dataset, Counter())
        d["cases"] += 1
        d["nets"] += n_nets
        d["nets_empty"] += st["nets_empty_edges"]
        d["edges"] += st["edges_ok"]
        d["edges_zero_length"] += st["edges_zero_length"]
        d["edges_malformed"] += st["edges_malformed"]
        d["usable" if ok else "rejected"] += 1

        rows.append({
            "case_id": case.case_id, "dataset": case.dataset, "source": case.source,
            "group": case.group, "image": case.image_path, "json": case.json_path,
            "size": [W, H], "nets": n_nets, "nets_empty_edges": st["nets_empty_edges"],
            "empty_ratio": round(empty_ratio, 4), "edges": st["edges_ok"],
            "edges_zero_length": st["edges_zero_length"],
            "usable": ok,
        })
        if ok and not a.audit_only:
            render(segs, (W, H), a.line_width).save(
                os.path.join(a.out, case.case_id + "_mask.png")
            )
            kept += 1

    report = {"args": vars(a),
              "per_dataset": {k: dict(v) for k, v in per_dataset.items()},
              "cases": rows}
    report_path = os.path.join(a.out if not a.audit_only else "out",
                               "mask_audit.json")
    os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print()
    for ds, c in sorted(per_dataset.items()):
        ratio = c["nets_empty"] / c["nets"] * 100 if c["nets"] else 0
        print(f"[mask] {ds:9} 样本 {c['cases']:4}  可用 {c['usable']:4}  剔除 {c['rejected']:4}")
        print(f"[mask] {'':9} 网络 {c['nets']:6}  空edges {c['nets_empty']:6} ({ratio:.1f}%)"
              f"  线段 {c['edges']:7}  零长度 {c['edges_zero_length']}")
    if not a.audit_only:
        print(f"[mask] 产出 {kept} 张 mask -> {a.out}")
    print(f"[mask] 报告 {report_path}")
    print(f"[mask] 过滤阈值 empty_ratio<={a.max_empty_ratio} 且 edges>={a.min_edges}")


if __name__ == "__main__":
    main()
