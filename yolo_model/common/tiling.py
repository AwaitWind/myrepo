"""切片与标签裁剪，两个 YOLO 共用。

切片参数依据 `赛题六_模型设计.md` §M1a: 1024 窗口、20% 重叠（步长 819）。
末片贴边对齐，避免右/下边缘出现残窄片导致目标被反复截断。
"""

from __future__ import annotations


def tile_origins(total: int, window: int, stride: int):
    """切片起点列表。末片贴边，保证覆盖整个边缘。"""
    if total <= window:
        return [0]
    pos = list(range(0, total - window + 1, stride))
    if pos[-1] != total - window:
        pos.append(total - window)
    return pos


def stride_of(window: int, overlap: float) -> int:
    return max(1, int(round(window * (1 - overlap))))


def clip_to_tile(boxes, ox: int, oy: int, window: int, keep_ratio: float = 0.5):
    """把整图坐标的框裁到某个切片内，返回 YOLO 归一化标签。

    boxes: [(x1, y1, x2, y2, *rest)]，整图像素坐标、左上原点。
    keep_ratio: 落在切片内的面积占比下限，低于此值丢弃。
      依据 §M1a "切片时丢弃被截断超过 50% 的实例" —— 截断过半的目标形状先验已破坏，
      留着会教模型把半个符号当完整符号。

    返回 (labels, n_dropped)；labels 为 [(cx, cy, w, h, rest)]，均已除以 window。
    """
    labels = []
    dropped = 0
    for box in boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        rest = box[4:]
        ix1, iy1 = max(x1, ox), max(y1, oy)
        ix2, iy2 = min(x2, ox + window), min(y2, oy + window)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        full = (x2 - x1) * (y2 - y1)
        if full <= 0 or (ix2 - ix1) * (iy2 - iy1) / full < keep_ratio:
            dropped += 1
            continue
        cx = (ix1 + ix2) / 2 - ox
        cy = (iy1 + iy2) / 2 - oy
        labels.append(
            ((cx / window), (cy / window), (ix2 - ix1) / window, (iy2 - iy1) / window, rest)
        )
    return labels, dropped


def format_label(class_id: int, cx: float, cy: float, w: float, h: float) -> str:
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def write_data_yaml(out_root: str, names: dict[int, str]) -> str:
    import os

    path = os.path.join(out_root, "data.yaml")
    lines = [
        f"path: {os.path.abspath(out_root)}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx in sorted(names):
        lines.append(f"  {idx}: {names[idx]}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
