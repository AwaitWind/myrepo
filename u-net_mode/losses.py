"""损失函数：导线像素占比极低，必须偏召回。

为什么偏召回不是调参偏好而是硬要求:
  断一条线会把一个 net 拆成两个，而评测对 net 做匈牙利匹配，
  一次断线同时损伤两个 net 的得分，是非线性的失分放大。
  多连一条线同样有害，但后续几何规则（线段是否与引脚对齐）还有机会过滤。
  失分不对称，损失和阈值就该不对称。

  Tversky 的 beta > alpha 即惩罚漏检重于误检。beta=0.7 是默认起点。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flat(logits, target):
    prob = torch.sigmoid(logits).reshape(logits.shape[0], -1)
    tgt = target.reshape(target.shape[0], -1).float()
    return prob, tgt


class DiceBCELoss(nn.Module):
    """BCE + Dice。pos_weight 缓解正负样本极度不平衡。"""

    def __init__(self, pos_weight: float = 10.0, dice_weight: float = 1.0,
                 smooth: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, target):
        pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=pw)
        prob, tgt = _flat(logits, target)
        inter = (prob * tgt).sum(1)
        dice = 1 - (2 * inter + self.smooth) / (prob.sum(1) + tgt.sum(1) + self.smooth)
        return bce + self.dice_weight * dice.mean()


class TverskyLoss(nn.Module):
    """beta > alpha 时漏检(FN)的惩罚重于误检(FP)，即偏召回。"""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0,
                 bce_weight: float = 0.5, pos_weight: float = 10.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
        self.bce_weight, self.pos_weight = bce_weight, pos_weight

    def forward(self, logits, target):
        prob, tgt = _flat(logits, target)
        tp = (prob * tgt).sum(1)
        fp = (prob * (1 - tgt)).sum(1)
        fn = ((1 - prob) * tgt).sum(1)
        tv = 1 - (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        loss = tv.mean()
        if self.bce_weight:
            pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            loss = loss + self.bce_weight * F.binary_cross_entropy_with_logits(
                logits, target.float(), pos_weight=pw
            )
        return loss


@torch.no_grad()
def seg_metrics(logits, target, thr: float = 0.4):
    """IoU / 召回 / 精确率。

    阈值默认 0.4 而非 0.5 —— 依据同上，断线的代价高于多连，
    二值化阈值应当偏向召回。
    """
    pred = (torch.sigmoid(logits) > thr)
    tgt = target > 0.5
    inter = (pred & tgt).sum().item()
    union = (pred | tgt).sum().item()
    tp = inter
    fp = (pred & ~tgt).sum().item()
    fn = (~pred & tgt).sum().item()
    return {
        "iou": inter / union if union else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
    }


def build(name: str, pos_weight: float, beta: float):
    if name == "dice_bce":
        return DiceBCELoss(pos_weight=pos_weight)
    if name == "tversky":
        return TverskyLoss(alpha=1.0 - beta, beta=beta, pos_weight=pos_weight)
    raise SystemExit(f"未知损失 {name}，可选: dice_bce / tversky")
