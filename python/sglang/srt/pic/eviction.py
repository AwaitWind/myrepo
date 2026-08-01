"""
PIC 淘汰策略模块。

详见文档 §13.10 — EvictionStrategy 接口约定：
  get_priority(entry) -> Comparable
  升序排序后从前往后淘汰（优先级值越小越先淘汰）。

提供策略：
  LRUStrategy   — 按 last_access_time 升序（最久未访问先淘汰）
  LFUStrategy   — 按 hit_count 升序（命中次数最少先淘汰）
  FIFOStrategy  — 按 creation_time 升序（最早创建先淘汰）
  MRUStrategy   — 按 last_access_time 降序（最近访问先淘汰）
  FILOStrategy  — 按 creation_time 降序（最晚创建先淘汰）
  PriorityStrategy — 按 entry.priority 升序（外部写入 priority 字段）
  SLRUStrategy  — Segmented LRU，hit_count >= threshold 的归入保护段（降序）
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.pic.picache import SegmentEntry


class EvictionStrategy(abc.ABC):
    """淘汰策略抽象基类。"""

    @abc.abstractmethod
    def get_priority(self, entry: "SegmentEntry") -> Any:
        """返回可比较的优先级值；升序排序后从头开始淘汰（值越小越先淘汰）。"""
        raise NotImplementedError


class LRUStrategy(EvictionStrategy):
    """最近最少使用（Least Recently Used）。

    按 last_access_time 升序：最久未访问的条目优先被淘汰。
    """

    def get_priority(self, entry: "SegmentEntry") -> float:
        return entry.last_access_time


class LFUStrategy(EvictionStrategy):
    """最不常使用（Least Frequently Used）。

    按 hit_count 升序：命中次数最少的条目优先被淘汰。
    相同 hit_count 时，以 last_access_time 为次级键。
    """

    def get_priority(self, entry: "SegmentEntry"):
        return (entry.hit_count, entry.last_access_time)


class FIFOStrategy(EvictionStrategy):
    """先进先出（First In First Out）。

    按 creation_time 升序：最早创建的条目优先被淘汰。
    """

    def get_priority(self, entry: "SegmentEntry") -> float:
        return entry.creation_time


class MRUStrategy(EvictionStrategy):
    """最近最常使用（Most Recently Used）。

    按 last_access_time 降序：最近访问的条目优先被淘汰。
    升序排序用负值实现降序语义。
    """

    def get_priority(self, entry: "SegmentEntry") -> float:
        return -entry.last_access_time


class FILOStrategy(EvictionStrategy):
    """先进后出（First In Last Out）。

    按 creation_time 降序：最新创建的条目优先被淘汰。
    """

    def get_priority(self, entry: "SegmentEntry") -> float:
        return -entry.creation_time


class PriorityStrategy(EvictionStrategy):
    """用户自定义优先级。

    直接读取 entry.priority；外部调用方负责在合适时机写入该字段。
    priority 值越小越先被淘汰。
    """

    def get_priority(self, entry: "SegmentEntry") -> int:
        return entry.priority


class SLRUStrategy(EvictionStrategy):
    """Segmented LRU（分段 LRU）。

    命中次数 < threshold 的条目归入 "试用段"（优先淘汰）；
    命中次数 >= threshold 的条目归入 "保护段"（后淘汰）。
    同段内按 last_access_time 升序。
    """

    def __init__(self, threshold: int = 2):
        self.threshold = threshold

    def get_priority(self, entry: "SegmentEntry"):
        # (0, time) < (1, time)，试用段先淘汰
        segment = 0 if entry.hit_count < self.threshold else 1
        return (segment, entry.last_access_time)
