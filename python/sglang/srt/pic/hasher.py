"""
PIC 段哈希模块。

规范：SHA-256(int32 little-endian bytes)[:16]
详见文档 §3 — 不可随意更改序列化方式（协议级常量）。
"""

from __future__ import annotations

import array
import hashlib
from typing import Iterable, Union

import torch

# 哈希摘要只取前 16 字节（128-bit），作为 _entries dict 的 key。
_HASH_LEN = 16

# TokenIdsLike：可迭代的整数序列或 torch.Tensor
TokenIdsLike = Union[torch.Tensor, Iterable[int]]


def segment_hash(token_ids: TokenIdsLike) -> bytes:
    """计算 token_ids 序列的 PIC 段哈希值。

    序列化方式：int32 小端序紧凑字节流 -> SHA-256 -> 取前 16 字节。

    Args:
        token_ids: Token id 序列，可以是 torch.Tensor 或任意 Iterable[int]。

    Returns:
        长度为 16 的 bytes 对象，作为 PICache._entries 的 dict key。

    重要约束（§3）：
    - 必须使用 int32（有符号）小端序
    - SHA-256 只取前 16 字节
    - 任何修改都会导致跨进程缓存不兼容
    """
    if isinstance(token_ids, torch.Tensor):
        # 规范：.detach().to(torch.int32).cpu().contiguous().numpy().tobytes()
        raw_bytes = (
            token_ids.detach().to(torch.int32).cpu().contiguous().numpy().tobytes()
        )
    else:
        # 规范：array.array("i", list(token_ids)).tobytes()
        # "i" = signed int32（平台相关字节序，但 x86/ARM 均为 little-endian）
        raw_bytes = array.array("i", list(token_ids)).tobytes()

    digest = hashlib.sha256(raw_bytes).digest()
    return digest[:_HASH_LEN]
