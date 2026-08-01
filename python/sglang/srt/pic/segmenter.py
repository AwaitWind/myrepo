"""
PIC 段切分模块。

详见文档 §2 — split_and_tokenize 数据契约与不变量 S1-S6。
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def _resolve_sink_token_id(tokenizer, sink_token_id: Optional[int]) -> int:
    """Pick the sink token id: explicit override → pad → bos → 0 fallback."""
    if sink_token_id is not None:
        return int(sink_token_id)
    for attr in ("pad_token_id", "bos_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            return int(tid)
    return 0


def split_and_tokenize(
    text: str,
    tokenizer,
    separator: str = "<<PIC_SEP>>",
    sink_len: int = 0,
    sink_token_id: Optional[int] = None,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """将 prompt 文本按分隔符切分并 tokenize，返回拼接 token ids 和段区间。

    Args:
        text:      用户原始 prompt（未经 tokenizer 处理）。
        tokenizer: HuggingFace tokenizer 对象，需支持 .encode(text, add_special_tokens=False)。
        separator: 分隔符字面字符串，默认 "<<PIC_SEP>>"，可由 --pic-separator-str 配置。
        sink_len:  Sink-prefix 长度。>0 时每段 tokenize 前拼接 sink_len 个 sink token
                   作为 attention-sink 吸收器；0 表示禁用（向后兼容）。
        sink_token_id: 显式指定 sink token id；None 表示按 pad_token_id → bos_token_id → 0
                       顺序自动选取。

    Returns:
        (input_ids, pic_segments) 其中：
        - input_ids: List[int]，所有段拼接后的 token id（不包含分隔符 token）。
        - pic_segments: List[Tuple[int, int]]，每个非空段在 input_ids 中的左闭右开区间 [start, end)。
                        当 sink_len>0 时，[start, start+sink_len) 是 sink 前缀，
                        [start+sink_len, end) 是原始内容；hasher/pool/attention 都按整个 [start, end) 处理。

    不变量（§2.1）：
        S1: pic_segments[0][0] == 0
        S2: pic_segments[-1][1] == len(input_ids)
        S3: pic_segments[i][1] == pic_segments[i+1][0]（相邻段端点严格衔接）
        S4: pic_segments[i][1] > pic_segments[i][0]（每段非空；空段会被跳过）
        S5: 各段独立 encode(part, add_special_tokens=False)；分隔符不产生 token
        S6: 单请求生效；批量输入需在调用点拒绝
        S7: sink_len>0 时，每段前 sink_len 个 token 为固定 sink 前缀
    """
    parts = text.split(separator)

    _sink_ids: List[int]
    if sink_len > 0:
        _sink_ids = [_resolve_sink_token_id(tokenizer, sink_token_id)] * int(sink_len)
    else:
        _sink_ids = []

    input_ids: List[int] = []
    pic_segments: List[Tuple[int, int]] = []

    for part in parts:
        if not part:
            # S4：跳过空段（split 出的空字符串）
            continue

        content_ids: List[int] = tokenizer.encode(part, add_special_tokens=False)
        if not content_ids:
            # S4：tokenize 后仍为空（纯空白等边缘情况）也跳过
            continue

        start = len(input_ids)
        if _sink_ids:
            input_ids.extend(_sink_ids)
        input_ids.extend(content_ids)
        end = len(input_ids)
        pic_segments.append((start, end))

    # 若整个文本没有有效段（极端情况），返回把全文当一段
    if not pic_segments and not input_ids:
        content_ids = tokenizer.encode(text, add_special_tokens=False)
        if content_ids:
            if _sink_ids:
                input_ids = list(_sink_ids) + content_ids
            else:
                input_ids = content_ids
            pic_segments = [(0, len(input_ids))]

    return input_ids, pic_segments
