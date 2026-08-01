"""Prompt builder — mirrors pic_bench.workloads.base.build_prompt.

Two-clause layout: `sys<sep>c1<sep>c2<sep>...<sep>cN<sep>query[<\\n\\n>suffix]`
- The leading `sys<sep>` guarantees every chunk (including the first one) sits
  at a non-trivial position in the prompt, so no PIC segment can hit purely
  because it happens to align to position 0.
- The trailing `<sep>` before the query closes the last chunk segment for
  sglang's segment cache (SegmentTokenDatabase in pic_bench parlance).

For non-PIC modes, pass `separator=""` — the layout collapses to
`sys + c1 + c2 + ... + query`, matching the historical hard-coded
`{SYS}{C1}{C2}{C3}{Q}` form used in quick_test_online.py.
"""

_DEFAULT_SYSTEM_PROMPT = (
    "Read the following documents and answer the question that follows."
)


def build_prompt(
    chunks,
    query: str,
    separator: str,
    query_suffix=None,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Concatenate sys + chunks + query with the backend-specific separator.

    Cache-behavior invariants must hold across measurement modes:
      * pic mode uses `separator=<<PIC_SEP>>` (or whatever
        --pic-separator-str is set to).
      * non-PIC modes use `separator=""`.

    `query_suffix` is appended after `\\n\\n` when set; e.g.
    `"Answer with only the answer, no explanation."` nudges CoT models
    toward terse output that scorer.extract_answer_only can parse.
    """
    if chunks:
        body = system_prompt + separator + separator.join(chunks) + separator + query
    else:
        body = system_prompt + separator + query
    if query_suffix:
        return body + "\n\n" + query_suffix
    return body
