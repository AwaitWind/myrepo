"""Extract the final answer from chain-of-thought outputs."""

import re

_CLOSE = "</think>"


def extract_final_answer(text: object) -> tuple[str, bool]:
    """If `</think>` appears, return everything after the LAST </think>
    (stripped) and True. Otherwise return text unchanged and False.
    Never raises - broad except returns (text, False).
    """
    try:
        if not isinstance(text, str):
            text = str(text)
        if _CLOSE not in text:
            return text, False
        tail = text.rsplit(_CLOSE, 1)[1]
        return tail.strip(), True
    except Exception:
        return (text if isinstance(text, str) else ""), False


def extract_answer_only(text: object) -> tuple[str, bool]:
    """Extract a single answer line from a CoT-style model response.

    Steps (in order):
        1. If `</think>` appears, take only the text AFTER the last `</think>`.
        2. Strip outer whitespace.
        3. Return the first non-empty line as the answer.
        4. If nothing remains, return ("", False) — caller treats as empty answer.

    `extracted_ok` is True iff a non-empty line was found, regardless of
    whether a `</think>` tag was present.
    """
    if text is None:
        return "", False
    s = text if isinstance(text, str) else str(text)
    if _CLOSE in s:
        s = s.rsplit(_CLOSE, 1)[1]
    s = s.strip()
    if not s:
        return "", False
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line, True
    return "", False


def extract_full_answer(text: object) -> tuple[str, bool]:
    """Like extract_answer_only but keeps the entire post-`</think>` body.

    For free-form generation (summarization) where the answer spans many
    lines: stripping to a single line discards the body. Keep everything
    after the last `</think>`, stripped.
    """
    if text is None:
        return "", False
    s = text if isinstance(text, str) else str(text)
    if _CLOSE in s:
        s = s.rsplit(_CLOSE, 1)[1]
    s = s.strip()
    return s, bool(s)


# Sentence terminator followed by whitespace — a candidate answer boundary.
_SENT_END = re.compile(r"[.!?](\s)")
# Abbreviations whose trailing "." is NOT a sentence boundary, so answers like
# "U.S. Route 66" / "Mr. Smith" / "St. Louis" / "Washington, D.C." survive the
# first-sentence cut. NOTE: "no" is deliberately EXCLUDED — in yes/no QA the
# answer itself is often "No." and we want to keep just "No".
_ABBREVS = frozenset(
    (
        "u.s", "u.k", "u.n", "e.g", "i.e", "mr", "mrs", "ms", "dr", "st",
        "jr", "sr", "vs", "inc", "ltd", "co", "d.c", "a.m", "p.m",
        "prof", "gen", "gov", "sen", "rep", "etc",
    )
)


def trim_to_short_answer(text: object) -> str:
    """Trim a QA answer to its core, dropping explanation the model appended on
    the SAME line.

    Raw-completion models (no chat template) tend to answer and then keep
    explaining on one line, e.g. `No. The Laleli Mosque is located in ...` or
    `3,677 seated (4,000 capacity)`, which a trailing `stop=["\\n"]` cannot cut.
    Two conservative passes:

      1. First sentence — cut at the first '.', '!' or '?' that is followed by
         whitespace, UNLESS the token before it is a known abbreviation
         (U.S., Mr., St., D.C. ...). Protects multi-word proper nouns.
      2. Strip one trailing parenthetical  `<answer> (extra note)`.

    Short-answer datasets only (hotpotqa / qasper / narrativeqa). Summarization
    (gov_report) must NOT use this — use extract_full_answer there instead.
    Idempotent; never raises.
    """
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    for m in _SENT_END.finditer(s):
        head = s[: m.start()].strip()
        if not head:
            continue
        last_word = re.split(r"[\s(]", head)[-1].lower().rstrip(".")
        if last_word in _ABBREVS:
            continue
        s = head
        break
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    return s
