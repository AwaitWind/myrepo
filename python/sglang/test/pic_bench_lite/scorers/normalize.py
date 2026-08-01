import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA-style normalization: lowercase, strip articles + punctuation, collapse whitespace."""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s
