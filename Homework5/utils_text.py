# utils_text.py
import re
import unicodedata
from typing import List

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F]")  # ASCII control chars

def _clean_text(t: str) -> str:
    if not t:
        return ""
    # Normalize Unicode (e.g., ligatures), drop control chars, collapse spaces
    t = unicodedata.normalize("NFKC", t)
    t = _CONTROL_CHARS_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def make_snippet(text: str, max_len: int = 200) -> str:
    t = _clean_text(text)
    return t if len(t) <= max_len else t[: max_len - 3] + "..."

def greedy_chunk(paragraphs: List[str], max_chars: int = 900, overlap: int = 120) -> List[str]:
    chunks, buf = [], ""
    for p in paragraphs:
        s = _clean_text(p)  # clean as we chunk (optional but nice)
        if not s:
            continue
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars:
            buf = buf + "\n" + s
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = (tail + "\n" + s).strip()
    if buf:
        chunks.append(buf)
    return chunks
