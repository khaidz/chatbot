"""Chuẩn hoá + tách từ tiếng Việt.

- normalize: NFC (bắt buộc — cùng chữ 'ế' có 2 cách mã hoá, không NFC thì BM25 trượt).
- tokenize: ưu tiên underthesea > pyvi > regex thường (không cần cài gì).
Cùng MỘT hàm dùng cho cả lúc index lẫn lúc query — lệch nhau là retrieval hỏng ngầm.
"""
import re
import unicodedata

_backend = None  # ("underthesea"|"pyvi"|"simple", callable)
_TOKEN_RE = re.compile(r"[0-9a-zà-ỹ]+", re.IGNORECASE)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _resolve_backend():
    global _backend
    if _backend is not None:
        return _backend
    try:
        from underthesea import word_tokenize  # type: ignore

        _backend = ("underthesea", lambda s: word_tokenize(s))
        return _backend
    except Exception:
        pass
    try:
        from pyvi import ViTokenizer  # type: ignore

        _backend = ("pyvi", lambda s: ViTokenizer.tokenize(s).split())
        return _backend
    except Exception:
        pass
    _backend = ("simple", None)
    return _backend


def tokenize(text: str) -> list[str]:
    """Trả về list token lowercase, đã NFC. Từ ghép nối bằng '_' (dữ_liệu)."""
    text = normalize(text).lower()
    name, fn = _resolve_backend()
    if name == "simple":
        return _TOKEN_RE.findall(text)
    words = fn(text)
    out: list[str] = []
    for w in words:
        w = w.strip().replace(" ", "_")
        if w and _TOKEN_RE.search(w):
            out.append(w)
            # thêm cả token đơn để query không segment vẫn khớp
            if "_" in w:
                out.extend(p for p in w.split("_") if p)
    return out
