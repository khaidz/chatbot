"""Alias số hiệu văn bản: "NĐ 13" <-> "Nghị định 13/2023/NĐ-CP".

Chỉ MỞ RỘNG query (thêm biến thể vào chuỗi tìm BM25) — không đụng vào text gốc.
"""
import re

# viết tắt -> dạng đầy đủ
_SHORT = {
    "nđ": "nghị định",
    "nd": "nghị định",
    "tt": "thông tư",
    "qđ": "quyết định",
    "qd": "quyết định",
    "l": "luật",
}
# "nđ 13", "NĐ13", "nd 13/2023", "tt 06"
_SHORT_RE = re.compile(r"\b(nđ|nd|tt|qđ|qd)\s*\.?\s*(\d{1,4})(?:\s*/\s*(\d{4}))?", re.IGNORECASE)
# "nghị định 13", "thông tư 06/2023"
_LONG_RE = re.compile(
    r"\b(nghị\s+định|thông\s+tư|quyết\s+định|luật)\s+(?:số\s+)?(\d{1,4})(?:\s*/\s*(\d{4}))?",
    re.IGNORECASE,
)


def expand_query(query: str) -> str:
    """Trả về query + các biến thể alias (nếu có), nối bằng khoảng trắng."""
    extra: list[str] = []
    for m in _SHORT_RE.finditer(query):
        long_form = _SHORT.get(m.group(1).lower(), "")
        num, year = m.group(2), m.group(3)
        if long_form:
            extra.append(f"{long_form} {num}")
            extra.append(f"{long_form} số {num}")
        if year:
            extra.append(f"{num}/{year}")
    for m in _LONG_RE.finditer(query):
        num, year = m.group(2), m.group(3)
        extra.append(f"số {num}")
        if year:
            extra.append(f"{num}/{year}")
    if not extra:
        return query
    return query + " " + " ".join(dict.fromkeys(extra))  # dedup giữ thứ tự
