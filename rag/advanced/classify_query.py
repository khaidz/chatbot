"""3.1 — Phân loại câu hỏi: smalltalk | factual | multihop.

Heuristic rẻ tiền trước, đủ dùng; chỉ nâng cấp lên LLM khi eval set chứng minh cần.
"""
import re

_SMALLTALK = re.compile(
    r"^(hi|hello|chào|xin chào|cảm ơn|thank|bye|tạm biệt|ok|test)\b", re.IGNORECASE
)
_MULTIHOP = re.compile(r"\bso sánh\b|\bkhác (gì|nhau)\b|\bvừa .+ vừa\b", re.IGNORECASE)


def classify_query(query: str) -> str:
    q = query.strip()
    if _SMALLTALK.match(q):
        return "smalltalk"
    if _MULTIHOP.search(q) or q.count("?") > 1:
        return "multihop"
    return "factual"
