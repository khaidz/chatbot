"""3.1 — Phân loại câu hỏi: smalltalk | factual | multihop.

Heuristic rẻ tiền trước, đủ dùng; chỉ nâng cấp lên LLM khi eval set chứng minh cần.
Nhận diện smalltalk (chào/cảm ơn/ok/ừ/không/?...) tách ra module smalltalk.
"""
import re

from rag.advanced import smalltalk

_MULTIHOP = re.compile(r"\bso sánh\b|\bkhác (gì|nhau)\b|\bvừa .+ vừa\b", re.IGNORECASE)


def classify_query(query: str) -> str:
    q = query.strip()
    if smalltalk.detect(q):
        return "smalltalk"
    if _MULTIHOP.search(q) or q.count("?") > 1:
        return "multihop"
    return "factual"
