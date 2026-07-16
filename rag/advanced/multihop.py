"""3.2 — Multi-hop: tách câu hỏi phức thành câu con, retrieve từng câu, gộp nguồn."""
import json
import re

import config
from rag.generate.llm import chat
from rag.retrieve.pipeline import retrieve
from rag.schema import Chunk

_JSON_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)


def decompose(query: str) -> list[str]:
    if config.offline_forced() or config.LLM_PROVIDER == "offline":
        return _split_simple(query)
    try:
        reply = chat(
            "Tách câu hỏi sau thành TỐI ĐA 3 câu hỏi con độc lập, mỗi câu tự đủ nghĩa. "
            'Trả về DUY NHẤT mảng JSON các chuỗi, ví dụ ["...", "..."].\n\n'
            f"Câu hỏi: {query}"
        )
        m = _JSON_ARR_RE.search(reply)
        subs = [s for s in json.loads(m.group(0)) if isinstance(s, str) and s.strip()]
        return subs[:3] or [query]
    except Exception:
        return _split_simple(query)


def _split_simple(query: str) -> list[str]:
    parts = re.split(r"\bvà\b|;|\?", query)
    subs = [p.strip() for p in parts if len(p.strip()) > 10]
    return subs[:3] or [query]


def retrieve_multihop(query: str, dept: str = "", clearance: bool = True) -> list[Chunk]:
    seen: set[str] = set()
    merged: list[Chunk] = []
    for sub in decompose(query):
        for p in retrieve(sub, dept, clearance):
            if p.chunk_id not in seen:
                seen.add(p.chunk_id)
                merged.append(p)
    return merged[: config.CONTEXT_MAX_PARENTS + 2]
