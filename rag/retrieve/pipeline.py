"""Retrieval là MỘT PIPELINE, không phải 1 câu search:

query -> NFC + segment + alias -> BM25 + vector -> RRF -> (RBAC đã lọc trong query)
      -> rerank -> child->parent -> context đánh số [n]
"""
from dataclasses import replace

import config
from rag.index import get_store
from rag.retrieve.hybrid import hybrid_search
from rag.retrieve.rerank import rerank
from rag.schema import Chunk


def retrieve(query: str, dept: str = "", clearance: bool = True) -> list[Chunk]:
    """Trả về list chunk CHA (đủ ngữ cảnh cho LLM), dedup, giữ thứ tự liên quan."""
    from rag.timing import span

    with span("retrieve_ms"):
        children = hybrid_search(query, dept, clearance)
    if not children:
        return []
    with span("rerank_ms"):
        top_children = rerank(query, children, config.RERANK_KEEP)
    # rerank llm trả [] = "không đoạn nào liên quan" (out-of-domain) -> loại truy vấn
    if not top_children:
        return []

    store = get_store()
    # điểm của cha = điểm RRF cao nhất trong các con được rerank chọn
    by_id: dict[str, Chunk] = {}
    best: dict[str, float] = {}
    order: list[str] = []
    for child in top_children:
        parent = store.get_parent(child.parent_id) if child.parent_id else None
        target = parent or child  # không tìm thấy cha thì dùng chính con
        tid = target.chunk_id
        if tid not in by_id:
            by_id[tid] = target
            best[tid] = child.score
            order.append(tid)
        else:
            best[tid] = max(best[tid], child.score)
    return [
        replace(by_id[t], score=best[t]) for t in order[: config.CONTEXT_MAX_PARENTS]
    ]


def build_context(parents: list[Chunk]) -> tuple[str, list[dict]]:
    """Context đánh số [n] + danh sách nguồn để in/verify citation."""
    blocks: list[str] = []
    sources: list[dict] = []
    max_score = max((p.score for p in parents), default=0.0) or 1.0
    for i, p in enumerate(parents, start=1):
        blocks.append(f"[{i}] (nguồn: {p.source or p.doc_id}, trang {p.page})\n{p.text}")
        sources.append(
            {
                "n": i,
                "chunk_id": p.chunk_id,
                "doc_id": p.doc_id,
                "source": p.source or p.doc_id,
                "page": p.page,
                "score": round(p.score, 4),        # điểm RRF (BM25 + vector)
                "rel": round(p.score / max_score, 4),  # % so với nguồn mạnh nhất
            }
        )
    return "\n\n---\n\n".join(blocks), sources
