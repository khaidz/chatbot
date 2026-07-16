"""BM25 + vector -> RRF fusion. Query đã NFC + segment + mở rộng alias trước khi vào đây.

RRF: score(c) = Σ 1/(RRF_K + rank_c) trên từng danh sách — không cần chuẩn hoá
score giữa 2 hệ khác thang đo.
"""
from dataclasses import replace

import config
from rag.embed import get_embedder
from rag.index import get_store
from rag.schema import Chunk
from rag.text.alias import expand_query
from rag.text.vi import normalize, tokenize


def hybrid_search(query: str, dept: str = "", clearance: bool = True) -> list[Chunk]:
    """Trả về list chunk CON đã fuse, xếp hạng giảm dần."""
    store = get_store()
    embedder = get_embedder()

    q = normalize(query)
    q_expanded = expand_query(q)          # "NĐ 13" -> thêm "nghị định 13"...
    q_tokens = tokenize(q_expanded)

    bm25_hits = store.search_bm25(q_tokens, config.TOP_K_BM25, dept, clearance)
    qvec = embedder.embed([q], is_query=True)[0]
    vec_hits = store.search_vector(qvec, config.TOP_K_VECTOR, dept, clearance)

    # Reciprocal Rank Fusion
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for hits in (bm25_hits, vec_hits):
        for rank, (chunk, _score) in enumerate(hits):
            by_id[chunk.chunk_id] = chunk
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (
                config.RRF_K + rank + 1
            )
    ranked = sorted(fused.items(), key=lambda x: -x[1])
    # gắn điểm RRF vào bản sao (không mutate object dùng chung trong store)
    return [replace(by_id[cid], score=s) for cid, s in ranked]
