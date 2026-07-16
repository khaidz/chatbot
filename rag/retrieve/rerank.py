"""Rerank — 3 chế độ qua RAG_RERANKER:

- tên model HF (mặc định BAAI/bge-reranker-v2-m3): cross-encoder, chất lượng cao nhất,
  CẦN Hugging Face (mạng công ty chặn HF -> không dùng được -> tự fallback lexical).
- "llm": gửi Gemini danh sách đoạn đánh số -> nhận mảng JSON thứ tự liên quan.
  Gần cross-encoder, +1 lần gọi LLM/câu, né HF hoàn toàn.
- "lexical": trùng token thô — tức thì, miễn phí, chất lượng thấp nhất.
Mọi chế độ lỗi đều fallback lexical (in ra, không hỏng ngầm).
"""
import json
import re

import config
from rag.schema import Chunk
from rag.text.vi import tokenize

_ce_model = None
_JSON_ARR_RE = re.compile(r"\[[\d,\s]*\]")


def rerank(query: str, chunks: list[Chunk], keep: int | None = None) -> list[Chunk]:
    keep = keep or config.RERANK_KEEP
    if len(chunks) <= keep:
        return chunks
    mode = "lexical" if config.offline_forced() else config.RERANKER
    if mode == "lexical":
        return _lexical(query, chunks, keep)
    if mode == "llm":
        try:
            return _llm(query, chunks, keep)
        except Exception as e:
            print(f"[rerank] fallback lexical (llm lỗi: {e})")
            return _lexical(query, chunks, keep)
    try:
        return _cross_encoder(mode, query, chunks, keep)
    except Exception as e:
        print(f"[rerank] fallback lexical (cross-encoder lỗi: {str(e)[:120]})")
        return _lexical(query, chunks, keep)


def _lexical(query: str, chunks: list[Chunk], keep: int) -> list[Chunk]:
    q = set(tokenize(query))
    scored = [
        (len(q & set(tokenize(c.text))) / (len(q) + 1), i, c) for i, c in enumerate(chunks)
    ]
    scored.sort(key=lambda x: (-x[0], x[1]))  # điểm bằng nhau giữ thứ tự RRF
    return [c for _, _, c in scored[:keep]]


def _llm(query: str, chunks: list[Chunk], keep: int) -> list[Chunk]:
    from rag.generate.llm import chat

    listing = "\n\n".join(f"[{i}] {c.text[:500]}" for i, c in enumerate(chunks))
    prompt = (
        f"Câu hỏi: {query}\n\n"
        f"Dưới đây là {len(chunks)} đoạn văn đánh số từ 0 đến {len(chunks) - 1}.\n"
        "Xếp hạng các đoạn theo mức LIÊN QUAN đến câu hỏi, giảm dần.\n"
        "Trả về DUY NHẤT một mảng JSON các chỉ số, ví dụ: [2, 0, 5]. Không giải thích.\n\n"
        f"{listing}"
    )
    reply = chat(prompt)
    m = _JSON_ARR_RE.search(reply)
    if not m:
        raise ValueError(f"LLM không trả JSON array: {reply[:120]!r}")
    order = [i for i in json.loads(m.group(0)) if isinstance(i, int) and 0 <= i < len(chunks)]
    order = list(dict.fromkeys(order))  # dedup
    if not order:
        raise ValueError("JSON array rỗng/sai chỉ số")
    order += [i for i in range(len(chunks)) if i not in order]  # phần LLM bỏ sót
    return [chunks[i] for i in order[:keep]]


def _cross_encoder(model_name: str, query: str, chunks: list[Chunk], keep: int) -> list[Chunk]:
    global _ce_model
    if _ce_model is None:
        from sentence_transformers import CrossEncoder  # cần HF

        _ce_model = CrossEncoder(model_name)
    scores = _ce_model.predict([(query, c.text) for c in chunks])
    ranked = sorted(zip(scores, range(len(chunks))), key=lambda x: -x[0])
    return [chunks[i] for _, i in ranked[:keep]]
