"""Rerank — 2 chế độ qua RAG_RERANKER:

- "llm" (mặc định): gửi Gemini danh sách đoạn đánh số -> nhận mảng JSON thứ tự liên quan.
  +1 lần gọi LLM/câu. LLM trả `[]` (không đoạn nào liên quan) -> rerank trả [] = van
  OUT-OF-DOMAIN: pipeline loại truy vấn, answer() trả "không tìm thấy" mà không gọi LLM
  sinh câu. Tín hiệu này tách câu lạc đề tốt hơn điểm RRF (RRF theo hạng nên không tách được).
- "lexical": trùng token thô — tức thì, miễn phí, thuần Python; cũng là DỰ PHÒNG khi
  gọi LLM lỗi (in ra, không hỏng ngầm).
"""
import json
import re

import config
from rag.schema import Chunk
from rag.text.vi import tokenize

_JSON_ARR_RE = re.compile(r"\[[\d,\s]*\]")


def rerank(query: str, chunks: list[Chunk], keep: int | None = None) -> list[Chunk]:
    keep = keep or config.RERANK_KEEP
    if len(chunks) <= keep:
        return chunks
    if config.RERANKER == "lexical":
        return _lexical(query, chunks, keep)
    # mặc định "llm" (mọi giá trị khác 'lexical' đều coi là llm); lỗi -> dự phòng lexical
    try:
        ranked = _llm(query, chunks, keep)
    except Exception as e:
        print(f"[rerank] fallback lexical (llm lỗi: {e})")
        return _lexical(query, chunks, keep)
    # ranked rỗng = LLM phán "không đoạn nào liên quan" (van out-of-domain).
    # KHÁC với lỗi ở trên (lỗi -> fallback lexical, vẫn trả nguồn).
    if not ranked:
        print("[rerank] LLM báo KHÔNG đoạn nào liên quan → loại (out-of-domain)")
    return ranked


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
    reply = chat(prompt, temperature=0.0)  # xếp hạng cần ổn định, không cần sáng tạo
    m = _JSON_ARR_RE.search(reply)
    if not m:
        raise ValueError(f"LLM không trả JSON array: {reply[:120]!r}")
    parsed = json.loads(m.group(0))
    order = [i for i in parsed if isinstance(i, int) and 0 <= i < len(chunks)]
    order = list(dict.fromkeys(order))  # dedup
    if not order:
        # `[]` TƯỜNG MINH = LLM phán "không đoạn nào liên quan" → tín hiệu out-of-domain
        # (đáng tin hơn điểm RRF, vốn không tách được câu lạc đề). Trả [] để pipeline
        # loại truy vấn → answer() rơi vào nhánh no-context, KHÔNG gọi LLM sinh câu.
        # Mảng CÓ nội dung nhưng toàn chỉ số rác = output hỏng thật → raise (fallback lexical).
        if not parsed:
            return []
        raise ValueError(f"JSON array sai chỉ số: {m.group(0)[:60]}")
    order += [i for i in range(len(chunks)) if i not in order]  # phần LLM bỏ sót
    return [chunks[i] for i in order[:keep]]
