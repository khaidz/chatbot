"""Sinh câu trả lời có trích nguồn [n] + VERIFY CITATION bằng code (chống bịa, ~0ms).

- LLM offline/lỗi -> extractive: nhặt câu liên quan nhất từ chính nguồn, kèm [n].
- grounded=True khi: mọi [n] trong câu trả lời đều trỏ đến nguồn tồn tại và có ít nhất
  một trích dẫn; hoặc trả lời trung thực "không tìm thấy".
"""
import re

import config
from rag.generate.llm import chat
from rag.retrieve.pipeline import build_context, retrieve
from rag.schema import Chunk
from rag.text.vi import tokenize

NOT_FOUND = "Không tìm thấy thông tin trong tài liệu."
_CITE_RE = re.compile(r"\[(\d+)\]")
_SENT_RE = re.compile(r"(?<=[\.\!\?\;])\s+|\n+")

_PROMPT = """Bạn là trợ lý trả lời câu hỏi DỰA HOÀN TOÀN vào các nguồn được đánh số dưới đây.

Quy tắc BẮT BUỘC:
1. Chỉ dùng thông tin có trong nguồn. KHÔNG suy diễn, KHÔNG dùng kiến thức ngoài.
2. Sau MỖI ý phải có trích dẫn [n] với n là số nguồn chứa ý đó.
3. Nếu các nguồn không chứa thông tin để trả lời, trả lời đúng một câu: "{not_found}"
4. Trả lời bằng tiếng Việt, ngắn gọn, đúng trọng tâm.

CÁC NGUỒN:
{context}

CÂU HỎI: {query}

TRẢ LỜI (kèm [n]):"""


def answer(query: str, dept: str = "", clearance: bool = True,
           parents: list[Chunk] | None = None) -> dict:
    """Trả về {answer, sources, cited, grounded, mode[, cached]}."""
    import time as _time

    from rag import cache
    from rag.querylog import log_query
    from rag.timing import record, span

    t0 = _time.perf_counter()
    # multihop truyền parents riêng -> câu trả lời phụ thuộc context ngoài, bỏ qua cache
    key = cache.make_key(query, dept, clearance) \
        if (cache.enabled() and parents is None) else None
    if key:
        hit = cache.get(key)
        if hit is not None:
            result, top = hit
            result = dict(result)
            result["cached"] = True
            record("total_ms", (_time.perf_counter() - t0) * 1000)
            log_query(query, query, [],
                      {**result, "mode": result["mode"] + "+cache"}, top_score=top)
            return result

    if parents is None:
        parents = retrieve(query, dept, clearance)
    top_score = max((p.score for p in parents), default=0.0)

    # Ngưỡng tự tin: không có nguồn HOẶC nguồn tốt nhất quá yếu -> từ chối, khỏi gọi LLM.
    if not parents or top_score < config.SCORE_MIN:
        result = {"answer": NOT_FOUND, "sources": [], "cited": [], "grounded": True,
                  "mode": "no-context"}
    else:
        context, sources = build_context(parents)
        mode = "llm"
        text = ""
        try:
            with span("llm_ms"):
                text = chat(_PROMPT.format(not_found=NOT_FOUND, context=context,
                                           query=query))
        except Exception as e:
            print(f"[llm] lỗi ({str(e)[:150]}) — chuyển sang extractive")
        if not text:
            mode = "extractive"
            text = _extractive(query, parents)

        cited, grounded = verify_citations(text, len(sources))
        used = pick_sources(text, cited, sources)
        result = {"answer": text, "sources": used, "cited": sorted(cited),
                  "grounded": grounded, "mode": mode}

    record("total_ms", (_time.perf_counter() - t0) * 1000)
    log_query(query, query, parents, result, top_score=top_score)
    if key and result["mode"] in ("llm", "no-context"):
        cache.put(key, top_score, result)
    return result


def is_not_found(text: str) -> bool:
    return NOT_FOUND.lower()[:20] in text.lower()


def pick_sources(text: str, cited: set[int], sources: list[dict]) -> list[dict]:
    """Nguồn hiển thị = nguồn ĐƯỢC DÙNG, không phải nguồn đã cân nhắc.
    - Có trích dẫn [n] -> chỉ các nguồn được trích.
    - Trả lời 'không tìm thấy' -> KHÔNG có nguồn (retrieval luôn trả top-k gần nhất
      kể cả khi lạc đề — trưng chúng ra sẽ gây hiểu lầm là căn cứ của câu trả lời).
    - Trả lời có nội dung nhưng quên [n] -> hiện hết nguồn tham chiếu (để soi)."""
    used = [s for s in sources if s["n"] in cited]
    if used:
        return used
    return [] if is_not_found(text) else sources


def verify_citations(text: str, n_sources: int) -> tuple[set[int], bool]:
    """Kiểm bằng CODE: mọi [n] phải trỏ nguồn có thật; phải có ít nhất 1 trích dẫn.
    Trả lời trung thực 'không tìm thấy' cũng tính là grounded."""
    if is_not_found(text):
        return set(), True
    refs = {int(m) for m in _CITE_RE.findall(text)}
    valid = {r for r in refs if 1 <= r <= n_sources}
    grounded = bool(refs) and refs == valid
    return valid, grounded


def _extractive(query: str, parents: list[Chunk]) -> str:
    """LLM không có: nhặt tối đa 3 câu trùng từ khoá nhất từ nguồn, kèm [n] thật.
    Chỉ so khớp từ CÓ NGHĨA — bỏ stopword ('của', 'là'...) kẻo câu lạc đề vẫn match."""
    from rag.text.vi import STOPWORDS

    q_tokens = set(tokenize(query)) - STOPWORDS
    if not q_tokens:
        return NOT_FOUND
    scored: list[tuple[float, int, str]] = []
    for idx, p in enumerate(parents, start=1):
        for sent in _SENT_RE.split(p.text):
            sent = sent.strip()
            if len(sent) < 15:
                continue
            overlap = len(q_tokens & (set(tokenize(sent)) - STOPWORDS))
            if overlap > 0:
                scored.append((overlap / (len(q_tokens) + 1), idx, sent))
    if not scored:
        return NOT_FOUND
    scored.sort(key=lambda x: -x[0])
    lines = [f"- {sent} [{idx}]" for _, idx, sent in scored[:3]]
    return "\n".join(lines)
