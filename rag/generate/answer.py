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
    """Trả về {answer, sources, cited, grounded, mode}."""
    if parents is None:
        parents = retrieve(query, dept, clearance)
    if not parents:
        return {"answer": NOT_FOUND, "sources": [], "cited": [], "grounded": True,
                "mode": "no-context"}

    context, sources = build_context(parents)
    mode = "llm"
    text = ""
    if not config.offline_forced() and config.LLM_PROVIDER != "offline":
        try:
            text = chat(_PROMPT.format(not_found=NOT_FOUND, context=context, query=query))
        except Exception as e:
            print(f"[llm] lỗi ({str(e)[:150]}) — chuyển sang extractive")
    if not text:
        mode = "extractive"
        text = _extractive(query, parents)

    cited, grounded = verify_citations(text, len(sources))
    used = [s for s in sources if s["n"] in cited] or sources
    return {"answer": text, "sources": used, "cited": sorted(cited),
            "grounded": grounded, "mode": mode}


def verify_citations(text: str, n_sources: int) -> tuple[set[int], bool]:
    """Kiểm bằng CODE: mọi [n] phải trỏ nguồn có thật; phải có ít nhất 1 trích dẫn.
    Trả lời trung thực 'không tìm thấy' cũng tính là grounded."""
    if NOT_FOUND.lower()[:20] in text.lower():
        return set(), True
    refs = {int(m) for m in _CITE_RE.findall(text)}
    valid = {r for r in refs if 1 <= r <= n_sources}
    grounded = bool(refs) and refs == valid
    return valid, grounded


def _extractive(query: str, parents: list[Chunk]) -> str:
    """LLM không có: nhặt tối đa 3 câu trùng từ khoá nhất từ nguồn, kèm [n] thật."""
    q_tokens = set(tokenize(query))
    scored: list[tuple[float, int, str]] = []
    for idx, p in enumerate(parents, start=1):
        for sent in _SENT_RE.split(p.text):
            sent = sent.strip()
            if len(sent) < 15:
                continue
            overlap = len(q_tokens & set(tokenize(sent)))
            if overlap > 0:
                scored.append((overlap / (len(q_tokens) + 1), idx, sent))
    if not scored:
        return NOT_FOUND
    scored.sort(key=lambda x: -x[0])
    lines = [f"- {sent} [{idx}]" for _, idx, sent in scored[:3]]
    return "\n".join(lines)
