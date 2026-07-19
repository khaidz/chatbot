"""3.4 — NLI check: từng câu trong câu trả lời có được context "hỗ trợ" không.

Hỏi LLM entailment; LLM lỗi -> rơi về ngưỡng trùng token (thô nhưng miễn phí).
Trả về list câu BỊ NGHI NGỜ (không được hỗ trợ).
"""
import re

from rag.generate.llm import chat
from rag.text.vi import tokenize

_SENT_RE = re.compile(r"(?<=[\.\!\?\;])\s+|\n+")
_CITE_RE = re.compile(r"\[\d+\]")


def check(answer_text: str, context: str) -> list[str]:
    sentences = [
        s.strip() for s in _SENT_RE.split(answer_text)
        if len(_CITE_RE.sub("", s).strip()) > 15
    ]
    suspects: list[str] = []
    use_llm = True  # LLM cloud luôn sẵn; lỗi -> except rơi xuống check lexical bên dưới
    ctx_tokens = set(tokenize(context))
    for sent in sentences:
        plain = _CITE_RE.sub("", sent).strip("-• ").strip()
        if use_llm:
            try:
                reply = chat(
                    "NGUỒN:\n" + context[:8000] + "\n\nCâu sau có được NGUỒN hỗ trợ không? "
                    "Trả lời đúng 1 từ: CO hoặc KHONG.\n\nCâu: " + plain
                )
                if "khong" in reply.lower().replace("ô", "o").replace("ộ", "o"):
                    suspects.append(plain)
                continue
            except Exception:
                pass  # rơi xuống check lexical
        toks = set(tokenize(plain))
        if toks and len(toks & ctx_tokens) / len(toks) < 0.6:
            suspects.append(plain)
    return suspects
