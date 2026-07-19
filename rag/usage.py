"""Đếm token + ước tính chi phí của MỘT lượt hỏi (giống rag/timing.py nhưng cho token).

Vì sao thread-local: mỗi request web / mỗi lượt CLI chạy trên một luồng riêng. Một lượt
hỏi có thể gọi LLM NHIỀU lần (condense + rerank + sinh câu trả lời), nên bucket CỘNG DỒN
mọi lần gọi trên luồng đó; log_query() gọi take() để chốt số của đúng lượt đang chạy.

Số token là SỐ THẬT do provider trả về (`usageMetadata` của Gemini, `usage` của OpenAI),
không phải ước lượng theo ký tự. Lượt trả từ cache = 0 token (không gọi API) — nhờ vậy
Dashboard nói được cache tiết kiệm bao nhiêu.

KHÔNG tính vào đây:
- Token EMBEDDING: API batchEmbedContents của Gemini không trả về số token, muốn biết
  phải gọi thêm countTokens (tốn thêm 1 request/lượt) — không đáng, và embedding câu hỏi
  rẻ hơn nhiều bậc so với sinh câu trả lời.
- Lượt gọi tóm tắt hội thoại (rag/chat/pipeline.py::_maybe_update_summary) chạy SAU
  log_query nên token của nó rơi vào lượt hỏi kế tiếp trên cùng luồng. Tổng cộng dồn vẫn
  đúng, chỉ phân bổ theo từng dòng log là xê dịch.
"""
import threading

import config

_local = threading.local()


def _bucket() -> dict:
    b = getattr(_local, "u", None)
    if b is None:
        b = _local.u = {"tok_in": 0, "tok_out": 0, "calls": 0, "cost_usd": 0.0}
    return b


def cost_usd(tok_in: int, tok_out: int) -> float:
    """Giá cấu hình theo USD / 1 TRIỆU token (RAG_PRICE_IN / RAG_PRICE_OUT)."""
    return (tok_in / 1e6) * config.PRICE_IN + (tok_out / 1e6) * config.PRICE_OUT


def add(tok_in, tok_out):
    """Cộng một lần gọi API vào lượt đang chạy. Provider không trả số -> bỏ qua (0)."""
    tok_in, tok_out = int(tok_in or 0), int(tok_out or 0)
    if not (tok_in or tok_out):
        return
    b = _bucket()
    b["tok_in"] += tok_in
    b["tok_out"] += tok_out
    b["calls"] += 1
    b["cost_usd"] += cost_usd(tok_in, tok_out)


def take() -> dict:
    """Chốt + xoá bucket của luồng này (lượt sau bắt đầu lại từ 0)."""
    b = _bucket()
    _local.u = None
    return b
