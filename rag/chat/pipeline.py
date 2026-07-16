"""Một lượt chat = condense -> retrieve (pipeline cũ, nguyên vẹn) -> LLM có lịch sử
-> verify citation -> lưu message -> tóm tắt phần hội thoại cũ khi dài.

Condense question: câu hỏi nối tiếp ("thế còn mức phạt?") đứng một mình vô nghĩa
với retrieval — LLM viết lại thành câu ĐỘC LẬP dựa trên hội thoại, rồi câu độc lập
này mới đưa vào retrieve(). Lượt đầu / offline / lỗi -> dùng câu gốc.
"""
import config
from rag.generate.answer import NOT_FOUND, _extractive, verify_citations
from rag.generate.llm import chat
from rag.retrieve.pipeline import build_context, retrieve

_CONDENSE_PROMPT = """Dưới đây là hội thoại giữa người dùng và trợ lý tra cứu tài liệu.
Viết lại "câu hỏi cuối" thành MỘT câu hỏi tiếng Việt ĐỘC LẬP, tự đủ nghĩa khi tách khỏi
hội thoại (giữ nguyên số hiệu văn bản, tên tài liệu, con số được nhắc đến).
Nếu câu hỏi cuối đã tự đủ nghĩa, trả về NGUYÊN VĂN nó.
Chỉ trả về đúng một câu hỏi, không giải thích.

HỘI THOẠI:
{history}

Câu hỏi cuối: {question}"""

_CHAT_PROMPT = """Bạn là trợ lý trả lời câu hỏi DỰA HOÀN TOÀN vào các nguồn được đánh số dưới đây.

Quy tắc BẮT BUỘC:
1. Chỉ dùng thông tin có trong nguồn. KHÔNG suy diễn, KHÔNG dùng kiến thức ngoài.
2. Sau MỖI ý phải có trích dẫn [n] với n là số nguồn chứa ý đó.
3. Nếu các nguồn không chứa thông tin để trả lời, trả lời đúng một câu: "{not_found}"
4. Trả lời bằng tiếng Việt, ngắn gọn, đúng trọng tâm.
5. Câu hỏi có thể tham chiếu hội thoại trước ("thế còn...", "điều đó...") — hiểu ý theo
   hội thoại, nhưng THÔNG TIN trả lời vẫn chỉ lấy từ nguồn.

{summary_block}HỘI THOẠI TRƯỚC:
{history}

CÁC NGUỒN:
{context}

CÂU HỎI HIỆN TẠI: {question}

TRẢ LỜI (kèm [n]):"""

_SUMMARY_PROMPT = """Tóm tắt hội thoại sau thành tối đa 8 câu tiếng Việt, giữ lại các chi tiết
quan trọng: số hiệu văn bản, con số, kết luận đã đưa ra. Chỉ trả về đoạn tóm tắt.

{old_summary}{body}"""


def _llm_available() -> bool:
    return not config.offline_forced() and config.LLM_PROVIDER != "offline"


def _format_history(messages: list[dict], limit_chars: int = 600) -> str:
    lines = []
    for m in messages:
        who = "Người dùng" if m["role"] == "user" else "Trợ lý"
        lines.append(f"{who}: {m['content'][:limit_chars]}")
    return "\n".join(lines) or "(chưa có)"


def condense(history: list[dict], question: str) -> str:
    """Viết lại câu hỏi nối tiếp thành câu độc lập. Mọi trường hợp lỗi -> câu gốc."""
    if not history or not _llm_available():
        return question
    prompt = _CONDENSE_PROMPT.format(
        history=_format_history(history[-config.CHAT_CONDENSE_MSGS:], 400),
        question=question,
    )
    try:
        out = chat(prompt).strip().strip('"').splitlines()[0].strip()
        if 0 < len(out) <= 300:
            return out
    except Exception as e:
        print(f"[condense] lỗi ({str(e)[:100]}) — dùng câu gốc")
    return question


def chat_turn(store, session_id: str, question: str) -> dict:
    """Trả về {answer, sources, grounded, mode, standalone, session_id}."""
    sess = store.get_session(session_id)
    if sess is None:
        raise ValueError(f"Không tìm thấy session '{session_id}' — xem: python cli.py chat --list")
    history = store.get_messages(session_id)

    standalone = condense(history, question)
    parents = retrieve(standalone, sess["dept"], sess["clearance"])

    if not parents:
        result = {"answer": NOT_FOUND, "sources": [], "grounded": True, "mode": "no-context"}
    else:
        context, sources = build_context(parents)
        text, mode = "", "llm"
        if _llm_available():
            recent = history[-(config.CHAT_KEEP_TURNS * 2):]
            summary_block = (
                f"TÓM TẮT HỘI THOẠI CŨ: {sess['summary']}\n\n" if sess.get("summary") else ""
            )
            try:
                text = chat(_CHAT_PROMPT.format(
                    not_found=NOT_FOUND, summary_block=summary_block,
                    history=_format_history(recent), context=context, question=question,
                ))
            except Exception as e:
                print(f"[llm] lỗi ({str(e)[:150]}) — chuyển sang extractive")
        if not text:
            mode = "extractive"
            text = _extractive(standalone, parents)
        cited, grounded = verify_citations(text, len(sources))
        used = [s for s in sources if s["n"] in cited] or sources
        result = {"answer": text, "sources": used, "grounded": grounded, "mode": mode}

    store.add_message(session_id, "user", question, [])
    store.add_message(session_id, "assistant", result["answer"], result["sources"])
    if not sess.get("title"):
        store.set_title(session_id, question[:60])
    _maybe_update_summary(store, session_id, sess, n_new=2)

    result["session_id"] = session_id
    result["standalone"] = standalone
    return result


def _maybe_update_summary(store, session_id: str, sess: dict, n_new: int):
    """Message cũ hơn cửa sổ CHAT_KEEP_TURNS -> gộp vào summary (chỉ khi có LLM)."""
    if not _llm_available():
        return
    total = len(store.get_messages(session_id))
    keep = config.CHAT_KEEP_TURNS * 2
    cut = total - keep                      # mọi message trước 'cut' phải nằm trong summary
    if cut - sess.get("summary_upto", 0) < 4:  # chưa dồn đủ phần cũ, khỏi tốn lượt gọi
        return
    old_part = store.get_messages(session_id)[sess.get("summary_upto", 0): cut]
    old_summary = (
        f"Tóm tắt trước đó: {sess['summary']}\n\n" if sess.get("summary") else ""
    )
    try:
        summary = chat(_SUMMARY_PROMPT.format(
            old_summary=old_summary, body=_format_history(old_part, 400)
        )).strip()
        if summary:
            store.update_summary(session_id, summary[:2000], cut)
    except Exception as e:
        print(f"[summary] lỗi ({str(e)[:100]}) — bỏ qua, sẽ thử lại lượt sau")
