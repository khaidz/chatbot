"""3.3 — Self-reflection: LLM tự soát câu trả lời so với nguồn, sửa 1 lần nếu lệch."""
from rag.generate.llm import chat


def reflect(query: str, answer_text: str, context: str) -> str:
    try:
        reply = chat(
            "Soát câu trả lời dưới đây so với NGUỒN. Nếu có ý KHÔNG được nguồn hỗ trợ, "
            "viết lại câu trả lời chỉ giữ các ý có nguồn (giữ trích dẫn [n]). "
            "Nếu đã chuẩn, trả về nguyên văn.\n\n"
            f"NGUỒN:\n{context}\n\nCÂU HỎI: {query}\n\nCÂU TRẢ LỜI:\n{answer_text}"
        )
        return reply or answer_text
    except Exception:
        return answer_text
