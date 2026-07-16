"""Chat đa phiên: kho tài liệu CHUNG, lịch sử hội thoại RIÊNG từng session.

Backend lưu session đi theo RAG_STORE: pgvector -> bảng PostgreSQL,
numpy -> file JSON trong RAG_DATA_DIR\\chat\\.
"""
import config

_chat_store = None


def get_chat_store():
    global _chat_store
    if _chat_store is not None:
        return _chat_store
    if config.STORE == "pgvector":
        from rag.chat.store import PgChatStore

        _chat_store = PgChatStore(config.PG_DSN)
    else:
        from rag.chat.store import JsonChatStore

        _chat_store = JsonChatStore(config.DATA_DIR)
    return _chat_store
