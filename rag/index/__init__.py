"""Factory chọn backend lưu trữ: numpy (file cục bộ) | pgvector (PostgreSQL)."""
import threading

import config

_store = None
_lock = threading.Lock()


def get_store():
    global _store
    if _store is not None:  # đường nhanh: đã tạo -> khỏi lock
        return _store
    with _lock:  # double-checked: nhiều request web đồng thời chỉ tạo MỘT store
        if _store is not None:
            return _store
        if config.STORE == "pgvector":
            from rag.index.pg_store import PgVectorStore

            _store = PgVectorStore(config.PG_DSN)
        else:
            from rag.index.store import NumpyStore

            _store = NumpyStore(config.DATA_DIR)
    return _store
