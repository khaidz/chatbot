"""Pool kết nối PostgreSQL dùng CHUNG cho mọi thành phần (index, chat, cache, querylog).

Vì sao pool thay vì một connection dùng chung:
- psycopg2 KHÔNG cho nhiều luồng dùng CHUNG một connection an toàn cùng lúc — nhất là
  khối BEGIN/COMMIT tường minh (add_doc). Trước đây server serialize mọi request qua một
  lock toàn cục để né việc này; bỏ lock để cho đa người dùng => MỖI thao tác mượn một
  connection riêng từ pool rồi trả lại ngay. Postgres tự chạy các connection song song.
- Mọi connection autocommit: mỗi câu lệnh là một transaction (trừ add_doc tự BEGIN/COMMIT
  tay trên đúng connection nó mượn). DB op đều NGẮN và KHÔNG giữ connection qua lúc gọi
  LLM/embedding (phần chậm nằm ngoài mọi lock DB), nên pool nhỏ (RAG_PG_POOL_MAX, mặc
  định 16) đủ cho dùng nội bộ.
"""
import threading
from contextlib import contextmanager

import config

_pool = None
_slots = None  # semaphore chặn: borrower CHỜ khi hết slot thay vì nổ "pool exhausted"
_lock = threading.Lock()


def _pool_ref():
    global _pool, _slots
    if _pool is None:
        with _lock:
            if _pool is None:
                try:
                    from psycopg2.pool import ThreadedConnectionPool
                except ImportError as e:
                    raise RuntimeError(
                        "Thiếu psycopg2 — chạy: pip install psycopg2-binary"
                    ) from e
                _slots = threading.Semaphore(config.PG_POOL_MAX)
                _pool = ThreadedConnectionPool(1, config.PG_POOL_MAX, config.PG_DSN)
    return _pool


@contextmanager
def connection():
    """Mượn một connection (autocommit=True) từ pool, chắc chắn trả lại khi xong.

    Semaphore gác đúng PG_POOL_MAX slot: nếu mọi connection đang bận, borrower CHỜ tới
    lượt (DB op vốn ngắn) thay vì bị ThreadedConnectionPool.getconn() ném "pool
    exhausted". add_doc dùng cái này để tự BEGIN/COMMIT trên đúng connection nó mượn.
    """
    pool = _pool_ref()
    _slots.acquire()
    conn = None
    broken = False
    try:
        conn = pool.getconn()
        if not conn.autocommit:
            conn.autocommit = True
        yield conn
    except Exception:
        broken = True
        raise
    finally:
        if conn is not None:
            if broken:  # thao tác vỡ giữa transaction -> dọn sạch trước khi trả pool
                try:
                    conn.rollback()
                except Exception:
                    pass
            pool.putconn(conn)
        _slots.release()


@contextmanager
def cursor():
    """Tiện ích cho thao tác autocommit một-câu: mượn connection + mở cursor."""
    with connection() as conn:
        with conn.cursor() as cur:
            yield cur


def close_pool():
    """Đóng toàn bộ connection (gọi lúc server shutdown; không bắt buộc)."""
    global _pool, _slots
    with _lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None
            _slots = None
