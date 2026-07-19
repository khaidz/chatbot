"""Báo cáo trích xuất per-trang — dual-backend theo RAG_STORE (giống cache/querylog):

- pgvector -> bảng `ingest_reports` (tiện THEO DÕI bằng SQL: đếm trang hỏng, lọc theo
  trạng thái, join với docs...). Đi qua connection pool chung `rag/db.py`.
- numpy    -> file <RAG_DATA_DIR>/ingest_reports/<doc_id>.json (ghi atomic).

Mỗi doc một bản ghi (khoá doc_id) — ingest lại là ghi đè. Cột `pages` là mảng JSON:
  [{page, kind, chars, status, note}, ...]  status: ok|ocr_failed|ocr_skipped|blank|error
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path

import config

_pg_init_done = False
_pg_init_lock = threading.Lock()


def _ensure_pg():
    """Tạo bảng ingest_reports một lần (idempotent)."""
    global _pg_init_done
    if _pg_init_done:
        return
    with _pg_init_lock:
        if _pg_init_done:
            return
        from rag import db

        with db.cursor() as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS ingest_reports(
                     doc_id text PRIMARY KEY,
                     source text DEFAULT '',
                     ts timestamptz DEFAULT now(),
                     total_pages int DEFAULT 0,
                     ok_pages int DEFAULT 0,
                     failed_pages int DEFAULT 0,
                     pages jsonb NOT NULL)"""
            )
        _pg_init_done = True


def _path(doc_id: str) -> Path:
    return Path(config.DATA_DIR) / "ingest_reports" / f"{doc_id}.json"


def _rows(pages) -> list[dict]:
    return [{"page": pg.number, "kind": pg.kind, "chars": len(pg.text.strip()),
             "status": pg.status, "note": pg.note} for pg in pages]


def save(doc_id: str, source: str, pages) -> list[dict]:
    """Lưu báo cáo (ghi đè bản cũ cùng doc_id). Trả về danh sách trang HỎNG."""
    rows = _rows(pages)
    bad = [r for r in rows if r["status"] != "ok"]
    total, failed = len(rows), len(bad)

    if config.STORE == "pgvector":
        _ensure_pg()
        from rag import db

        with db.cursor() as cur:
            cur.execute(
                """INSERT INTO ingest_reports
                     (doc_id, source, ts, total_pages, ok_pages, failed_pages, pages)
                   VALUES (%s,%s, now(), %s,%s,%s, %s::jsonb)
                   ON CONFLICT (doc_id) DO UPDATE SET
                     source = EXCLUDED.source, ts = now(),
                     total_pages = EXCLUDED.total_pages, ok_pages = EXCLUDED.ok_pages,
                     failed_pages = EXCLUDED.failed_pages, pages = EXCLUDED.pages""",
                (doc_id, source, total, total - failed, failed,
                 json.dumps(rows, ensure_ascii=False)),
            )
        return bad

    payload = {
        "doc_id": doc_id, "source": source,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "total_pages": total, "ok_pages": total - failed, "failed_pages": failed,
        "pages": rows,
    }
    p = _path(doc_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)  # atomic — UI đọc lúc ingest chạy sẽ không trúng file dở
    return bad


def load(doc_id: str) -> dict | None:
    """Đọc báo cáo, hoặc None nếu chưa có."""
    if config.STORE == "pgvector":
        _ensure_pg()
        from rag import db

        with db.cursor() as cur:
            cur.execute(
                """SELECT doc_id, source, ts::text, total_pages, ok_pages, failed_pages, pages
                   FROM ingest_reports WHERE doc_id=%s""",
                (doc_id,),
            )
            r = cur.fetchone()
        if not r:
            return None
        return {"doc_id": r[0], "source": r[1], "ts": r[2], "total_pages": r[3],
                "ok_pages": r[4], "failed_pages": r[5], "pages": r[6]}

    p = _path(doc_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8-sig") or "{}")
    except json.JSONDecodeError:
        return None


def delete(doc_id: str):
    """Xoá báo cáo kèm khi xoá tài liệu."""
    if config.STORE == "pgvector":
        _ensure_pg()
        from rag import db

        with db.cursor() as cur:
            cur.execute("DELETE FROM ingest_reports WHERE doc_id=%s", (doc_id,))
    else:
        _path(doc_id).unlink(missing_ok=True)
