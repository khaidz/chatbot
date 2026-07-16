"""Query log — ghi MỌI lượt hỏi (kể cả "không tìm thấy") để theo dõi chất lượng
và cân chỉnh ngưỡng tự tin (Todo.md mục 2).

Mỗi lượt một dòng: ts, câu hỏi, câu đã condense (nếu khác), top_score (rrf của nguồn
TỐT NHẤT retrieval tìm được — kể cả khi sau đó trả lời "không tìm thấy"), số nguồn
được dùng, grounded, mode, session_id.

Backend theo RAG_STORE: pgvector -> bảng query_log; numpy -> <RAG_DATA_DIR>\\query_log.jsonl.
Ghi log KHÔNG BAO GIỜ được làm hỏng câu trả lời — mọi lỗi chỉ in cảnh báo rồi bỏ qua.
"""
import json
from datetime import datetime
from pathlib import Path

import config

_FIELDS = ("ts", "question", "standalone", "top_score", "n_sources",
           "grounded", "mode", "session_id")
_pg_conn = None


def _pg():
    global _pg_conn
    if _pg_conn is None:
        import psycopg2

        _pg_conn = psycopg2.connect(config.PG_DSN)
        _pg_conn.autocommit = True
        with _pg_conn.cursor() as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS query_log(
                     id bigserial PRIMARY KEY,
                     ts timestamptz DEFAULT now(),
                     question text NOT NULL,
                     standalone text DEFAULT '',
                     top_score real DEFAULT 0,
                     n_sources int DEFAULT 0,
                     grounded bool,
                     mode text DEFAULT '',
                     session_id text DEFAULT '')"""
            )
    return _pg_conn


def log_query(question: str, standalone: str, parents: list, result: dict,
              session_id: str = ""):
    try:
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "standalone": standalone if standalone != question else "",
            "top_score": round(max((p.score for p in parents), default=0.0), 4),
            "n_sources": len(result.get("sources", [])),
            "grounded": bool(result.get("grounded")),
            "mode": result.get("mode", ""),
            "session_id": session_id,
        }
        if config.STORE == "pgvector":
            with _pg().cursor() as cur:
                cur.execute(
                    """INSERT INTO query_log(question, standalone, top_score,
                         n_sources, grounded, mode, session_id)
                       VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                    tuple(rec[f] for f in _FIELDS[1:]),
                )
        else:
            p = Path(config.DATA_DIR) / "query_log.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[querylog] lỗi ghi log ({str(e)[:100]}) — bỏ qua")


def read_log() -> list[dict]:
    if config.STORE == "pgvector":
        with _pg().cursor() as cur:
            cur.execute(
                f"SELECT ts::text, {', '.join(_FIELDS[1:])} FROM query_log ORDER BY id"
            )
            return [dict(zip(_FIELDS, r)) for r in cur.fetchall()]
    p = Path(config.DATA_DIR) / "query_log.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]
