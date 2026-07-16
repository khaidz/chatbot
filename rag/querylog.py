"""Query log — ghi MỌI lượt hỏi (kể cả "không tìm thấy") để theo dõi chất lượng,
cân chỉnh ngưỡng tự tin (Todo.md mục 2) và soi latency từng khâu.

Mỗi lượt một dòng: ts, câu hỏi, câu đã condense (nếu khác), top_score (rrf của nguồn
TỐT NHẤT retrieval tìm được — kể cả khi sau đó trả lời "không tìm thấy"), số nguồn
được dùng, grounded, mode (kèm "+cache" nếu trả từ cache), session_id,
và latency: retrieve_ms / rerank_ms / llm_ms / total_ms (lấy từ rag.timing).

Backend theo RAG_STORE: pgvector -> bảng query_log; numpy -> <RAG_DATA_DIR>\\query_log.jsonl.
Ghi log KHÔNG BAO GIỜ được làm hỏng câu trả lời — mọi lỗi chỉ in cảnh báo rồi bỏ qua.
"""
import json
from datetime import datetime
from pathlib import Path

import config
from rag.timing import take

_LAT_COLS = ("retrieve_ms", "rerank_ms", "llm_ms", "total_ms")
_FIELDS = ("ts", "question", "standalone", "top_score", "n_sources",
           "grounded", "mode", "session_id") + _LAT_COLS
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
            for col in _LAT_COLS:  # bảng cũ tạo trước khi có cột latency
                cur.execute(
                    f"ALTER TABLE query_log ADD COLUMN IF NOT EXISTS {col} int DEFAULT 0"
                )
    return _pg_conn


def log_query(question: str, standalone: str, parents: list, result: dict,
              session_id: str = "", top_score: float | None = None):
    try:
        timings = take()
        if top_score is None:
            top_score = max((p.score for p in parents), default=0.0)
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "standalone": standalone if standalone != question else "",
            "top_score": round(top_score, 4),
            "n_sources": len(result.get("sources", [])),
            "grounded": bool(result.get("grounded")),
            "mode": result.get("mode", ""),
            "session_id": session_id,
            "retrieve_ms": timings.get("retrieve_ms", 0),
            "rerank_ms": timings.get("rerank_ms", 0),
            "llm_ms": timings.get("llm_ms", 0),
            "total_ms": timings.get("total_ms", 0) or sum(
                timings.get(c, 0) for c in ("retrieve_ms", "rerank_ms", "llm_ms")
            ),
        }
        if config.STORE == "pgvector":
            with _pg().cursor() as cur:
                cols = _FIELDS[1:]
                cur.execute(
                    f"INSERT INTO query_log({', '.join(cols)}) "
                    f"VALUES ({', '.join(['%s'] * len(cols))})",
                    tuple(rec[f] for f in cols),
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
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # dòng ghi dở (process bị kill giữa chừng) — bỏ qua, không chết cả log
    return rows
