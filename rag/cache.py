"""Answer cache exact-match — câu hỏi lặp lại trả ngay, khỏi tốn embedding + rerank + LLM.

- Key = md5(câu hỏi NFC-lower-gọn khoảng trắng + dept + clearance) — RBAC nằm TRONG key,
  người không có quyền không bao giờ trúng cache của người có quyền.
- Vô hiệu khi KHO ĐỔI: mỗi entry lưu corpus_sig (md5 danh sách sha tài liệu trong kho);
  ingest/xoá tài liệu là sig đổi -> entry cũ thành miss và bị dọn. Không cần TTL thời gian.
- Chỉ cache mode "llm" và "no-context" (extractive = suy giảm tạm thời do LLM lỗi — không cache).
- Tắt bằng: set RAG_CACHE=off
- Backend theo RAG_STORE: bảng answer_cache (pgvector) | <RAG_DATA_DIR>\\answer_cache.json.
"""
import hashlib
import json
import re
from pathlib import Path

import config
from rag.text.vi import normalize

_pg_conn = None
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)  # bỏ dấu câu: "Xin chào." == "xin chào"


def enabled() -> bool:
    return config.CACHE != "off"


def make_key(question: str, dept: str, clearance: bool) -> str:
    text = _PUNCT_RE.sub(" ", normalize(question).lower())
    base = f"{' '.join(text.split())}|{dept}|{int(clearance)}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def _corpus_sig() -> str:
    from rag.index import get_store

    return get_store().corpus_signature()


# ---------- backends ----------
def _pg():
    global _pg_conn
    if _pg_conn is None:
        import psycopg2

        _pg_conn = psycopg2.connect(config.PG_DSN)
        _pg_conn.autocommit = True
        with _pg_conn.cursor() as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS answer_cache(
                     key text PRIMARY KEY,
                     corpus_sig text NOT NULL,
                     top_score real DEFAULT 0,
                     result jsonb NOT NULL,
                     created_at timestamptz DEFAULT now())"""
            )
    return _pg_conn


def _json_path() -> Path:
    return Path(config.DATA_DIR) / "answer_cache.json"


def get(key: str):
    """Trả về (result, top_score) hoặc None. Mọi lỗi cache = miss, không chết lượt hỏi."""
    try:
        sig = _corpus_sig()
        if config.STORE == "pgvector":
            with _pg().cursor() as cur:
                cur.execute(
                    "SELECT corpus_sig, top_score, result FROM answer_cache WHERE key=%s",
                    (key,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                if row[0] != sig:  # kho đã đổi -> dọn entry cũ
                    cur.execute("DELETE FROM answer_cache WHERE key=%s", (key,))
                    return None
                return row[2], float(row[1])
        p = _json_path()
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        entry = data.get(key)
        if not entry:
            return None
        if entry["sig"] != sig:
            del data[key]
            p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return None
        return entry["result"], float(entry["top_score"])
    except Exception as e:
        print(f"[cache] lỗi đọc ({str(e)[:100]}) — bỏ qua cache")
        return None


def put(key: str, top_score: float, result: dict):
    try:
        sig = _corpus_sig()
        payload = {k: v for k, v in result.items() if k != "cached"}
        if config.STORE == "pgvector":
            with _pg().cursor() as cur:
                cur.execute(
                    """INSERT INTO answer_cache(key, corpus_sig, top_score, result)
                       VALUES (%s,%s,%s,%s::jsonb)
                       ON CONFLICT (key) DO UPDATE SET corpus_sig = EXCLUDED.corpus_sig,
                         top_score = EXCLUDED.top_score, result = EXCLUDED.result,
                         created_at = now()""",
                    (key, sig, round(top_score, 4), json.dumps(payload, ensure_ascii=False)),
                )
            return
        p = _json_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        data[key] = {"sig": sig, "top_score": round(top_score, 4), "result": payload}
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[cache] lỗi ghi ({str(e)[:100]}) — bỏ qua")
