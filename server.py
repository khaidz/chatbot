"""Web UI cho chat đa phiên.

Chạy:  run_gemini.bat  (nạp cấu hình)  rồi  python server.py  ->  http://localhost:8000

API (frontend web/index.html dùng, cũng gọi được từ app khác):
  GET    /api/sessions              danh sách phiên
  POST   /api/sessions              tạo phiên  {dept?, clearance?}
  GET    /api/sessions/{sid}        thông tin phiên + toàn bộ messages
  POST   /api/sessions/{sid}/messages   hỏi  {question}  -> answer/sources/grounded/mode
  DELETE /api/sessions/{sid}        xoá phiên (messages xoá theo)
  GET    /api/documents             danh sách tài liệu + trạng thái (processing/ingested/failed/duplicate)
  POST   /api/documents             upload multipart (file, dept?, confidential?) — ingest chạy nền
  DELETE /api/documents/{doc_id}    xoá tài liệu khỏi kho (chunks + vectors)

Upload gốc lưu ở uploads\; trạng thái ingest persist ở <RAG_DATA_DIR>\ingest_status.json.

Đa người dùng: KHÔNG còn lock toàn cục. Kho pgvector đi qua connection pool chung
(rag/db.py), latency đo bằng bộ nhớ thread-local (rag/timing.py), các singleton
(store/embedder/chat store) tạo một lần qua double-checked lock rồi chỉ ĐỌC. Nhờ đó
hai người hỏi song song thật — call LLM/embedding chậm không giữ khoá nào. Backend
numpy (dev) tự khoá bộ nhớ trong ở tầng store. Chỉ còn _status_lock cho file trạng
thái ingest.
"""
import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import config
from rag.chat import get_chat_store
from rag.chat.pipeline import chat_turn, chat_turn_stream
from rag.index import get_store
from rag.ingest.pipeline import _doc_id, ingest_files

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def _warmup():
    """Tạo sẵn các singleton MỘT LẦN lúc khởi động, trước khi nhận request đồng thời —
    tránh nhiều luồng cùng lazy-init (và để lỗi kết nối DB nổ ngay lúc start, không
    phải ở request đầu tiên)."""
    from rag.embed import get_embedder
    from rag.text.vi import tokenize

    get_embedder()
    get_store()
    get_chat_store()
    tokenize("khởi động")  # nạp model underthesea/pyvi single-thread trước khi phục vụ


def _shutdown():
    if config.STORE == "pgvector":
        from rag import db

        db.close_pool()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warmup()
    yield
    _shutdown()


app = FastAPI(title="RAG Chatbot tiếng Việt", lifespan=lifespan)
WEB_DIR = Path(__file__).parent / "web"
UPLOAD_DIR = Path(__file__).parent / "uploads"
ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md", ".markdown"}
_status_lock = threading.RLock()  # RLock: _set_status giữ lock rồi gọi _load_status


# ---------- trạng thái ingest (persist ra file, sống qua restart) ----------
# Đọc/ghi đều qua _status_lock + ghi ATOMIC (file tạm -> os.replace) — thread ingest
# nền ghi trong lúc UI poll GET /api/documents sẽ không bao giờ đọc trúng file dở.
def _status_path() -> Path:
    return Path(config.DATA_DIR) / "ingest_status.json"


def _load_status() -> dict:
    with _status_lock:
        p = _status_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8-sig") or "{}")
        except json.JSONDecodeError:
            print("[status] ingest_status.json hỏng (ghi dở từ lần crash trước?) — coi như rỗng")
            return {}


def _write_status(data: dict):
    p = _status_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)  # atomic trên cùng ổ đĩa


def _set_status(doc_id: str, filename: str, status: str, message: str):
    with _status_lock:
        data = _load_status()
        data[doc_id] = {
            "filename": filename, "status": status, "message": message,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        _write_status(data)


def _drop_status(doc_id: str):
    with _status_lock:
        data = _load_status()
        if doc_id in data:
            del data[doc_id]
            _write_status(data)


def _ingest_background(path: Path, dept: str, confidential: bool, doc_id: str):
    try:
        # embed (chậm) chạy song song với các request khác; store tự khoá phần ghi ngắn.
        results = ingest_files([str(path)], dept=dept, confidential=confidential)
        _, st, msg = results[0]
        _set_status(doc_id, path.name, st.value, msg)
    except Exception as e:
        _set_status(doc_id, path.name, "failed", str(e)[:300])


class NewSession(BaseModel):
    dept: str = ""
    clearance: bool = True


class Question(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/sessions")
def list_sessions():
    return get_chat_store().list_sessions()


@app.post("/api/sessions")
def create_session(body: NewSession):
    sid = get_chat_store().create_session(dept=body.dept, clearance=body.clearance)
    return {"session_id": sid}


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    store = get_chat_store()
    sess = store.get_session(sid)
    if sess is None:
        raise HTTPException(404, f"Không tìm thấy session '{sid}'")
    return {**sess, "messages": store.get_messages(sid), "score_min": config.SCORE_MIN}


@app.post("/api/sessions/{sid}/messages")
def post_message(sid: str, body: Question):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Câu hỏi rỗng")
    try:
        return chat_turn(get_chat_store(), sid, q)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Lỗi xử lý: {str(e)[:300]}")


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    store = get_chat_store()
    if store.get_session(sid) is None:
        raise HTTPException(404, f"Không tìm thấy session '{sid}'")
    store.delete_session(sid)
    return {"ok": True}


@app.post("/api/sessions/{sid}/messages/stream")
def post_message_stream(sid: str, body: Question):
    """SSE: data: {"type":"delta","text":...} nhiều lần, kết thúc {"type":"result",...}."""
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Câu hỏi rỗng")

    def _sse(obj: dict) -> str:
        return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

    def gen():
        # Không giữ lock nào qua vòng đời generator -> nhiều người stream song song thật.
        store = get_chat_store()
        if store.get_session(sid) is None:
            yield _sse({"type": "error", "detail": f"Không tìm thấy session '{sid}'"})
            return
        try:
            for kind, payload in chat_turn_stream(store, sid, q):
                if kind == "delta":
                    yield _sse({"type": "delta", "text": payload})
                else:
                    yield _sse({"type": "result", **payload})
        except Exception as e:
            yield _sse({"type": "error", "detail": str(e)[:300]})

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------- dashboard ----------------
def _score_dist(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0, "min": None, "avg": None, "max": None}
    return {"n": len(scores), "min": round(min(scores), 4),
            "avg": round(sum(scores) / len(scores), 4), "max": round(max(scores), 4)}


@app.get("/api/stats")
def dashboard_stats():
    """Số liệu cho Dashboard: kho + tổng hợp query log (cùng logic `python cli.py log`)."""
    from rag.embed import get_embedder
    from rag.querylog import read_log

    store = get_store()
    emb = get_embedder()
    store_stats = store.stats()

    try:
        rows = read_log()
    except Exception as e:
        print(f"[stats] không đọc được query log ({str(e)[:100]}) — coi như rỗng")
        rows = []

    total = len(rows)
    answered = [r for r in rows if (r.get("n_sources") or 0) > 0]
    notfound = [r for r in rows if (r.get("n_sources") or 0) == 0]
    cached = [r for r in rows if "+cache" in (r.get("mode") or "")]
    grounded = [r for r in rows if r.get("grounded")]
    fresh = [r for r in rows if "+cache" not in (r.get("mode") or "")]

    modes: dict[str, int] = {}
    for r in rows:
        m = r.get("mode") or "(none)"
        modes[m] = modes.get(m, 0) + 1

    def _avg(col: str) -> int:
        vals = [r.get(col, 0) or 0 for r in fresh]
        return round(sum(vals) / len(vals)) if vals else 0

    ans_scores = [r["top_score"] for r in answered if r.get("top_score") is not None]
    nf_scores = [r["top_score"] for r in notfound if r.get("top_score") is not None]
    gap = None  # khoảng "tách" hai nhóm để gợi ý ngưỡng tự tin
    if ans_scores and nf_scores:
        lo, hi = max(nf_scores), min(ans_scores)
        gap = {"lo": round(lo, 4), "hi": round(hi, 4), "separated": lo < hi}

    recent = [
        {k: r.get(k) for k in
         ("ts", "question", "standalone", "top_score", "n_sources",
          "grounded", "mode", "total_ms", "tok_in", "tok_out", "cost_usd", "session_id")}
        for r in rows[-25:][::-1]
    ]

    # --- token + tiền: cộng dồn toàn bộ log; lượt cache = 0 token nên "tiết kiệm" tính
    # bằng chi phí TRUNG BÌNH của một lượt tươi nhân số lượt trúng cache.
    tok_in = sum(r.get("tok_in", 0) or 0 for r in rows)
    tok_out = sum(r.get("tok_out", 0) or 0 for r in rows)
    spent = sum(r.get("cost_usd", 0) or 0 for r in rows)
    priced = [r for r in fresh if (r.get("tok_in", 0) or 0) > 0]
    avg_cost = (sum(r.get("cost_usd", 0) or 0 for r in priced) / len(priced)) if priced else 0.0

    return {
        "corpus": {
            "backend": store_stats.get("backend", ""),
            "docs": store_stats.get("docs", 0),
            "parents": store_stats.get("parents", 0),
            "children": store_stats.get("children", 0),
            "dim": store_stats.get("dim", 0),
            "embedder": f"{emb.provider}/{emb.model}",
        },
        "queries": {
            "total": total,
            "answered": len(answered),
            "notfound": len(notfound),
            "grounded": len(grounded),
            "cache_hits": len(cached),
            "grounded_rate": round(len(grounded) / total, 3) if total else None,
            "cache_rate": round(len(cached) / total, 3) if total else None,
        },
        "modes": modes,
        "tokens": {
            "tok_in": tok_in,
            "tok_out": tok_out,
            "total": tok_in + tok_out,
            "cost_usd": round(spent, 6),
            "avg_cost_usd": round(avg_cost, 6),
            "avg_tok_in": round(sum(r.get("tok_in", 0) or 0 for r in priced) / len(priced)) if priced else 0,
            "avg_tok_out": round(sum(r.get("tok_out", 0) or 0 for r in priced) / len(priced)) if priced else 0,
            "n_priced": len(priced),          # số lượt THẬT SỰ gọi API (có token)
            "saved_usd": round(avg_cost * len(cached), 6),   # ước tính cache tiết kiệm
            "price_in": config.PRICE_IN,      # USD / 1 triệu token — để UI ghi rõ giả định
            "price_out": config.PRICE_OUT,
        },
        "latency_ms": {
            "n_fresh": len(fresh), "n_cache": len(cached),
            "retrieve": _avg("retrieve_ms"), "rerank": _avg("rerank_ms"),
            "llm": _avg("llm_ms"), "total": _avg("total_ms"),
        },
        "scores": {
            "answered": _score_dist(ans_scores),
            "notfound": _score_dist(nf_scores),
            "score_min": config.SCORE_MIN,
            "gap": gap,
        },
        "recent": recent,
    }


# ---------------- documents ----------------
@app.get("/api/documents")
def list_documents():
    in_store = {d["doc_id"]: d for d in get_store().list_docs()}
    status = _load_status()
    out = []
    for doc_id, d in in_store.items():
        st = status.get(doc_id, {})
        out.append({**d, "status": "ingested",
                    "message": st.get("message", ""), "ts": st.get("ts", "")})
    for doc_id, st in status.items():  # đang xử lý / lỗi — chưa (không) vào kho
        if doc_id not in in_store and st.get("status") != "ingested":
            out.append({"doc_id": doc_id, "source": st.get("filename", doc_id),
                        "parents": 0, "children": 0, **st})
    out.sort(key=lambda d: (d.get("ts") or "", d["source"].lower()))
    return out


@app.post("/api/documents")
def upload_document(
    file: UploadFile = File(...),
    dept: str = Form(""),
    confidential: str = Form("false"),
):
    filename = Path(file.filename or "").name  # chặn path traversal
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Chỉ nhận {', '.join(sorted(ALLOWED_EXT))} — nhận được '{ext}'")
    UPLOAD_DIR.mkdir(exist_ok=True)
    dest = UPLOAD_DIR / filename
    dest.write_bytes(file.file.read())

    doc_id = _doc_id(dest)
    _set_status(doc_id, filename, "processing", "đang trích xuất + embed...")
    threading.Thread(
        target=_ingest_background,
        args=(dest, dept.strip(), confidential.lower() == "true", doc_id),
        daemon=True,
    ).start()
    return {"doc_id": doc_id, "status": "processing"}


@app.get("/api/documents/{doc_id}/report")
def document_report(doc_id: str):
    """Báo cáo trích xuất per-trang: trang nào OK / OCR lỗi / bỏ qua / trắng / lỗi đọc."""
    from rag.ingest import report

    rep = report.load(doc_id)
    if rep is None:
        raise HTTPException(404, "Chưa có báo cáo trích xuất cho tài liệu này "
                                 "(ingest trước khi thêm tính năng này thì không có).")
    return rep


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    from rag.ingest import report

    found = get_store().delete_doc(doc_id)
    had_status = doc_id in _load_status()
    _drop_status(doc_id)
    report.delete(doc_id)  # dọn báo cáo per-trang kèm theo
    if not (found or had_status):
        raise HTTPException(404, f"Không tìm thấy tài liệu '{doc_id}'")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    print("Web UI: http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
