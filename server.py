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

Lưu ý: store/embedder là singleton chưa thread-safe -> mọi endpoint serialize qua
một lock. Đủ cho nội bộ/single-user; nhiều user đồng thời thì chuyển sang
connection pool + worker riêng (chưa cần bây giờ).
"""
import json
import sys
import threading
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

app = FastAPI(title="RAG Chatbot tiếng Việt")
WEB_DIR = Path(__file__).parent / "web"
UPLOAD_DIR = Path(__file__).parent / "uploads"
ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md", ".markdown"}
_lock = threading.Lock()
_status_lock = threading.Lock()


# ---------- trạng thái ingest (persist ra file, sống qua restart) ----------
def _status_path() -> Path:
    return Path(config.DATA_DIR) / "ingest_status.json"


def _load_status() -> dict:
    p = _status_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _set_status(doc_id: str, filename: str, status: str, message: str):
    with _status_lock:
        data = _load_status()
        data[doc_id] = {
            "filename": filename, "status": status, "message": message,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        p = _status_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _drop_status(doc_id: str):
    with _status_lock:
        data = _load_status()
        if doc_id in data:
            del data[doc_id]
            _status_path().write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
            )


def _ingest_background(path: Path, dept: str, confidential: bool, doc_id: str):
    try:
        with _lock:
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
    with _lock:
        return get_chat_store().list_sessions()


@app.post("/api/sessions")
def create_session(body: NewSession):
    with _lock:
        sid = get_chat_store().create_session(dept=body.dept, clearance=body.clearance)
    return {"session_id": sid}


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    with _lock:
        store = get_chat_store()
        sess = store.get_session(sid)
        if sess is None:
            raise HTTPException(404, f"Không tìm thấy session '{sid}'")
        return {**sess, "messages": store.get_messages(sid)}


@app.post("/api/sessions/{sid}/messages")
def post_message(sid: str, body: Question):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Câu hỏi rỗng")
    with _lock:
        try:
            return chat_turn(get_chat_store(), sid, q)
        except ValueError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(500, f"Lỗi xử lý: {str(e)[:300]}")


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    with _lock:
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
        _lock.acquire()  # generator sống qua nhiều chunk -> giữ lock đến khi xong
        try:
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
        finally:
            _lock.release()

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------- documents ----------------
@app.get("/api/documents")
def list_documents():
    with _lock:
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


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    with _lock:
        found = get_store().delete_doc(doc_id)
    had_status = doc_id in _load_status()
    _drop_status(doc_id)
    if not (found or had_status):
        raise HTTPException(404, f"Không tìm thấy tài liệu '{doc_id}'")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    print("Web UI: http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
