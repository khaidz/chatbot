"""Lưu session chat. Hai backend cùng interface:

- create_session(dept, clearance) -> session_id
- get_session(sid) -> dict{session_id, dept, clearance, title, summary, summary_upto} | None
- list_sessions() -> list[dict{session_id, title, created_at, n_messages}]
- add_message(sid, role, content, sources)
- get_messages(sid) -> list[dict{role, content, sources}]
- set_title(sid, title) / update_summary(sid, summary, upto)

RBAC (dept/clearance) gắn vào SESSION lúc tạo — không truyền tay từng câu,
tránh lỗ hổng câu sau quên cờ.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------- JSON (numpy)
class JsonChatStore:
    def __init__(self, data_dir: str):
        self.dir = Path(data_dir) / "chat"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        return self.dir / f"{sid}.json"

    def _read(self, sid: str) -> dict | None:
        p = self._path(sid)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            print(f"[chat] file phiên {sid} hỏng — coi như không tồn tại")
            return None

    def _write(self, sid: str, data: dict):
        self._path(sid).write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def create_session(self, dept: str = "", clearance: bool = True) -> str:
        sid = _new_id()
        self._write(sid, {
            "session_id": sid, "dept": dept, "clearance": clearance,
            "title": "", "summary": "", "summary_upto": 0,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "messages": [],
        })
        return sid

    def get_session(self, sid: str) -> dict | None:
        d = self._read(sid)
        if d is None:
            return None
        return {k: d[k] for k in
                ("session_id", "dept", "clearance", "title", "summary", "summary_upto")}

    def list_sessions(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "session_id": d["session_id"], "title": d.get("title", ""),
                "created_at": d.get("created_at", ""), "n_messages": len(d.get("messages", [])),
            })
        return out

    def add_message(self, sid: str, role: str, content: str, sources: list):
        d = self._read(sid)
        d["messages"].append({"role": role, "content": content, "sources": sources})
        self._write(sid, d)

    def get_messages(self, sid: str) -> list[dict]:
        d = self._read(sid)
        return d["messages"] if d else []

    def set_title(self, sid: str, title: str):
        d = self._read(sid)
        d["title"] = title
        self._write(sid, d)

    def update_summary(self, sid: str, summary: str, upto: int):
        d = self._read(sid)
        d["summary"], d["summary_upto"] = summary, upto
        self._write(sid, d)

    def delete_session(self, sid: str):
        self._path(sid).unlink(missing_ok=True)


# ------------------------------------------------------------ PostgreSQL (pgvector)
class PgChatStore:
    def __init__(self, dsn: str):
        import psycopg2

        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS chat_sessions(
                     session_id text PRIMARY KEY,
                     dept text DEFAULT '',
                     clearance bool DEFAULT true,
                     title text DEFAULT '',
                     summary text DEFAULT '',
                     summary_upto int DEFAULT 0,
                     created_at timestamptz DEFAULT now())"""
            )
            cur.execute(
                """CREATE TABLE IF NOT EXISTS chat_messages(
                     id bigserial PRIMARY KEY,
                     session_id text NOT NULL
                       REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                     role text NOT NULL,
                     content text NOT NULL,
                     sources jsonb DEFAULT '[]',
                     created_at timestamptz DEFAULT now())"""
            )
            cur.execute(
                """CREATE INDEX IF NOT EXISTS chat_messages_sid_idx
                   ON chat_messages(session_id, id)"""
            )

    def create_session(self, dept: str = "", clearance: bool = True) -> str:
        sid = _new_id()
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions(session_id, dept, clearance) VALUES (%s,%s,%s)",
                (sid, dept, clearance),
            )
        return sid

    def get_session(self, sid: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT session_id, dept, clearance, title, summary, summary_upto
                   FROM chat_sessions WHERE session_id=%s""", (sid,),
            )
            r = cur.fetchone()
        if not r:
            return None
        return {"session_id": r[0], "dept": r[1], "clearance": r[2],
                "title": r[3], "summary": r[4], "summary_upto": r[5]}

    def list_sessions(self) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT s.session_id, s.title, s.created_at::text,
                          count(m.id)
                   FROM chat_sessions s
                   LEFT JOIN chat_messages m ON m.session_id = s.session_id
                   GROUP BY s.session_id, s.title, s.created_at
                   ORDER BY s.created_at"""
            )
            return [
                {"session_id": r[0], "title": r[1], "created_at": r[2], "n_messages": r[3]}
                for r in cur.fetchall()
            ]

    def add_message(self, sid: str, role: str, content: str, sources: list):
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO chat_messages(session_id, role, content, sources)
                   VALUES (%s,%s,%s,%s::jsonb)""",
                (sid, role, content, json.dumps(sources, ensure_ascii=False)),
            )

    def get_messages(self, sid: str) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT role, content, sources FROM chat_messages
                   WHERE session_id=%s ORDER BY id""", (sid,),
            )
            return [{"role": r[0], "content": r[1], "sources": r[2]} for r in cur.fetchall()]

    def set_title(self, sid: str, title: str):
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET title=%s WHERE session_id=%s", (title, sid)
            )

    def update_summary(self, sid: str, summary: str, upto: int):
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET summary=%s, summary_upto=%s WHERE session_id=%s",
                (summary, upto, sid),
            )

    def delete_session(self, sid: str):
        with self.conn.cursor() as cur:  # messages tự xoá theo (ON DELETE CASCADE)
            cur.execute("DELETE FROM chat_sessions WHERE session_id=%s", (sid,))
