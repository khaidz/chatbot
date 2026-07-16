"""pgvector backend — PostgreSQL + extension vector (docker pgvector/pgvector:pg16).

- Vector: cosine `<=>` + index HNSW.
- Keyword: Postgres full-text search ('simple') trên text ĐÃ segment tiếng Việt.
  Cần BM25 Okapi thật trong DB -> extension pg_search (ParadeDB), thay search_bm25.
- Chiều vector CỐ ĐỊNH khi tạo bảng (RAG_EMBED_DIM). Đổi embedding = DB/bảng mới.
"""
import numpy as np

from rag.schema import Chunk
from rag.text.vi import tokenize


class PgVectorStore:
    def __init__(self, dsn: str):
        try:
            import psycopg2
        except ImportError as e:
            raise RuntimeError(
                "Thiếu psycopg2 — chạy: pip install psycopg2-binary"
            ) from e
        from rag.embed import get_embedder

        self.dim = get_embedder().dim
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        self._init_schema()

    def _init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """CREATE TABLE IF NOT EXISTS docs(
                     sha text PRIMARY KEY, doc_id text NOT NULL, source text)"""
            )
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS chunks(
                      chunk_id text PRIMARY KEY,
                      doc_id text NOT NULL,
                      page int NOT NULL,
                      is_parent bool NOT NULL,
                      parent_id text DEFAULT '',
                      dept text DEFAULT '',
                      confidential bool DEFAULT false,
                      source text DEFAULT '',
                      text text NOT NULL,
                      tsv tsvector,
                      embedding vector({self.dim}))"""
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN(tsv)"
            )
            cur.execute(
                """CREATE INDEX IF NOT EXISTS chunks_emb_idx ON chunks
                   USING hnsw (embedding vector_cosine_ops)"""
            )
            # bảng đã tạo với dim khác -> báo ngay, không hỏng ngầm
            cur.execute(
                """SELECT atttypmod FROM pg_attribute
                   WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"""
            )
            row = cur.fetchone()
            if row and row[0] not in (-1, self.dim):
                raise RuntimeError(
                    f"Bảng chunks có vector({row[0]}) nhưng embedder dim={self.dim}. "
                    "Đổi embedding = tạo DB/bảng mới."
                )

    # ---------- write ----------
    def has_doc_sha(self, sha: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM docs WHERE sha=%s", (sha,))
            return cur.fetchone() is not None

    def add_doc(self, sha, doc_id, source, parents, children, vectors: np.ndarray):
        # transaction tường minh: ghi đủ cả doc + parents + children hoặc không gì cả
        # (autocommit=True cho DDL, nên phải BEGIN/COMMIT tay ở đây)
        with self.conn.cursor() as cur:
            try:
                cur.execute("BEGIN")
                cur.execute(
                    "INSERT INTO docs(sha, doc_id, source) VALUES (%s,%s,%s)",
                    (sha, doc_id, source),
                )
                for c in parents:
                    cur.execute(
                        """INSERT INTO chunks(chunk_id,doc_id,page,is_parent,parent_id,dept,
                             confidential,source,text,tsv,embedding)
                           VALUES (%s,%s,%s,true,'',%s,%s,%s,%s,NULL,NULL)
                           ON CONFLICT (chunk_id) DO NOTHING""",
                        (c.chunk_id, c.doc_id, c.page, c.dept, c.confidential, c.source, c.text),
                    )
                for c, v in zip(children, vectors):
                    cur.execute(
                        """INSERT INTO chunks(chunk_id,doc_id,page,is_parent,parent_id,dept,
                             confidential,source,text,tsv,embedding)
                           VALUES (%s,%s,%s,false,%s,%s,%s,%s,%s,
                                   to_tsvector('simple', %s), %s::vector)
                           ON CONFLICT (chunk_id) DO NOTHING""",
                        (
                            c.chunk_id, c.doc_id, c.page, c.parent_id, c.dept,
                            c.confidential, c.source, c.text,
                            " ".join(tokenize(c.text)),
                            "[" + ",".join(f"{x:.6f}" for x in v) + "]",
                        ),
                    )
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise

    def has_doc_id(self, doc_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM docs WHERE doc_id=%s LIMIT 1", (doc_id,))
            return cur.fetchone() is not None

    def delete_doc(self, doc_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id=%s", (doc_id,))
            cur.execute("DELETE FROM docs WHERE doc_id=%s", (doc_id,))
            return cur.rowcount > 0

    def list_docs(self) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT d.doc_id, min(d.source),
                          count(c.chunk_id) FILTER (WHERE c.is_parent),
                          count(c.chunk_id) FILTER (WHERE NOT c.is_parent)
                   FROM docs d LEFT JOIN chunks c ON c.doc_id = d.doc_id
                   GROUP BY d.doc_id ORDER BY min(d.source)"""
            )
            return [
                {"doc_id": r[0], "source": r[1], "parents": r[2], "children": r[3]}
                for r in cur.fetchall()
            ]

    # ---------- read ----------
    _RBAC = "(NOT confidential OR %(clearance)s) AND (dept = '' OR dept = %(dept)s)"

    def _rows_to_chunks(self, rows):
        return [
            (
                Chunk(r[0], r[1], r[2], r[8], r[4], r[3], r[5], r[6], r[7]),
                float(r[9]),
            )
            for r in rows
        ]

    _COLS = "chunk_id,doc_id,page,is_parent,parent_id,dept,confidential,source,text"

    def search_vector(self, qvec: np.ndarray, k: int, dept: str = "", clearance: bool = True):
        vec = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
        with self.conn.cursor() as cur:
            cur.execute(
                f"""SELECT {self._COLS}, 1 - (embedding <=> %(v)s::vector) AS score
                    FROM chunks
                    WHERE NOT is_parent AND {self._RBAC}
                    ORDER BY embedding <=> %(v)s::vector
                    LIMIT %(k)s""",
                {"v": vec, "k": k, "dept": dept, "clearance": clearance},
            )
            return self._rows_to_chunks(cur.fetchall())

    def search_bm25(self, query_tokens: list[str], k: int, dept: str = "", clearance: bool = True):
        if not query_tokens:
            return []
        tsquery = " | ".join(set(query_tokens))
        with self.conn.cursor() as cur:
            cur.execute(
                f"""SELECT {self._COLS}, ts_rank(tsv, to_tsquery('simple', %(q)s)) AS score
                    FROM chunks
                    WHERE NOT is_parent AND tsv @@ to_tsquery('simple', %(q)s) AND {self._RBAC}
                    ORDER BY score DESC
                    LIMIT %(k)s""",
                {"q": tsquery, "k": k, "dept": dept, "clearance": clearance},
            )
            return self._rows_to_chunks(cur.fetchall())

    def get_parent(self, parent_id: str) -> Chunk | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._COLS} FROM chunks WHERE chunk_id=%s", (parent_id,)
            )
            r = cur.fetchone()
        if not r:
            return None
        return Chunk(r[0], r[1], r[2], r[8], r[4], r[3], r[5], r[6], r[7])

    def stats(self) -> dict:
        with self.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM docs")
            docs = cur.fetchone()[0]
            cur.execute("SELECT count(*) FILTER (WHERE is_parent), count(*) FILTER (WHERE NOT is_parent) FROM chunks")
            parents, children = cur.fetchone()
        return {
            "backend": "pgvector",
            "docs": docs,
            "parents": parents,
            "children": children,
            "dim": self.dim,
        }
