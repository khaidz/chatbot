"""Numpy backend — kho file cục bộ trong RAG_DATA_DIR (mặc định storage\\).

Files: meta.json (embed provider/model/dim — quyết định khó đảo ngược #1),
docs.json (sha256 -> doc, dedup), parents.jsonl, children.jsonl, vectors.npy.
"""
import json
from pathlib import Path

import numpy as np

from rag.index.bm25 import BM25Okapi
from rag.schema import Chunk
from rag.text.vi import tokenize


class NumpyStore:
    def __init__(self, data_dir: str):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta = self._read_json("meta.json", {})
        self._docs = self._read_json("docs.json", {})
        self._parents: dict[str, Chunk] = {
            c.chunk_id: c for c in self._read_chunks("parents.jsonl")
        }
        self._children: list[Chunk] = self._read_chunks("children.jsonl")
        vec_path = self.dir / "vectors.npy"
        self._vectors = np.load(vec_path) if vec_path.exists() else None
        self._bm25 = None  # lazy

    # ---------- io ----------
    def _read_json(self, name: str, default):
        p = self.dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    def _write_json(self, name: str, obj):
        (self.dir / name).write_text(
            json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def _read_chunks(self, name: str) -> list[Chunk]:
        p = self.dir / name
        if not p.exists():
            return []
        return [
            Chunk.from_dict(json.loads(line))
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _append_chunks(self, name: str, chunks: list[Chunk]):
        with open(self.dir / name, "a", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    # ---------- write ----------
    def has_doc_sha(self, sha: str) -> bool:
        return sha in self._docs

    def add_doc(self, sha, doc_id, source, parents, children, vectors: np.ndarray):
        from rag.embed import get_embedder

        emb = get_embedder()
        if self._meta:
            if (
                self._meta.get("dim") != emb.dim
                or self._meta.get("provider") != emb.provider
                or self._meta.get("model") != emb.model  # đổi model = vector không so sánh được
            ):
                raise RuntimeError(
                    f"Kho hiện có embedding {self._meta.get('provider')}/"
                    f"{self._meta.get('model')} dim={self._meta.get('dim')} nhưng đang chạy "
                    f"{emb.provider}/{emb.model} dim={emb.dim}. Một kho chỉ dùng MỘT provider "
                    f"— xoá kho (rmdir /s /q {self.dir}) rồi ingest lại."
                )
        else:
            self._meta = {"provider": emb.provider, "model": emb.model, "dim": emb.dim}
            self._write_json("meta.json", self._meta)

        self._docs[sha] = {"doc_id": doc_id, "source": source}
        self._write_json("docs.json", self._docs)
        self._append_chunks("parents.jsonl", parents)
        self._append_chunks("children.jsonl", children)
        for c in parents:
            self._parents[c.chunk_id] = c
        self._children.extend(children)
        self._vectors = (
            vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        )
        np.save(self.dir / "vectors.npy", self._vectors)
        self._bm25 = None  # đánh dấu build lại

    # ---------- read ----------
    def _visibility_mask(self, dept: str, clearance: bool) -> list[bool]:
        return [c.visible_to(dept, clearance) for c in self._children]

    def search_vector(self, qvec: np.ndarray, k: int, dept: str = "", clearance: bool = True):
        """RBAC lọc TRONG query (mask trước khi xếp hạng). Trả về list (Chunk, score)."""
        if self._vectors is None or not len(self._children):
            return []
        if self._vectors.shape[1] != qvec.shape[0]:
            raise RuntimeError(
                f"Chiều vector kho ({self._vectors.shape[1]}) khác chiều query "
                f"({qvec.shape[0]}) — trộn kho khác provider. Xoá storage rồi ingest lại."
            )
        scores = self._vectors @ qvec  # đã normalize => dot = cosine
        mask = np.array(self._visibility_mask(dept, clearance))
        scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(self._children[i], float(scores[i])) for i in top if scores[i] > -np.inf]

    def search_bm25(self, query_tokens: list[str], k: int, dept: str = "", clearance: bool = True):
        if not self._children:
            return []
        if self._bm25 is None:
            self._bm25 = BM25Okapi([tokenize(c.text) for c in self._children])
        mask = self._visibility_mask(dept, clearance)
        return [(self._children[i], s) for i, s in self._bm25.top_k(query_tokens, k, mask)]

    def get_parent(self, parent_id: str) -> Chunk | None:
        return self._parents.get(parent_id)

    def stats(self) -> dict:
        return {
            "backend": "numpy (file cục bộ)",
            "path": str(self.dir.resolve()),
            "docs": len(self._docs),
            "parents": len(self._parents),
            "children": len(self._children),
            "dim": self._meta.get("dim", 0),
            "embed": f"{self._meta.get('provider','-')}/{self._meta.get('model','-')}",
        }
