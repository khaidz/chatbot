"""Orchestrate ingestion: dedup(sha256) -> extract -> GATE -> chunk -> embed -> store.

GATE: tổng 0 ký tự => FAIL. KHÔNG bao giờ báo 'ingested' giả —
"Rác vào -> rác ra, kèm thái độ tự tin."
"""
import re
from pathlib import Path

from rag.embed import get_embedder
from rag.index import get_store
from rag.ingest.chunk import build_chunks
from rag.ingest.extract import extract_pages
from rag.schema import DocStatus, sha256_file

_SLUG_RE = re.compile(r"[^0-9a-zA-Zà-ỹÀ-Ỹ_-]+")


def _doc_id(path: Path) -> str:
    return _SLUG_RE.sub("-", path.stem).strip("-").lower() or "doc"


def ingest_files(paths: list[str], dept: str = "", confidential: bool = False):
    """Trả về list (path, DocStatus, message)."""
    store = get_store()
    embedder = get_embedder()
    results = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            results.append((raw, DocStatus.FAILED, "file không tồn tại"))
            print(f"[FAIL] {raw}: file không tồn tại")
            continue
        try:
            status, msg = _ingest_one(store, embedder, p, dept, confidential)
        except Exception as e:
            status, msg = DocStatus.FAILED, str(e)
        tag = {"ingested": "OK", "failed": "FAIL", "duplicate": "DUP"}[status.value]
        print(f"[{tag}] {p.name}: {msg}")
        results.append((raw, status, msg))
    return results


def _ingest_one(store, embedder, p: Path, dept: str, confidential: bool):
    sha = sha256_file(str(p))
    if store.has_doc_sha(sha):
        return DocStatus.DUPLICATE, "đã có trong kho (trùng sha256), bỏ qua"

    doc_id = _doc_id(p)
    pages = extract_pages(str(p), confidential=confidential)

    total_chars = sum(len(pg.text.strip()) for pg in pages)
    if total_chars == 0:
        # GATE — PDF scan chưa bật Vision, hoặc file rỗng
        return DocStatus.FAILED, (
            "0 ký tự trích được — nếu là PDF scan hãy bật Vision "
            "(set RAG_VISION_PROVIDER=gemini, xem cẩm nang mục 9)"
        )

    parents, children = build_chunks(doc_id, pages, dept, confidential, p.name)
    if not children:
        return DocStatus.FAILED, "không tạo được chunk nào"

    vectors = embedder.embed([c.text for c in children])
    store.add_doc(sha, doc_id, p.name, parents, children, vectors)
    kinds = ", ".join(sorted({pg.kind for pg in pages}))
    return DocStatus.INGESTED, (
        f"{len(pages)} trang ({kinds}), {len(parents)} cha / {len(children)} con, "
        f"{total_chars} ký tự"
    )
