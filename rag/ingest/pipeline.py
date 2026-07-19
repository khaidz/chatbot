"""Orchestrate ingestion: dedup(sha256) -> extract -> GATE -> chunk -> embed -> store.

GATE: tổng 0 ký tự => FAIL. KHÔNG bao giờ báo 'ingested' giả —
"Rác vào -> rác ra, kèm thái độ tự tin."

Báo cáo per-trang (rag/ingest/report.py, lưu DB pgvector hoặc file numpy): trang nào
OK / OCR lỗi / bỏ qua / trắng / lỗi đọc. Trang hỏng KHÔNG chặn cả tài liệu (các trang
tốt vẫn vào kho), chỉ được ghi lại để biết mà xử lý.
"""
import re
from pathlib import Path

from rag.embed import get_embedder
from rag.index import get_store
from rag.ingest import report
from rag.ingest.chunk import build_chunks
from rag.ingest.extract import extract_pages
from rag.schema import DocStatus, sha256_file

_SLUG_RE = re.compile(r"[^0-9a-zA-Zà-ỹÀ-Ỹ_-]+")

# nhãn tiếng Việt cho từng trạng thái trang (dùng trong message ingest)
STATUS_VI = {
    "ok": "OK",
    "ocr_failed": "OCR lỗi",
    "ocr_skipped": "OCR bỏ qua (Vision offline)",
    "blank": "trắng/không đọc được",
    "error": "lỗi đọc trang",
}


def _doc_id(path: Path) -> str:
    return _SLUG_RE.sub("-", path.stem).strip("-").lower() or "doc"


def _fmt_failures(bad: list[dict]) -> str:
    """Gom trang hỏng theo trạng thái: 'OCR lỗi [trang 12, 45]; trắng [trang 90]'."""
    groups: dict[str, list[int]] = {}
    for r in bad:
        groups.setdefault(r["status"], []).append(r["page"])
    parts = []
    for st, nums in groups.items():
        shown = ", ".join(map(str, nums[:12])) + (f"…(+{len(nums) - 12})" if len(nums) > 12 else "")
        parts.append(f"{STATUS_VI.get(st, st)} [trang {shown}]")
    return "; ".join(parts)


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

    # Ghi báo cáo per-trang TRƯỚC (kể cả khi sắp FAIL) -> luôn có log trang nào hỏng.
    bad = report.save(doc_id, p.name, pages)
    total_chars = sum(len(pg.text.strip()) for pg in pages)
    if total_chars == 0:
        # GATE — mọi trang đều rỗng (PDF scan chưa bật Vision, file rỗng, hoặc trang lỗi)
        detail = _fmt_failures(bad) if bad else "file rỗng"
        return DocStatus.FAILED, (
            f"0 ký tự trích được ({len(pages)} trang: {detail}) — nếu là PDF scan hãy bật "
            "Vision (set RAG_VISION_PROVIDER=gemini, xem cẩm nang mục 10)"
        )

    parents, children = build_chunks(doc_id, pages, dept, confidential, p.name)
    if not children:
        return DocStatus.FAILED, "không tạo được chunk nào"

    vectors = embedder.embed([c.text for c in children])

    # file CÙNG TÊN nhưng nội dung MỚI (sha khác) -> thay thế bản cũ, tránh trộn
    # chunk cũ-mới dưới cùng doc_id. Chỉ xoá SAU khi embed thành công (bản mới
    # fail thì bản cũ còn nguyên).
    replaced = store.has_doc_id(doc_id)
    if replaced:
        store.delete_doc(doc_id)

    store.add_doc(sha, doc_id, p.name, parents, children, vectors)
    kinds = ", ".join(sorted({pg.kind for pg in pages if pg.kind != "error"}))
    msg = (f"{len(pages)} trang ({kinds}), {len(parents)} cha / {len(children)} con, "
           f"{total_chars} ký tự")
    if bad:  # ingest một phần: trang tốt đã vào kho, nêu rõ trang nào KHÔNG đọc được
        msg += f" — ⚠ {len(bad)}/{len(pages)} trang không đọc được: {_fmt_failures(bad)}"
    if replaced:
        msg += " — đã thay thế bản cũ cùng tên"
    return DocStatus.INGESTED, msg
