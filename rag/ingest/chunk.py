"""Chunking parent-child.

- Cha (~1200 ký tự, cắt theo đoạn văn): đưa vào context cho LLM đọc — đủ ngữ cảnh.
- Con (~350 ký tự, cắt theo câu, overlap 60): được embed — đủ "sắc" để tìm trúng.
- ID deterministic: doc::pN::PNNN (cha), doc::pN::cNNN (con).
"""
import re

import config
from rag.schema import Chunk
from rag.text.vi import normalize

_PARA_RE = re.compile(r"\n\s*\n+")
_SENT_RE = re.compile(r"(?<=[\.\!\?\;\:])\s+|\n+")


def _pack(pieces: list[str], limit: int) -> list[str]:
    """Gộp các mảnh liên tiếp lại, mỗi khối <= limit ký tự (mảnh quá dài thì cắt cứng)."""
    blocks: list[str] = []
    cur = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        while len(piece) > limit:  # mảnh đơn quá dài -> cắt cứng
            if cur:
                blocks.append(cur)
                cur = ""
            blocks.append(piece[:limit])
            piece = piece[limit:].strip()
        if len(cur) + len(piece) + 1 <= limit:
            cur = (cur + "\n" + piece).strip()
        else:
            if cur:
                blocks.append(cur)
            cur = piece
    if cur:
        blocks.append(cur)
    return blocks


def build_chunks(
    doc_id: str,
    pages,
    dept: str = "",
    confidential: bool = False,
    source: str = "",
) -> tuple[list[Chunk], list[Chunk]]:
    """Trả về (parents, children). Chỉ children được embed."""
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for page in pages:
        text = normalize(page.text).strip()
        if not text:
            continue
        child_counter = 0  # đánh số con theo TRANG (doc::pN::cNNN)
        parent_blocks = _pack(_PARA_RE.split(text), config.PARENT_CHARS)
        for pi, ptext in enumerate(parent_blocks):
            pid = f"{doc_id}::p{page.number}::P{pi:03d}"
            parents.append(
                Chunk(pid, doc_id, page.number, ptext, "", True, dept, confidential, source)
            )
            sentences = [s for s in _SENT_RE.split(ptext) if s.strip()]
            child_blocks = _pack(sentences, config.CHILD_CHARS)
            prev_tail = ""
            for ctext in child_blocks:
                cid = f"{doc_id}::p{page.number}::c{child_counter:03d}"
                child_counter += 1
                body = (prev_tail + " " + ctext).strip() if prev_tail else ctext
                children.append(
                    Chunk(cid, doc_id, page.number, body, pid, False, dept, confidential, source)
                )
                prev_tail = ctext[-config.CHILD_OVERLAP :]
    return parents, children
