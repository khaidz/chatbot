"""Kiểu dữ liệu lõi. chunk_id DETERMINISTIC (doc::pN::cNNN) — quyết định khó đảo ngược #2:
re-index cùng tài liệu ra cùng id => citation cũ không chết.
"""
import hashlib
from dataclasses import dataclass, asdict, field
from enum import Enum


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class DocStatus(str, Enum):
    INGESTED = "ingested"
    FAILED = "failed"       # GATE: 0 ký tự => KHÔNG bao giờ báo 'ingested' giả
    DUPLICATE = "duplicate"  # dedup theo sha256


@dataclass
class Chunk:
    chunk_id: str            # deterministic: doc::pN::cNNN (con) / doc::pN::PNNN (cha)
    doc_id: str
    page: int
    text: str
    parent_id: str = ""      # rỗng nếu là chunk cha
    is_parent: bool = False
    dept: str = ""           # RBAC: phòng ban ("" = công khai)
    confidential: bool = False
    source: str = ""         # tên file gốc
    score: float = 0.0       # điểm liên quan lúc RETRIEVE (RRF) — KHÔNG lưu vào kho

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("score")  # transient — chỉ có nghĩa trong 1 lần truy vấn
        return d

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(**d)

    def visible_to(self, dept: str, clearance: bool) -> bool:
        """RBAC ở tầng retrieval (lọc TRONG query) — quyết định khó đảo ngược #3."""
        if self.confidential and not clearance:
            return False
        if self.dept and self.dept != dept:
            return False
        return True
