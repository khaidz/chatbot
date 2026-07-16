"""Trích text theo TRANG + phân loại trang: text | scan | hybrid (theo mật độ ký tự).

- .md/.txt: đọc thẳng, 1 "trang".
- .pdf: PyMuPDF; trang scan/hybrid => Vision OCR (render trang thành PNG, gửi VLM
  đọc thành Markdown — giữ bảng, dấu tiếng Việt, bỏ mộc/chữ ký).
- Tài liệu confidential KHÔNG BAO GIỜ gửi ảnh lên Gemini (ép về Ollama local).
"""
from dataclasses import dataclass
from pathlib import Path

import config
from rag.text.vi import normalize

_OCR_PROMPT = (
    "Đây là ảnh một trang tài liệu tiếng Việt (có thể là văn bản pháp luật, hợp đồng...). "
    "Hãy chép lại TOÀN BỘ nội dung thành Markdown: giữ nguyên dấu tiếng Việt, "
    "giữ cấu trúc bảng bằng bảng Markdown, giữ tiêu đề/điều/khoản. "
    "BỎ QUA con dấu, chữ ký, watermark. Chỉ trả về nội dung, không giải thích."
)


@dataclass
class Page:
    number: int   # 1-based
    text: str
    kind: str     # text | scan | hybrid


def extract_pages(path: str, confidential: bool = False) -> list[Page]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".txt", ".md", ".markdown"):
        text = normalize(p.read_text(encoding="utf-8", errors="replace"))
        return [Page(1, text, "text")]
    if suffix == ".pdf":
        return _extract_pdf(p, confidential)
    raise ValueError(f"Định dạng chưa hỗ trợ: {suffix} (hỗ trợ .pdf/.txt/.md)")


def _extract_pdf(p: Path, confidential: bool) -> list[Page]:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError(
            "Thiếu PyMuPDF để đọc PDF — chạy: pip install PyMuPDF"
        ) from e

    pages: list[Page] = []
    with fitz.open(p) as doc:
        for i, page in enumerate(doc, start=1):
            text = normalize(page.get_text().strip())
            density = len(text)
            if density >= config.PAGE_TEXT_MIN_CHARS:
                pages.append(Page(i, text, "text"))
                continue
            kind = "scan" if density == 0 else "hybrid"
            ocr = _ocr_page(page, confidential)
            merged = (text + "\n\n" + ocr).strip() if kind == "hybrid" else ocr
            pages.append(Page(i, normalize(merged), kind))
    return pages


def _ocr_page(page, confidential: bool) -> str:
    """Render trang -> PNG -> Vision OCR. Trả về "" nếu vision offline/lỗi."""
    from rag.generate.llm import vision_ocr

    provider = config.VISION_PROVIDER
    if confidential and provider == "gemini":
        # tài liệu MẬT: cấm gửi ảnh ra cloud — ép về Ollama local
        print("[vision] tài liệu confidential: ép Vision về ollama (không gửi Gemini)")
        provider = "ollama"
    if provider == "offline" or config.offline_forced():
        return ""
    png = page.get_pixmap(dpi=150).tobytes("png")
    try:
        return vision_ocr(png, _OCR_PROMPT, provider=provider)
    except Exception as e:
        print(f"[vision] OCR lỗi trang {page.number + 1}: {e}")
        return ""
