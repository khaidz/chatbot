"""Chuẩn hoá + tách từ tiếng Việt.

- normalize: NFC (bắt buộc — cùng chữ 'ế' có 2 cách mã hoá, không NFC thì BM25 trượt).
- tokenize: ưu tiên underthesea > pyvi > regex thường (không cần cài gì).
Cùng MỘT hàm dùng cho cả lúc index lẫn lúc query — lệch nhau là retrieval hỏng ngầm.
"""
import re
import threading
import unicodedata

_backend = None  # ("underthesea"|"pyvi"|"simple", callable)
_backend_lock = threading.Lock()
_TOKEN_RE = re.compile(r"[0-9a-zà-ỹ]+", re.IGNORECASE)

# Từ chức năng phổ biến — xuất hiện ở mọi câu nên VÔ GIÁ TRỊ khi so khớp liên quan
# (dùng cho extractive/lexical scoring; BM25 tự hạ trọng số qua IDF nên không cần).
STOPWORDS = frozenset(
    "của là gì và có cho được thì mà các những một này đó kia với về theo trong "
    "trên dưới tại từ đến khi nào bao nhiêu không ai đâu sao vậy như nếu hay hoặc "
    "cũng đã sẽ đang bị do bởi ra vào lại nữa rồi ở nó họ tôi bạn anh chị em".split()
)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _resolve_backend():
    global _backend
    if _backend is not None:  # đường nhanh
        return _backend
    # underthesea/pyvi nạp model LAZY ở lần gọi đầu và KHÔNG thread-safe lúc nạp. Server
    # đa người dùng (không còn lock toàn cục) -> chọn backend + "hâm nóng" model một lần
    # dưới lock; sau khi nạp xong, predict là read-only nên gọi song song an toàn.
    with _backend_lock:
        if _backend is not None:
            return _backend
        chosen = None
        try:
            from underthesea import word_tokenize  # type: ignore

            fn = lambda s: word_tokenize(s)  # noqa: E731
            fn("khởi động")  # ép nạp model NGAY, trong lock
            chosen = ("underthesea", fn)
        except Exception:
            chosen = None
        if chosen is None:
            try:
                from pyvi import ViTokenizer  # type: ignore

                fn = lambda s: ViTokenizer.tokenize(s).split()  # noqa: E731
                fn("khởi động")
                chosen = ("pyvi", fn)
            except Exception:
                chosen = None
        _backend = chosen or ("simple", None)
        return _backend


def tokenize(text: str) -> list[str]:
    """Trả về list token lowercase, đã NFC. Từ ghép nối bằng '_' (dữ_liệu)."""
    text = normalize(text).lower()
    name, fn = _resolve_backend()
    if name == "simple":
        return _TOKEN_RE.findall(text)
    words = fn(text)
    out: list[str] = []
    for w in words:
        w = w.strip().replace(" ", "_")
        if w and _TOKEN_RE.search(w):
            out.append(w)
            # thêm cả token đơn để query không segment vẫn khớp
            if "_" in w:
                out.extend(p for p in w.split("_") if p)
    return out
