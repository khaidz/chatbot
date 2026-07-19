"""Nhận diện smalltalk / chitchat để CẮT trước khi vào RAG.

Ý tưởng: những câu như "cảm ơn", "ok", "?" không phải truy vấn tri thức —
đưa vào retrieve() chỉ tốn LLM + trả NOT_FOUND. Bắt chúng bằng heuristic rẻ,
trả câu đáp mẫu, KHÔNG chạm pgvector.

Nguyên tắc chống nhận nhầm: chỉ coi là smalltalk khi TOÀN BỘ câu (sau khi bỏ
các từ đệm thuần trang trí) khớp đúng một cụm đã biết. Câu có lẫn nội dung thật
("cảm ơn, cho tôi hỏi về nghỉ phép") vẫn còn từ thừa -> KHÔNG khớp -> đi RAG.
"""
import re
import unicodedata
from typing import Optional

# ----- Cụm theo intent (bản có dấu). Bản không dấu tự sinh ở dưới. -----
_PHRASES: dict[str, tuple[str, ...]] = {
    "greet": (
        "xin chào", "chào", "chào bạn", "chào anh", "chào chị", "chào em",
        "chào ad", "alo", "chào buổi sáng", "chào buổi trưa",
        "chào buổi chiều", "chào buổi tối",
        "hello", "helo", "hi", "hey", "good morning",
        "good afternoon", "good evening",
    ),
    "thanks": (
        "cảm ơn", "cám ơn", "cảm ơn nhé", "cảm ơn bạn", "cảm ơn anh",
        "cảm ơn chị", "cảm ơn em", "xin cảm ơn", "đa tạ",
        "thank", "thank you", "thanks", "thx", "tks", "ty", "tysm",
    ),
    "bye": (
        "tạm biệt", "chào nhé", "hẹn gặp lại", "hẹn gặp lại sau",
        "gặp lại sau", "chúc ngủ ngon",
        "bye", "goodbye", "see you", "see ya", "see you later",
        "see you soon", "take care", "good night",
    ),
    # ----- intent mới: không nên đi RAG -----
    "ok": (
        "ok", "okay", "oke", "okey", "oki", "okie", "uk",
        "được", "được rồi", "ổn", "đồng ý", "được đấy",
    ),
    "affirm": (
        "vâng", "dạ", "ừ", "uh", "đúng", "đúng rồi", "đúng vậy",
        "chuẩn", "phải rồi", "yes", "yep", "yeah",
    ),
    "negative": (
        "không", "ko", "không phải", "chưa", "chưa đâu", "đâu có",
        "no", "nope",
    ),
    "confused": (
        "hả", "gì", "gì vậy", "gì thế", "gì cơ", "sao cơ", "ơ", "ủa",
        "what", "huh",
    ),
}

# Câu đáp mẫu cho từng intent.
REPLIES: dict[str, str] = {
    "greet": "Xin chào! Tôi có thể giúp gì cho bạn?",
    "thanks": "Rất vui được hỗ trợ bạn! Cần gì thêm bạn cứ hỏi nhé.",
    "bye": "Tạm biệt! Hẹn gặp lại bạn.",
    "ok": "Vâng. Bạn cần hỏi gì thêm cứ nhắn nhé.",
    "affirm": "Dạ vâng. Bạn muốn tìm hiểu thêm điều gì không?",
    "negative": "Được rồi. Nếu cần gì bạn cứ hỏi nhé.",
    "confused": "Xin lỗi nếu câu trả lời chưa rõ. Bạn có thể hỏi lại cụ thể hơn giúp tôi không?",
}

# Từ đệm THUẦN trang trí -> bỏ đi trước khi so khớp. KHÔNG chứa từ nào là
# "mỏ neo" của một intent (vâng/dạ/ừ/ok/không...) — nếu không sẽ bị bào rỗng.
_STRIP_FILLER: frozenset[str] = frozenset({
    "bạn", "mình", "tôi", "tao", "tớ", "anh", "chị", "em",
    "ạ", "à", "nhé", "nha", "nhá", "ha", "hen", "ơi",
    "rất", "nhiều", "quá", "lắm",
    "you", "very", "much", "so", "please",
})


def _strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _normalize(text: str, *, deaccent: bool = False) -> str:
    """Thường hoá: chữ thường, bỏ dấu câu, gộp khoảng trắng. Giữ chữ + số."""
    t = text.lower().strip()
    if deaccent:
        t = _strip_accents(t)
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)  # dấu câu -> khoảng trắng
    return re.sub(r"\s+", " ", t).strip()


def _strip_filler(norm: str, *, deaccent: bool = False) -> str:
    filler = _STRIP_FILLER if not deaccent else _FILLER_DA
    kept = [tok for tok in norm.split() if tok not in filler]
    return " ".join(kept)


# ----- Bảng tra cứu dựng sẵn lúc import: cụm chuẩn hoá -> intent -----
def _build_lookup(deaccent: bool) -> dict[str, str]:
    table: dict[str, str] = {}
    for intent, phrases in _PHRASES.items():
        for p in phrases:
            key = _normalize(p, deaccent=deaccent)
            table.setdefault(key, intent)  # cụm khai báo trước thắng nếu trùng
    return table


_LOOKUP = _build_lookup(deaccent=False)
_LOOKUP_DA = _build_lookup(deaccent=True)
_FILLER_DA = frozenset(_strip_accents(w) for w in _STRIP_FILLER)

_ONLY_QMARK = re.compile(r"^[?？\s]+$")


def detect(query: str) -> Optional[str]:
    """Trả về tên intent smalltalk, hoặc None nếu là truy vấn thật."""
    if _ONLY_QMARK.match(query):  # "?", "??", "？" -> confused
        return "confused"

    norm = _normalize(query)
    if not norm:
        return None

    # Có dấu: khớp nguyên câu, rồi khớp sau khi bỏ từ đệm.
    for cand in (norm, _strip_filler(norm)):
        if cand in _LOOKUP:
            return _LOOKUP[cand]

    # Không dấu (gõ vội "cam on ban", "khong"): lặp lại quy trình trên.
    da = _normalize(query, deaccent=True)
    for cand in (da, _strip_filler(da, deaccent=True)):
        if cand in _LOOKUP_DA:
            return _LOOKUP_DA[cand]

    return None


def reply(query: str) -> Optional[str]:
    """Câu đáp mẫu nếu là smalltalk, ngược lại None."""
    intent = detect(query)
    return REPLIES[intent] if intent else None
