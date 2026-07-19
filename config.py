"""MỌI lựa chọn model/tham số tập trung ở đây — đọc từ biến môi trường.

Chỉ chạy CLOUD: embedding = Gemini; LLM/Vision = Gemini hoặc OpenAI; rerank = llm|lexical.
KHÔNG còn chế độ offline (hashing giả) hay local (Ollama/bge-m3/cross-encoder HF).
Thiếu key/lỗi provider => BÁO LỖI RÕ (không fallback giả); riêng rerank/answer có dự
phòng THUẦN-PYTHON (lexical / extractive) để không rớt hẳn khi Gemini lỗi tạm.
"""
import os


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# --- Storage ---
STORE = _env("RAG_STORE", "numpy")  # numpy | pgvector
PG_DSN = _env("RAG_PG_DSN", "postgresql://postgres:123456@localhost:5432/rag")
# Số connection tối đa trong pool pgvector (rag/db.py). Mỗi request đa người dùng mượn
# 1 connection cho từng thao tác DB ngắn rồi trả ngay -> 16 dư cho dùng nội bộ.
PG_POOL_MAX = int(_env("RAG_PG_POOL_MAX", "16"))
DATA_DIR = _env("RAG_DATA_DIR", "storage")

# --- Embedding (chỉ Gemini) ---
EMBED_PROVIDER = _env("RAG_EMBED_PROVIDER", "gemini")  # gemini
EMBED_MODEL = _env("RAG_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(_env("RAG_EMBED_DIM", "1536"))

# --- LLM sinh câu trả lời (gemini | openai) ---
LLM_PROVIDER = _env("RAG_LLM_PROVIDER", "gemini")  # gemini | openai
LLM_MODEL = _env("RAG_LLM_MODEL", "gemini-flash-latest")

# --- Vision OCR cho PDF scan (gemini | openai) ---
VISION_PROVIDER = _env("RAG_VISION_PROVIDER", "gemini")  # gemini | openai
VISION_MODEL = _env("RAG_VISION_MODEL", "gemini-flash-latest")

# --- Reranker: "llm" (Gemini) | "lexical" (trùng token, thuần Python) ---
RERANKER = _env("RAG_RERANKER", "llm")

# --- Retry khi gọi Gemini/OpenAI ---
# 0 (mặc định) = KHÔNG retry, fail ngay — hợp dev/demo.
# Prod: set RAG_MAX_RETRIES=2 — chỉ retry lỗi tạm (429/500/503), không tốn thêm phí
# (request lỗi không được bill), tránh rớt câu hỏi của user vì throttle thoáng qua.
MAX_RETRIES = int(_env("RAG_MAX_RETRIES", "0"))

# --- Keys / endpoints (key KHÔNG bao giờ ghi vào file trong repo) ---
GEMINI_API_KEY = _env("GEMINI_API_KEY")
GEMINI_BASE = _env("GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta")
OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_BASE = _env("OPENAI_BASE", "https://api.openai.com/v1")

# --- Answer cache (exact-match, tự vô hiệu khi kho đổi) ---
CACHE = _env("RAG_CACHE", "on")  # on | off

# --- Giá token để Dashboard quy ra tiền: USD / 1 TRIỆU token ---
# ⚠ Đây chỉ là MẶC ĐỊNH theo bậc giá gemini-*-flash-lite tại thời điểm viết — giá nhà
# cung cấp thay đổi theo thời gian và theo model. Số token trên Dashboard là SỐ THẬT do
# API trả về, còn cột tiền chỉ đúng khi hai biến này khớp bảng giá bạn đang bị tính:
#   https://ai.google.dev/pricing   (Gemini)   |   https://openai.com/api/pricing (OpenAI)
# Dùng chung một cặp giá cho mọi lần gọi text (condense + rerank + trả lời); nếu bạn đặt
# RAG_VISION_MODEL khác bậc giá RAG_LLM_MODEL thì phần Vision sẽ lệch.
PRICE_IN = float(_env("RAG_PRICE_IN", "0.10"))
PRICE_OUT = float(_env("RAG_PRICE_OUT", "0.40"))

# --- Timeout gọi LLM/embedding (giây) ---
# Quá hạn -> lỗi rõ ràng "Quá thời gian chờ", không treo user.
TIMEOUT = int(_env("RAG_TIMEOUT", "30"))
VISION_TIMEOUT = int(_env("RAG_VISION_TIMEOUT", "120"))  # OCR 1 trang ảnh vốn lâu hơn

# --- Chat đa phiên ---
CHAT_KEEP_TURNS = 8      # số lượt gần nhất giữ NGUYÊN VĂN trong prompt (cũ hơn -> tóm tắt)
CHAT_CONDENSE_MSGS = 6   # số message gần nhất dùng để viết lại câu hỏi độc lập

# --- Tham số pipeline ---
PARENT_CHARS = 1200        # kích thước chunk cha
CHILD_CHARS = 350          # kích thước chunk con (được embed)
CHILD_OVERLAP = 60         # ký tự chồng lấn giữa 2 chunk con
PAGE_TEXT_MIN_CHARS = 200  # trang PDF >= ngưỡng này => "text"; ==0 => "scan"; giữa => "hybrid"
TOP_K_BM25 = 20
TOP_K_VECTOR = 20
RRF_K = 60                 # hằng số Reciprocal Rank Fusion
RERANK_KEEP = 5            # số chunk con giữ lại sau rerank
CONTEXT_MAX_PARENTS = 5    # số chunk cha tối đa đưa vào context

# --- Ngưỡng tự tin: van chặn rác TRƯỚC cửa LLM ---
# Điểm RRF của nguồn tốt nhất < ngưỡng -> trả thẳng "Không tìm thấy", KHÔNG gọi LLM
# (chống bịa từ gốc + tiết kiệm quota cho câu lạc đề). RRF theo hạng nên độc lập
# provider embedding; thang quan sát hiện ~0.0125–0.0328. 0 = tắt.
# LƯU Ý: 0.017 là giá trị chọn theo vùng dự kiến, CHƯA chốt bằng eval — cao quá sẽ
# từ chối oan câu hợp lệ diễn đạt khác chữ. Chỉnh/tắt qua RAG_SCORE_MIN.
SCORE_MIN = float(_env("RAG_SCORE_MIN", "0.017"))
