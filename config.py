"""MỌI lựa chọn model/tham số tập trung ở đây — đọc từ biến môi trường.

Xem bảng biến môi trường trong cẩm nang (mục 10).
"""
import os


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# --- Chế độ ---
OFFLINE = _env("RAG_OFFLINE")  # "force" | "" (auto-fallback) | "off" (cấm fallback)

# --- Storage ---
STORE = _env("RAG_STORE", "numpy")  # numpy | pgvector
PG_DSN = _env("RAG_PG_DSN", "postgresql://postgres:123456@localhost:5432/rag")
DATA_DIR = _env("RAG_DATA_DIR", "storage")

# --- Embedding ---
EMBED_PROVIDER = _env("RAG_EMBED_PROVIDER", "local")  # local | gemini | offline
_DEFAULT_EMBED_MODEL = {
    "local": "BAAI/bge-m3",
    "gemini": "gemini-embedding-001",
    "offline": "hashing-256",
}
_DEFAULT_EMBED_DIM = {"local": 1024, "gemini": 1536, "offline": 256}
EMBED_MODEL = _env("RAG_EMBED_MODEL", _DEFAULT_EMBED_MODEL.get(EMBED_PROVIDER, "BAAI/bge-m3"))
EMBED_DIM = int(_env("RAG_EMBED_DIM", str(_DEFAULT_EMBED_DIM.get(EMBED_PROVIDER, 1024))))
OFFLINE_EMBED_DIM = 256  # chiều của hashing embedder (giả)

# --- LLM sinh câu trả lời ---
LLM_PROVIDER = _env("RAG_LLM_PROVIDER", "offline")  # ollama | openai | gemini | offline
LLM_MODEL = _env("RAG_LLM_MODEL", "qwen2.5:7b")

# --- Vision OCR cho PDF scan ---
VISION_PROVIDER = _env("RAG_VISION_PROVIDER", "offline")  # ollama | openai | gemini | offline
VISION_MODEL = _env("RAG_VISION_MODEL", "qwen2-vl")

# --- Reranker: tên model HF (cross-encoder) | "llm" | "lexical" ---
RERANKER = _env("RAG_RERANKER", "BAAI/bge-reranker-v2-m3")

# --- Retry khi gọi Gemini ---
# 0 (mặc định) = KHÔNG retry, fail ngay — hợp dev/demo.
# Prod: set RAG_MAX_RETRIES=2 — chỉ retry lỗi tạm (429/500/503), không tốn thêm phí
# (request lỗi không được bill), tránh rớt câu hỏi của user vì throttle thoáng qua.
MAX_RETRIES = int(_env("RAG_MAX_RETRIES", "0"))

# --- Keys / endpoints (key KHÔNG bao giờ ghi vào file trong repo) ---
GEMINI_API_KEY = _env("GEMINI_API_KEY")
GEMINI_BASE = _env("GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta")
OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_BASE = _env("OPENAI_BASE", "https://api.openai.com/v1")
OLLAMA_BASE = _env("OLLAMA_BASE", "http://localhost:11434")

# --- Answer cache (exact-match, tự vô hiệu khi kho đổi) ---
CACHE = _env("RAG_CACHE", "on")  # on | off

# --- Timeout gọi LLM/embedding (giây) ---
# Quá hạn -> lỗi rõ ràng "Quá thời gian chờ", không treo user.
# Ollama local lần đầu nạp model vào RAM có thể >30s -> run_local.bat set 120.
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


def offline_forced() -> bool:
    return OFFLINE.lower() == "force"


def offline_banned() -> bool:
    """RAG_OFFLINE=off nghĩa là CẤM fallback lặng lẽ — lỗi thì phải nổ."""
    return OFFLINE.lower() == "off"
