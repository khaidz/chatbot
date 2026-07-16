"""Embedder: local (bge-m3) | gemini (gemini-embedding-001) | offline (hashing giả).

Quy tắc: một kho chỉ dùng MỘT provider. Store ghi (provider, model, dim) vào metadata
và kiểm tra mỗi lần add/search — trộn chiều vector là lỗi ngay, không hỏng ngầm.

Auto-fallback: thiếu dep/thiếu key => in "[embed] fallback OFFLINE" rồi dùng hashing.
RAG_OFFLINE=off => cấm fallback (lỗi thì nổ). RAG_OFFLINE=force => offline luôn.
"""
import hashlib

import numpy as np

import config
from rag.text.vi import tokenize

_embedder = None


class Embedder:
    def __init__(self, provider: str, model: str, dim: int):
        self.provider = provider
        self.model = model
        self.dim = dim
        self._st_model = None  # sentence-transformers, lazy

    # ---------- public ----------
    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Trả về ma trận (n, dim) float32, ĐÃ chuẩn hoá L2 (dùng cosine = dot)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.provider == "offline":
            arr = self._embed_offline(texts)
        elif self.provider == "gemini":
            arr = self._embed_gemini(texts, is_query)
        else:
            arr = self._embed_local(texts)
        return _l2_normalize(arr.astype(np.float32))

    # ---------- providers ----------
    def _embed_offline(self, texts: list[str]) -> np.ndarray:
        """Hashing embedder (GIẢ — chỉ để smoke-test luồng). Deterministic qua md5,
        KHÔNG dùng hash() builtin (bị randomize mỗi lần chạy Python)."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            # thêm 3-gram ký tự cho chuỗi số hiệu kiểu "13/2023"
            flat = "".join(toks)
            grams = [flat[j : j + 3] for j in range(max(0, len(flat) - 2))]
            for tok in toks + grams:
                idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim
                out[i, idx] += 1.0
        return out

    def _embed_gemini(self, texts: list[str], is_query: bool) -> np.ndarray:
        from rag.net import post_json

        url = (
            f"{config.GEMINI_BASE}/models/{self.model}:batchEmbedContents"
            f"?key={config.GEMINI_API_KEY}"
        )
        task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 50):
            batch = texts[i : i + 50]
            body = {
                "requests": [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task,
                        "outputDimensionality": self.dim,
                    }
                    for t in batch
                ]
            }
            r = post_json(url, body, tag="Gemini embed")  # timeout = RAG_TIMEOUT
            vectors.extend(e["values"] for e in r.json()["embeddings"])
        return np.array(vectors, dtype=np.float32)

    def _embed_local(self, texts: list[str]) -> np.ndarray:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer  # noqa: needs HF

            self._st_model = SentenceTransformer(self.model)
        return np.asarray(
            self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _fallback(reason: str) -> Embedder:
    if config.offline_banned():
        raise RuntimeError(f"Embedding provider lỗi và RAG_OFFLINE=off cấm fallback: {reason}")
    print(f"[embed] fallback OFFLINE (hashing {config.OFFLINE_EMBED_DIM}d) — lý do: {reason}")
    return Embedder("offline", "hashing-256", config.OFFLINE_EMBED_DIM)


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is not None:
        return _embedder

    if config.offline_forced() or config.EMBED_PROVIDER == "offline":
        _embedder = Embedder("offline", "hashing-256", config.OFFLINE_EMBED_DIM)
        return _embedder

    if config.EMBED_PROVIDER == "gemini":
        if not config.GEMINI_API_KEY:
            _embedder = _fallback("thiếu GEMINI_API_KEY")
        else:
            try:
                import requests  # noqa: F401
                _embedder = Embedder("gemini", config.EMBED_MODEL, config.EMBED_DIM)
            except ImportError:
                _embedder = _fallback("thiếu package 'requests' (pip install requests)")
        return _embedder

    # local (bge-m3) — cần Hugging Face; mạng công ty chặn HF thì sẽ fallback lúc gọi
    try:
        import sentence_transformers  # noqa: F401
        _embedder = Embedder("local", config.EMBED_MODEL, config.EMBED_DIM)
    except ImportError:
        _embedder = _fallback("thiếu sentence-transformers (pip install sentence-transformers)")
    return _embedder
