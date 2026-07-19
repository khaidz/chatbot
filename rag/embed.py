"""Embedder — CHỈ Gemini (gemini-embedding-001).

Quy tắc: một kho chỉ dùng MỘT provider/model/dim. Store ghi (provider, model, dim) vào
metadata và kiểm tra mỗi lần add/search — trộn chiều vector là lỗi ngay, không hỏng ngầm.
Thiếu GEMINI_API_KEY hoặc lỗi gọi API => BÁO LỖI RÕ (không còn fallback offline giả).
"""
import threading

import numpy as np

import config

_embedder = None
_embedder_lock = threading.Lock()


class Embedder:
    def __init__(self, provider: str, model: str, dim: int):
        self.provider = provider
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Trả về ma trận (n, dim) float32, ĐÃ chuẩn hoá L2 (dùng cosine = dot)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        arr = self._embed_gemini(texts, is_query)
        return _l2_normalize(arr.astype(np.float32))

    def _embed_gemini(self, texts: list[str], is_query: bool) -> np.ndarray:
        from rag.net import post_json

        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "Thiếu GEMINI_API_KEY (setx GEMINI_API_KEY \"AIza...\" rồi mở cmd MỚI)"
            )
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


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is not None:  # đường nhanh: đã tạo -> khỏi lock
        return _embedder
    with _embedder_lock:  # double-checked: đa người dùng đồng thời chỉ tạo MỘT embedder
        if _embedder is not None:
            return _embedder
        if config.EMBED_PROVIDER != "gemini":
            raise RuntimeError(
                f"Chỉ hỗ trợ embedding Gemini — RAG_EMBED_PROVIDER='{config.EMBED_PROVIDER}' "
                "không hợp lệ. Đặt RAG_EMBED_PROVIDER=gemini."
            )
        _embedder = Embedder("gemini", config.EMBED_MODEL, config.EMBED_DIM)
    return _embedder
