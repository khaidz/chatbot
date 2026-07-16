@echo off
REM ============================================================================
REM  CHE DO LOCAL 100% — khong gui bat ky du lieu nao ra ngoai (tai lieu MAT).
REM  Kho RIENG: storage_local\ (numpy) — song song kho Gemini/pgvector.
REM  Chuan bi 1 lan:
REM    winget install Ollama.Ollama
REM    ollama pull qwen2.5:7b          (LLM, ~4.7GB; may yeu: qwen2.5:3b)
REM    ollama pull qwen2.5vl:7b        (chi can neu ingest PDF scan)
REM    pip install sentence-transformers   (bge-m3 ~2GB tu tai lan dau)
REM ============================================================================

set RAG_OFFLINE=
set PYTHONUTF8=1
set GEMINI_API_KEY=

REM --- Kho: file cuc bo, tach biet voi kho Gemini ---
set RAG_STORE=numpy
set RAG_DATA_DIR=storage_local

REM --- Embedding local (bge-m3, 1024 chieu) ---
set RAG_EMBED_PROVIDER=local
set RAG_EMBED_MODEL=BAAI/bge-m3
set RAG_EMBED_DIM=1024

REM --- LLM + Vision qua Ollama ---
set RAG_LLM_PROVIDER=ollama
set RAG_LLM_MODEL=qwen2.5:7b
set RAG_VISION_PROVIDER=ollama
set RAG_VISION_MODEL=qwen2.5vl:7b

REM --- Reranker: cross-encoder (chat luong cao nhat, can HF). May cham -> llm | lexical ---
set RAG_RERANKER=BAAI/bge-reranker-v2-m3

REM --- Van hanh (timeout noi: Ollama lan dau nap model vao RAM co the >30s) ---
set RAG_MAX_RETRIES=0
set RAG_CACHE=on
set RAG_TIMEOUT=120
set RAG_VISION_TIMEOUT=120

echo [ok] LOCAL: bge-m3 (1024d) + Ollama %RAG_LLM_MODEL% - kho: storage_local\ timeout=%RAG_TIMEOUT%s
echo     Kiem tra Ollama dang chay:  ollama list
