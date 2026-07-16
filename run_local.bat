@echo off
REM Che do LOCAL 100% — khong gui bat ky du lieu nao ra ngoai (tai lieu MAT dung che do nay).
REM Can chuan bi (1 lan):
REM   1) Cai Ollama:  winget install Ollama.Ollama   (hoac tai tu ollama.com)
REM   2) ollama pull qwen2.5:7b          (LLM tra loi, ~4.7GB; may yeu: qwen2.5:3b)
REM   3) ollama pull qwen2.5vl:7b        (chi can neu ingest PDF scan - Vision OCR)
REM   4) pip install sentence-transformers   (bge-m3 ~2GB tu tai tu HF lan dau)
REM
REM Kho RIENG (storage_local, numpy) - song song voi kho Gemini/pgvector, khong dung cham nhau.

set RAG_OFFLINE=
set PYTHONUTF8=1

set RAG_EMBED_PROVIDER=local
set RAG_EMBED_MODEL=BAAI/bge-m3
set RAG_EMBED_DIM=1024

set RAG_LLM_PROVIDER=ollama
set RAG_LLM_MODEL=qwen2.5:7b

REM Reranker: cross-encoder chat luong cao nhat (can HF). May cham -> doi thanh: llm (di qua Ollama) hoac lexical
set RAG_RERANKER=BAAI/bge-reranker-v2-m3

set RAG_VISION_PROVIDER=ollama
set RAG_VISION_MODEL=qwen2.5vl:7b

set RAG_STORE=numpy
set RAG_DATA_DIR=storage_local
set RAG_MAX_RETRIES=0
REM Ollama lan dau nap model vao RAM co the >30s -> noi timeout rieng cho local
set RAG_TIMEOUT=120
set RAG_VISION_TIMEOUT=120

echo [ok] Che do LOCAL: bge-m3 (1024d) + Ollama %RAG_LLM_MODEL% - kho: storage_local\
echo     Kiem tra Ollama dang chay:  ollama list
