@echo off
REM ============================================================================
REM  CHE DO OFFLINE — smoke-test luong, 0 dong, khong can key/mang/model.
REM  Embedding = hashing GIA (256d), tra loi = extractive. CHI de thu luong.
REM  Kho RIENG: storage_offline\ — khong dung cham kho that nao.
REM ============================================================================

set RAG_OFFLINE=force
set PYTHONUTF8=1
set GEMINI_API_KEY=

set RAG_STORE=numpy
set RAG_DATA_DIR=storage_offline

set RAG_EMBED_PROVIDER=offline
set RAG_EMBED_MODEL=hashing-256
set RAG_EMBED_DIM=256

set RAG_LLM_PROVIDER=offline
set RAG_LLM_MODEL=
set RAG_VISION_PROVIDER=offline
set RAG_VISION_MODEL=
set RAG_RERANKER=lexical

set RAG_MAX_RETRIES=0
set RAG_CACHE=on
set RAG_TIMEOUT=30
set RAG_VISION_TIMEOUT=120

echo [ok] OFFLINE (hashing 256d + extractive) - kho: storage_offline\
