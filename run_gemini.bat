@echo off
REM ============================================================================
REM  CHE DO CLOUD GEMINI  (kho: PostgreSQL/pgvector, du lieu KHONG mat)
REM  Nguyen tac: bat nay set DU MOI BIEN -> chay bat nao ra dung moi truong do,
REM  khong thua ke rac tu bat chay truoc trong cung cua so cmd (su co #10).
REM  Key nen de ngoai file:  setx GEMINI_API_KEY "..."  roi mo cmd MOI.
REM ============================================================================

set RAG_OFFLINE=
set PYTHONUTF8=1

REM --- Key (dang hardcode theo lua chon cua ban — KHONG commit file nay len git) ---
set GEMINI_API_KEY=
REM --- Kho: PostgreSQL + pgvector ---
set RAG_STORE=pgvector
set RAG_PG_DSN=postgresql://postgres:123456@localhost:5432/rag
set RAG_DATA_DIR=storage

REM --- Embedding (LUU Y: doi model = phai xoa kho + ingest lai, vector 2 model
REM     khong so sanh duoc; he thong se tu chan neu lech) ---
set RAG_EMBED_PROVIDER=gemini
set RAG_EMBED_MODEL=gemini-embedding-001
set RAG_EMBED_DIM=1536

REM --- LLM + Vision (doi thoai mai, khong can re-ingest; 404 -> dung alias *-latest) ---
set RAG_LLM_PROVIDER=gemini
set RAG_LLM_MODEL=gemini-3.1-flash-lite
set RAG_VISION_PROVIDER=gemini
set RAG_VISION_MODEL=gemini-3.1-flash-lite
set RAG_RERANKER=llm

REM --- Van hanh ---
set RAG_MAX_RETRIES=0
set RAG_CACHE=on
set RAG_TIMEOUT=30
set RAG_VISION_TIMEOUT=120

if "%GEMINI_API_KEY%"=="" (
  echo [!] GEMINI_API_KEY chua duoc dat. Chay:  setx GEMINI_API_KEY "..."  roi MO CMD MOI.
) else (
  echo [ok] GEMINI: embed=%RAG_EMBED_MODEL% llm=%RAG_LLM_MODEL% kho=pgvector timeout=%RAG_TIMEOUT%s
)
