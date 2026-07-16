@echo off
REM ⭐ Cấu hình ĐANG CHẠY ĐƯỢC: Gemini toàn tuyến (mạng công ty thông Google, chặn HF).
REM Key đặt riêng 1 lần bằng:  setx GEMINI_API_KEY "AIza..."  (rồi mở cmd MỚI)
REM `set` chỉ sống trong cửa sổ cmd hiện tại -> chạy lại file này mỗi phiên.

set RAG_OFFLINE=
set GEMINI_API_KEY=
set PYTHONUTF8=1
set RAG_EMBED_PROVIDER=gemini
REM embedding-001 het quota (429) -> dung gemini-embedding-2 (bucket quota rieng, in 8192 token)
set RAG_EMBED_MODEL=gemini-embedding-2-preview
set RAG_EMBED_DIM=1536
set RAG_LLM_PROVIDER=gemini
REM su co #6: ten ban cung (gemini-2.5-flash-lite) bi khoa voi user moi (404) -> dung ALIAS *-latest
set RAG_LLM_MODEL=gemini-3-flash-preview
set RAG_RERANKER=llm
set RAG_VISION_PROVIDER=gemini
set RAG_VISION_MODEL=gemini-flash-lite-latest

set RAG_STORE=pgvector

REM Retry loi tam (429/500/503): mac dinh 0 = KHONG retry (dev). Prod: doi thanh 2.
set RAG_MAX_RETRIES=0
set RAG_TIMEOUT=30
set RAG_VISION_TIMEOUT=120

if "%GEMINI_API_KEY%"=="" (
  echo [!] GEMINI_API_KEY chua duoc dat. Chay:  setx GEMINI_API_KEY "AIza..."  roi MO CMD MOI.
) else (
  echo [ok] Da nap cau hinh Gemini. Vi du:
  echo    python cli.py ingest examples\nd13.md examples\hopdong.md
  echo    python cli.py ask "ND 13 dinh nghia du lieu ca nhan the nao"
)
