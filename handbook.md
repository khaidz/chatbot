# Cẩm nang triển khai RAG chatbot (từ đầu → chạy được)

Tài liệu này gộp toàn bộ quá trình: kiến trúc, quyết định, các bước cài đặt, cấu hình đang
chạy được, và **nhật ký xử lý mọi lỗi đã gặp**. Mang file này + thư mục code là dựng lại được y hệt.

Môi trường tham chiếu: **Windows 11, Python 3.11, cmd**. Chỉ chạy CLOUD (Gemini/OpenAI) — đã bỏ mọi provider offline/local.

---

## Mục lục
1. [Mục tiêu & triết lý](#1-mục-tiêu--triết-lý)
2. [Kiến trúc 4 tầng](#2-kiến-trúc-4-tầng)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài từ đầu (venv + deps)](#4-cài-từ-đầu-venv--deps)
5. [Một chế độ: Cloud](#5-một-chế-độ-cloud-gemini-hoặc-openai-cho-llmvision)
6. [⭐ Cấu hình Gemini ĐANG CHẠY ĐƯỢC (mạng công ty)](#6--cấu-hình-gemini-đang-chạy-được-mạng-công-ty)
7. [On-prem / local — ĐÃ GỠ](#7--on-prem--local--đã-gỡ)
8. [Lưu vào database (pgvector)](#8-lưu-vào-database-pgvector)
9. [Reranker — 3 chế độ](#9-reranker--3-chế-độ)
10. [Vision OCR cho PDF scan](#10-vision-ocr-cho-pdf-scan)
11. [Bảng biến môi trường](#11-bảng-biến-môi-trường)
12. [Lệnh CLI + Web](#12-lệnh-cli--web)
13. [📓 Nhật ký xử lý sự cố](#13--nhật-ký-xử-lý-sự-cố)
14. [Quyết định khó đảo ngược](#14-quyết-định-khó-đảo-ngược)
15. [Checklist dựng lại nhanh](#15-checklist-dựng-lại-nhanh)

---

## 1. Mục tiêu & triết lý

Xây RAG chatbot đọc tài liệu (nghị định, thông tư, hợp đồng...) tiếng Việt, trả lời **có trích nguồn**, **không bịa**.

Nguyên tắc cốt lõi (rút từ kinh nghiệm thực chiến):
- **80% chất lượng nằm ở ingestion + hybrid search + rerank + citation**, không phải ở model to.
- "Rác vào → rác ra, kèm thái độ tự tin." Ingestion bẩn thì retrieval xịn mấy cũng vô nghĩa.
- **Model nhỏ chuyên dụng ở mọi tầng giữa, model to chỉ ở tầng cuối.**
- Tầng nâng cao (multi-hop, self-reflection, NLI, agentic) là **thuốc theo triệu chứng** — chỉ bật khi
  eval set chứng minh cần. Build cả bộ "cho đủ" = cái bẫy 6 tháng.
- **Điều kiện tiên quyết trước khi tối ưu: eval set 50-100 câu có đáp án + nguồn.**
  (Hiện eval mới ở dạng nháp `evalset_draft.json`, chưa duyệt, chưa chạy lần nào — xem `Todo.md` mục 1.)

**Về "nội bộ":** với tài liệu MẬT (hợp đồng, luật) yêu cầu dữ liệu KHÔNG rời hệ thống ra cloud.
**Chỉ chạy CLOUD** (Gemini/OpenAI) — chế độ on-prem/offline đã được gỡ bỏ (xem mục 7). Vì vậy KHÔNG dùng cho tài liệu mật cần dữ liệu ở-lại-nội-bộ.

---

## 2. Kiến trúc 4 tầng

```
TẦNG 1 — INGESTION   (rag/ingest/)   ← khó nhất, làm kỹ nhất
  file -> dedup(sha256) -> phân loại trang (text/scan/hybrid theo mật độ ký tự)
       -> text: PyMuPDF | scan: Vision OCR->markdown | hybrid: merge
       -> chunk parent-child -> embed child -> store (transaction)
       -> mỗi trang bắt lỗi RIÊNG (1 trang hỏng KHÔNG giết cả tài liệu) + ghi báo cáo
          per-trang (pgvector: bảng ingest_reports | numpy: file): ok/ocr_failed/ocr_skipped/blank/error
       -> GATE: TỔNG 0 ký tự => FAIL (KHÔNG bao giờ báo 'ingested' giả). Còn ký tự nhưng
          vài trang hỏng => ingest MỘT PHẦN, message nêu rõ trang nào (GET /api/documents/<id>/report)

TẦNG 2 — RETRIEVAL   (rag/retrieve/)  ← là một pipeline, không phải 1 câu search
  query -> smalltalk cut -> NFC + segment tiếng Việt + alias số hiệu văn bản
        -> BM25 + vector -> RRF fusion -> RBAC filter (trong query)
        -> rerank -> NGƯỠNG tự tin (score < SCORE_MIN -> "không tìm thấy", KHÔNG gọi LLM)
        -> child->parent -> context đánh số [n]
        -> LLM trả lời -> VERIFY CITATION bằng code (chống bịa ~0ms)

TẦNG 3 — NÂNG CAO    (rag/advanced/)  ← code sẵn, TẮT mặc định, chỉ bật khi eval cần
  classify_query (3.1) | multihop (3.2) | reflect (3.3) | nli (3.4)   (bật qua --advanced / --nli)

TẦNG 4 — AGENTIC + CONNECTORS + WORKFLOW  ← CHƯA cần làm
  Phần lớn KHÔNG còn là RAG (nó "hành động": gửi mail, cam kết). Tách sản phẩm riêng.
```

**Vì sao tầng 4 chưa cần:** agent gọi retrieval như một tool — retrieval chưa tốt thì agent chỉ
nhân cái sai lên. Phải làm tốt tầng 1-2 (đo bằng eval set) trước.

---

## 3. Cấu trúc thư mục

```
chatbot/
  config.py                 # MỌI lựa chọn model/tham số tập trung ở đây
  cli.py                    # ingest / ask / chat / eval / log / stats
  server.py                 # Web UI + REST API (FastAPI, http://localhost:8000)
  web/index.html            # giao diện chat (1 file, không cần build)
  list_models.py            # liệt kê model Gemini mà API key được dùng
  requirements.txt
  run_gemini.bat            # nạp cấu hình CLOUD Gemini + pgvector
  run_web.bat               # = run_gemini.bat + python server.py
  run_web.bat               # = run_gemini.bat + python server.py
  rag/db.py                 # connection pool PostgreSQL dùng chung (đa người dùng)
  examples/                 # tài liệu mẫu (.pdf) + evalset_draft.json
  rag/
    schema.py               # Chunk (chunk_id DETERMINISTIC), DocStatus, sha256
    embed.py                # Embedder: CHỈ Gemini (gemini-embedding-001)
    net.py                  # HTTP + retry tuỳ chọn (RAG_MAX_RETRIES) + timeout
    cache.py                # answer cache exact-match (tự vô hiệu khi kho đổi)
    querylog.py             # ghi + đọc query_log (score/nguồn/mode/latency)
    timing.py               # đo latency từng khâu retrieve/rerank/llm
    text/
      vi.py                 # NFC + word segmentation (underthesea/pyvi/regex)
      alias.py              # "NĐ 13" <-> "Nghị định 13/2023/NĐ-CP"
    ingest/
      extract.py            # phân loại + trích text theo TRANG (PyMuPDF, python-docx)
      chunk.py              # chunking parent-child
      pipeline.py           # orchestrate + GATE
    index/
      __init__.py           # get_store() factory (numpy | pgvector)
      store.py              # numpy backend (file cục bộ storage\)
      pg_store.py           # pgvector backend (PostgreSQL, transaction)
      bm25.py               # BM25 Okapi tự viết (cho numpy backend)
    retrieve/
      hybrid.py             # BM25 + vector + RRF
      rerank.py             # llm(Gemini) | lexical (thuần Python, cũng là dự phòng)
      pipeline.py           # retrieve() + build_context()
    generate/
      llm.py                # chat + vision + stream: gemini | openai
      answer.py             # prompt có nguồn + verify citation + ngưỡng tự tin
    chat/
      store.py              # session + messages: PostgreSQL | file JSON (theo RAG_STORE)
      pipeline.py           # condense question + prompt có lịch sử + tóm tắt dần + stream
    advanced/               # tầng 3
      classify_query.py | multihop.py | reflect.py | nli.py
      smalltalk.py          # chặn chào/cảm ơn/ok trước cửa RAG (không tốn LLM)
    eval/
      harness.py            # đo hit@k, keyword_recall (hỗ trợ câu no-answer)
```

---

## 4. Cài từ đầu (venv + deps)

Mở **Command Prompt (cmd)** tại thư mục `chatbot`:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -U pip
```

**Mạng công ty chặn TLS** (lỗi `CERTIFICATE_VERIFY_FAILED`) — cài ngay, dùng cho cả pip/requests:
```bat
pip install pip-system-certs
```

Cài dependency:
```bat
pip install -r requirements.txt
```

`requirements.txt` (một số dòng để comment, bỏ comment khi cần):

| Package | Bắt buộc? | Dùng cho |
|---|---|---|
| `numpy` | ✅ | mọi môi trường |
| `requests`, `pip-system-certs` | ✅ (cloud) | gọi API + fix proxy TLS |
| `fastapi`, `uvicorn`, `python-multipart` | Web UI | `server.py` |
| `PyMuPDF` | ingest PDF | đọc PDF + render trang cho Vision |
| `python-docx` | ingest .docx | đoạn văn + bảng (→ Markdown); `.doc` cũ phải Save As `.docx` |
| `psycopg2-binary` | pgvector | PostgreSQL |
| `underthesea` / `pyvi` | nên có | tách từ tiếng Việt (không có → fallback regex) |
| `pip-system-certs` | nếu mạng chặn TLS | fix `CERTIFICATE_VERIFY_FAILED` do proxy giải mã TLS |

---

## 5. Một chế độ: Cloud (Gemini, hoặc OpenAI cho LLM/Vision)

Đã **bỏ hoàn toàn** offline (hashing giả) và local/on-prem (Ollama/bge-m3/cross-encoder HF).

| Thành phần | Provider | Model mặc định | Đổi được sang |
|---|---|---|---|
| Embedding | Gemini | `gemini-embedding-001` (1536d) | *(chỉ Gemini)* |
| LLM sinh câu | Gemini | `gemini-flash-latest` | OpenAI (`RAG_LLM_PROVIDER=openai`) |
| Vision OCR | Gemini | `gemini-flash-latest` | OpenAI |
| Rerank | `llm` (Gemini) | — | `lexical` (thuần Python) |

> ⚠️ Thiếu `GEMINI_API_KEY` hoặc lỗi API ⇒ **BÁO LỖI RÕ ngay** (không còn "âm thầm fallback
> offline GIẢ"). Dự phòng khi Gemini lỗi TẠM: rerank → `lexical`, answer → `extractive` (đều
> thuần-Python, có log). ⇒ Chỉ cần đặt key + `run_gemini.bat`, không còn phải đoán "đang ở chế độ nào".

**Quy tắc bất biến:** một kho chỉ dùng MỘT embedding (provider + model + dim). Đổi embedding =
xoá kho rồi ingest lại (numpy: `rmdir /s /q storage`; pgvector: `DROP TABLE chunks, docs;`).
Code tự chặn nếu trộn (lỗi ngay, không hỏng ngầm).

Cấu hình đang chạy → xem mục 6.

---

## 6. ⭐ Cấu hình Gemini ĐANG CHẠY ĐƯỢC (mạng công ty)

Đây là cấu hình đã chạy thành công đầu-cuối trên mạng công ty (thông Google + PyPI).
**Chỉ dùng cho tài liệu KHÔNG mật** — text tài liệu + câu hỏi được gửi lên Google.

### Bước 1 — Lấy API key
https://aistudio.google.com/apikey → **Create API key** → key dạng `AIza...` hoặc `AQ...`
- Nếu bị `429 hết credit`: tạo key trong **project MỚI chưa bật billing** → free-tier (giới hạn tốc độ).
- API restriction (Cloud Console): chọn **Gemini API** (= Generative Language API).
- Mỗi model một bucket quota riêng — 429 model này thì `python list_models.py` đổi model khác.

### Bước 2 — Cài
```bat
pip install -r requirements.txt
```

### Bước 3 — Đặt key (1 lần, KHÔNG ghi vào file trong repo)
```bat
setx GEMINI_API_KEY "AIza...key_that"
```
Rồi **mở cmd MỚI** (setx chỉ có tác dụng ở cmd mở sau đó).

### Bước 4 — Nạp cấu hình + chạy
```bat
run_gemini.bat
python cli.py ingest examples\nhnn.pdf examples\luatthue.pdf
python cli.py stats
python cli.py ask "trình tự thủ tục giám sát ngân hàng gồm mấy bước"
```

Nội dung `run_gemini.bat` (đang dùng):
```bat
set RAG_STORE=pgvector
set RAG_PG_DSN=postgresql://postgres:123456@localhost:5432/rag
set RAG_EMBED_PROVIDER=gemini
set RAG_EMBED_MODEL=gemini-embedding-001
set RAG_EMBED_DIM=1536
set RAG_LLM_PROVIDER=gemini
set RAG_LLM_MODEL=gemini-3.1-flash-lite
set RAG_VISION_PROVIDER=gemini
set RAG_VISION_MODEL=gemini-3.1-flash-lite
set RAG_RERANKER=llm
set RAG_MAX_RETRIES=0        REM prod đổi thành 2
```

**Dấu hiệu chạy thật:** `stats` báo `dim: 1536`, `ask` trả câu tiếng Việt tự nhiên kèm `[1]`,
`grounded=True`, `mode=llm`. Thiếu key → báo lỗi `Thiếu GEMINI_API_KEY` ngay (không fallback giả).

> ⚠️ `run_gemini.bat` hiện đặt `gemini-3.1-flash-lite` (tên **bản cứng**). Theo sự cố #6 dưới đây,
> tên bản cứng có thể bị Google khoá với user mới (404) — nếu gặp, đổi sang **alias** `gemini-flash-lite-latest`.

---

## 7. ❌ On-prem / local — ĐÃ GỠ

Chế độ on-prem (bge-m3 + cross-encoder qua Hugging Face + Ollama) và chế độ offline (hashing giả)
**đã được gỡ bỏ hoàn toàn** khỏi code theo quyết định "chỉ chạy cloud". Không còn
`RAG_OFFLINE`, `run_offline.bat`, `run_local.bat`, `sentence-transformers`, provider `ollama`/`local`.

⚠️ Hệ quả: hệ thống **CHỈ chạy cloud** — text tài liệu + câu hỏi được gửi lên Google (hoặc OpenAI).
Cờ `--confidential` giờ chỉ còn: (1) RBAC (chỉ người có clearance thấy), (2) **bỏ qua OCR** trang
scan của tài liệu mật (không gửi ảnh lên cloud → trang scan mật ghi `ocr_skipped`, không trích được text).
**KHÔNG dùng hệ thống này cho tài liệu mật cần dữ liệu ở-lại-nội-bộ.** Nếu sau này cần on-prem trở lại,
phải khôi phục các provider local — xem lịch sử git.

---

## 8. Lưu vào database (pgvector)

Mặc định (`RAG_STORE=numpy`) vector nằm ở file `storage\vectors.npy`. Muốn DB thật:

```bat
docker run -d --name rag-pg -p 5432:5432 -e POSTGRES_PASSWORD=123456 -e POSTGRES_DB=rag pgvector/pgvector:pg16

pip install psycopg2-binary
set RAG_STORE=pgvector
set RAG_PG_DSN=postgresql://postgres:123456@localhost:5432/rag

python cli.py ingest examples\nhnn.pdf
python cli.py stats          REM báo backend: pgvector
```
Kiểm tra: `docker exec -it rag-pg psql -U postgres -d rag -c "SELECT count(*) FROM chunks;"`

- Vector: pgvector cosine `<=>` + index HNSW. Keyword: Postgres full-text search trên text đã segment VN.
- Ghi theo **transaction**: một tài liệu ghi đủ (doc + cha + con) hoặc không ghi gì.
- Chiều vector cố định khi tạo bảng (`RAG_EMBED_DIM`). Đổi embedding = `DROP TABLE chunks, docs;` rồi ingest lại.
- Chat đa phiên, query log, answer cache, **báo cáo trích xuất per-trang** cũng lưu ở PostgreSQL khi
  `RAG_STORE=pgvector` (bảng `chat_*`, `query_log`, `answer_cache`, `ingest_reports`). Ví dụ theo dõi:
  `SELECT source, failed_pages FROM ingest_reports WHERE failed_pages > 0 ORDER BY ts DESC;`
  hoặc liệt kê từng trang hỏng: `SELECT doc_id, p->>'page', p->>'status' FROM ingest_reports,
  jsonb_array_elements(pages) p WHERE p->>'status' <> 'ok';`
- Cần BM25 Okapi thật trong DB → extension `pg_search` (ParadeDB), đổi `search_bm25` trong `pg_store.py`.

---

## 9. Reranker — 2 chế độ

Chọn qua `RAG_RERANKER`:

| Giá trị | Cần gì | Chất lượng | Ghi chú |
|---|---|---|---|
| `llm` | Gemini | Cao | **mặc định**; +1 lần gọi Gemini/câu; LLM trả `[]` = van out-of-domain (loại câu lạc đề) |
| `lexical` | Không cần gì | Thô (trùng từ) | Tức thì, miễn phí, thuần Python |

Khi `llm` lỗi/hết quota → **tự fallback `lexical`** (in `[rerank] fallback lexical ...`, không hỏng
ngầm) — hệ thống vẫn trả lời, chỉ giảm chất lượng xếp hạng.

Cơ chế `llm`: gửi Gemini danh sách đoạn đánh số → nhận mảng JSON thứ tự liên quan → lấy top 5.

---

## 10. Vision OCR cho PDF scan

PDF scan (ảnh, 0 ký tự text) → GATE báo FAIL. Bật Vision để đọc:
```bat
pip install PyMuPDF          REM render trang PDF thành ảnh
set RAG_VISION_PROVIDER=gemini
set RAG_VISION_MODEL=gemini-flash-lite-latest
python cli.py ingest examples\file_scan.pdf
```
Vision đọc cả trang → Markdown (giữ bảng, dấu tiếng Việt, bỏ mộc/chữ ký).

**Tài liệu `--confidential` KHÔNG bao giờ gửi ảnh lên cloud.** Vì Vision chỉ còn cloud (gemini/openai),
**các trang scan của tài liệu MẬT bị BỎ QUA OCR** → ghi `ocr_skipped` trong báo cáo per-trang, không
trích được text. (Đã bỏ Vision local; muốn OCR tài liệu mật cần dựng kênh OCR nội bộ riêng.)

---

## 11. Bảng biến môi trường

Mặc định dưới đây là **giá trị code trong `config.py`** (bare, chưa nạp bat nào).

| Biến | Giá trị | Mặc định (config.py) |
|---|---|---|
| `RAG_STORE` | `numpy` \| `pgvector` | `numpy` |
| `RAG_PG_DSN` | chuỗi kết nối PostgreSQL | `postgresql://postgres:123456@localhost:5432/rag` |
| `RAG_PG_POOL_MAX` | số connection tối đa trong pool pgvector (đa người dùng) | `16` |
| `RAG_DATA_DIR` | thư mục kho numpy | `storage` |
| `RAG_EMBED_PROVIDER` | `gemini` (chỉ Gemini) | `gemini` |
| `RAG_EMBED_MODEL` | tên model embedding | `gemini-embedding-001` |
| `RAG_EMBED_DIM` | số chiều | `1536` |
| `GEMINI_API_KEY` | key `AIza...`/`AQ...` | rỗng (đừng commit) |
| `OPENAI_API_KEY` / `OPENAI_BASE` | key + endpoint khi dùng LLM/Vision OpenAI | rỗng / `api.openai.com/v1` |
| `RAG_LLM_PROVIDER` | `gemini` \| `openai` | `gemini` |
| `RAG_LLM_MODEL` | tên model | `gemini-flash-latest` |
| `RAG_RERANKER` | `llm` \| `lexical` | `llm` |
| `RAG_VISION_PROVIDER` | `gemini` \| `openai` | `gemini` |
| `RAG_VISION_MODEL` | tên model | `gemini-flash-latest` |
| `RAG_MAX_RETRIES` | `0` (dev) \| `1-2` (prod, chỉ retry 429/500/503) | `0` |
| `RAG_CACHE` | `on` \| `off` — answer cache exact-match | `on` |
| `RAG_TIMEOUT` | giây chờ tối đa mỗi cú gọi LLM/embedding | `30` |
| `RAG_VISION_TIMEOUT` | giây chờ Vision OCR 1 trang | `120` |
| `RAG_SCORE_MIN` | ngưỡng tự tin (RRF) — dưới ngưỡng trả "không tìm thấy", không gọi LLM. `0`=tắt | `0.017` (ĐOÁN, chưa chốt bằng eval) |
| `PYTHONUTF8` | `1` — tránh UnicodeEncodeError console Windows | *(nên set)* |

> `set` chỉ sống trong cửa sổ cmd hiện tại. `setx` lưu lâu dài (áp dụng cmd mở MỚI).
> `run_gemini.bat` **ghi đè** mặc định config để chốt cấu hình (key, pgvector, model).

---

## 12. Lệnh CLI + Web

```bat
python cli.py ingest <file...> [--confidential] [--dept legal]   REM .pdf/.docx/.txt/.md, nhận glob
python cli.py ask "<câu hỏi>" [--dept X] [--no-clearance] [--advanced] [--nli]
python cli.py chat [--session <id>] [--list] [--once "<câu hỏi>"] [--dept X] [--no-clearance]
python cli.py eval examples\evalset_draft.json [-k 5]
python cli.py log [--tail 20]      REM query log + thống kê score (cân ngưỡng tự tin)
python cli.py stats
```

**Chat đa phiên:** kho tài liệu CHUNG, cái riêng từng phiên là lịch sử hội thoại. Câu nối tiếp
("thế còn mức phạt?") được LLM **condense** thành câu độc lập trước khi retrieve; giữ nguyên văn
8 lượt gần nhất, cũ hơn thì **tóm tắt dần**. RBAC đặt lúc TẠO phiên, cố định cả phiên.

**Query log:** mọi lượt hỏi (`ask`/`chat`/web) ghi `top_score/n_sources/grounded/mode` + latency
từng khâu — kể cả khi "không tìm thấy". `python cli.py log` in phân bố score hai nhóm (tự gợi ý
vùng ngưỡng tự tin) + latency trung bình (loại lượt cache).

**Answer cache** (exact-match, mặc định bật): câu lặp lại trả ngay từ cache. Key gồm dept/clearance
(RBAC không rò qua cache); tự vô hiệu khi kho đổi. Tắt: `set RAG_CACHE=off`.

**Web UI:**
```bat
pip install fastapi uvicorn python-multipart
run_web.bat              REM = run_gemini.bat + python server.py
```
Mở **http://localhost:8000** — giao diện **SPA một file** (`web\index.html`, không cần Node/build),
điều hướng bằng hash, 3 trang:
- **💬 Trò chuyện** (`#/chat`): sidebar phiên, trích dẫn `[n]` tô màu, streaming đáp án qua SSE.
- **📄 Tài liệu** (`#/documents`): upload/xoá tài liệu, ingest chạy nền (trạng thái tự cập nhật 2s).
  Nút 🔍 mở **báo cáo trích xuất per-trang** (trang nào OK / OCR lỗi / bỏ qua / trắng / lỗi đọc).
- **📊 Dashboard** (`#/dashboard`): tình trạng kho + thống kê query log (lượt hỏi, tỉ lệ grounded/cache,
  phân bố mode, latency từng khâu, gợi ý ngưỡng tự tin, lượt hỏi gần đây) — lấy từ `GET /api/stats`.

**Đa người dùng:** server KHÔNG còn lock toàn cục — hai người hỏi song song thật (call
LLM/embedding chậm không giữ khoá nào). Kho pgvector đi qua connection pool chung
(`rag/db.py`, tối đa `RAG_PG_POOL_MAX` connection, mặc định 16; borrower CHỜ khi hết slot
thay vì lỗi); latency đo bằng bộ nhớ thread-local; các singleton (store/embedder/chat
store/tokenizer) hâm nóng một lần lúc khởi động rồi chỉ đọc. Backend numpy (dev) tự khoá
bộ nhớ trong ở tầng store. Muốn nhiều connection hơn: `set RAG_PG_POOL_MAX=32`.

---

## 13. 📓 Nhật ký xử lý sự cố

| # | Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|---|
| 1 | `No module named 'requests'` | Thiếu lib (Gemini/OpenAI cần) | `pip install requests` (đúng venv) |
| 2 | `CERTIFICATE_VERIFY_FAILED self-signed` | Proxy công ty giải mã TLS | `pip install pip-system-certs` |
| 3 | `403 Forbidden` gọi Gemini | Thường là billing | xem #4 |
| 4 | `429 RESOURCE_EXHAUSTED` | Hết quota **của model đó** (mỗi model 1 bucket) | `list_models.py` đổi model; embedding đổi = xoá kho + ingest lại; hoặc key free-tier project chưa bật billing |
| 5 | `401`, URL vẫn key cũ | Key set ở cửa sổ khác / `setx` cũ đè | `echo %GEMINI_API_KEY%`; set key + chạy python CÙNG cửa sổ |
| 6 | `404 no longer available` (`gemini-2.5-flash`, `gemini-3.1-flash-lite`?) | Tên model **bản cứng** bị khoá với user mới (dù vẫn hiện trong ListModels) | Dùng alias `*-latest` (vd `gemini-flash-lite-latest`) |
| 7 | `[rerank] fallback lexical` liên tục | Rerank `llm` lỗi/hết quota (Gemini) | Kiểm key/quota (`list_models.py`); hoặc chủ động `set RAG_RERANKER=lexical` |
| 8 | `Thiếu GEMINI_API_KEY` khi ingest/ask | chưa đặt key (hoặc set ở cửa sổ khác) | `setx GEMINI_API_KEY "..."` rồi mở cmd MỚI + `run_gemini.bat` |
| 9 | `curl: (35) schannel REVOCATION` | curl siết kiểm tra thu hồi cert | `curl --ssl-no-revoke ...` (Python không dính) |
| 10 | Biến `set` "mất" liên tục | `set` chỉ sống trong 1 cửa sổ cmd | Chạy `run_*.bat` mỗi phiên; key thì `setx` |
| 11 | `[FAIL] ... 0 ký tự` khi ingest PDF | Thiếu PyMuPDF, HOẶC PDF là scan | `pip install PyMuPDF`; scan → bật Vision (mục 10) |
| 12 | `could not broadcast (1536) vs (256)` / khác model | Trộn kho khác chiều/model vector | Xoá kho (`rmdir /s /q storage` hoặc `DROP TABLE chunks, docs;`) rồi ingest lại |
| 13 | `UnicodeEncodeError` (cp1252) | Console Windows | `cli.py` đã ép UTF-8; chỗ khác: `set PYTHONUTF8=1` |
| 14 | `.doc chưa hỗ trợ` khi ingest | Word 97-2003 nhị phân, python-docx không đọc | Mở Word → Save As → `.docx` rồi nạp lại |

**Yêu cầu mạng:** chỉ cần thông `generativelanguage.googleapis.com` (Google) — và `api.openai.com`
nếu dùng OpenAI — cùng PyPI để cài lib. Không còn phụ thuộc `huggingface.co` (đã bỏ model local).

---

## 14. Quyết định khó đảo ngược (đã xử lý sẵn trong code)

1. **Embedding (provider+model+dim)** lưu vào metadata TỪNG vector → migrate từng phần. Đổi chiều = báo lỗi ngay.
2. **chunk_id deterministic** (`doc::pN::cNNN`) → citation cũ không chết khi re-index.
3. **RBAC ở tầng retrieval** (lọc trong query vector store), KHÔNG lọc sau khi model đã đọc.

---

## 15. Checklist dựng lại nhanh

Chế độ **Cloud Gemini** (tài liệu KHÔNG mật):
```bat
REM 1) venv + deps
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -U pip
pip install pip-system-certs
pip install -r requirements.txt

REM 2) đặt key 1 lần rồi MỞ CMD MỚI
setx GEMINI_API_KEY "AIza...key_that"

REM --- cmd mới: vào thư mục + activate venv ---
.venv\Scripts\activate.bat

REM 3) (tuỳ chọn) DB thật
docker run -d --name rag-pg -p 5432:5432 -e POSTGRES_PASSWORD=123456 -e POSTGRES_DB=rag pgvector/pgvector:pg16

REM 4) nạp cấu hình + chạy
run_gemini.bat
python cli.py ingest examples\nhnn.pdf examples\luatthue.pdf
python cli.py stats
python cli.py ask "trình tự thủ tục giám sát ngân hàng gồm mấy bước"
```

**Nghiệm thu đạt:** `stats` = `dim: 1536` (hoặc `backend: pgvector`); `ask` trả tiếng Việt kèm `[1]`,
`grounded=True`, `mode=llm`. Thiếu key → báo lỗi `Thiếu GEMINI_API_KEY` ngay (không fallback giả).

> Hệ thống **chỉ chạy cloud** (Gemini/OpenAI) — không dùng cho tài liệu mật cần ở-lại-nội-bộ.

---

## Bước tiếp theo nên làm (xem `Todo.md` để chi tiết)

1. **Duyệt + chạy eval set** (Todo mục 1) — biết retrieval đủ tốt chưa (chưa chạy lần nào).
2. Chốt **ngưỡng tự tin** từ số liệu eval/log (Todo mục 2) — hiện 0.017 là đoán.
3. Chỉ bật **1-2 mảnh tầng 3** nếu eval lộ điểm yếu cụ thể (đừng bật cả bộ).
