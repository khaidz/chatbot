# RAG Chatbot tiếng Việt — hỏi đáp tài liệu có trích nguồn

Chatbot đọc tài liệu (nghị định, thông tư, hợp đồng... dạng `.pdf/.txt/.md`), trả lời **có trích nguồn `[n]`**, **không bịa** (verify citation bằng code). Hỗ trợ tiếng Việt đầy đủ: chuẩn hoá NFC, tách từ (underthesea), alias số hiệu văn bản ("NĐ 13" ↔ "Nghị định 13/2023/NĐ-CP").

Môi trường tham chiếu: **Windows 11, Python 3.11, cmd**.

---

## Mục lục

1. [Kiến trúc](#1-kiến-trúc)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Cài môi trường từ đầu](#3-cài-môi-trường-từ-đầu)
4. [Ba môi trường chạy](#4-ba-môi-trường-chạy)
5. [⭐ Cấu hình Gemini đang chạy được](#5--cấu-hình-gemini-đang-chạy-được)
6. [Lưu vào PostgreSQL (pgvector)](#6-lưu-vào-postgresql-pgvector)
7. [Reranker — 3 chế độ](#7-reranker--3-chế-độ)
8. [Vision OCR cho PDF scan](#8-vision-ocr-cho-pdf-scan)
9. [Bảng biến môi trường](#9-bảng-biến-môi-trường)
10. [Lệnh CLI](#10-lệnh-cli)
11. [Đọc kết quả `ask`](#11-đọc-kết-quả-ask)
12. [Eval — đo chất lượng](#12-eval--đo-chất-lượng)
13. [Xử lý sự cố](#13-xử-lý-sự-cố)
14. [Checklist dựng lại nhanh](#14-checklist-dựng-lại-nhanh)

---

## 1. Kiến trúc

```
TẦNG 1 — INGESTION   (rag/ingest/)   ← khó nhất, làm kỹ nhất
  file -> dedup(sha256) -> phân loại trang (text/scan/hybrid theo mật độ ký tự)
       -> text: PyMuPDF | scan: Vision OCR->markdown | hybrid: merge
       -> chunk parent-child (cha ~1200 ký tự cho LLM đọc, con ~350 để embed)
       -> embed con -> store (transaction: ghi đủ hoặc không ghi gì)
       -> GATE: 0 ký tự => FAIL (KHÔNG bao giờ báo 'ingested' giả)

TẦNG 2 — RETRIEVAL   (rag/retrieve/)  ← là một pipeline, không phải 1 câu search
  query -> NFC + tách từ tiếng Việt + alias số hiệu văn bản
        -> BM25 + vector -> RRF fusion -> RBAC lọc TRONG query
        -> rerank -> child->parent -> context đánh số [n]
        -> LLM trả lời -> VERIFY CITATION bằng code (chống bịa, ~0ms)

TẦNG 3 — NÂNG CAO    (rag/advanced/)  ← code sẵn, TẮT mặc định, chỉ bật khi eval cần
  classify_query | multihop | reflect | nli   (bật qua cờ --advanced / --nli)
```

Nguyên tắc: **80% chất lượng nằm ở ingestion + hybrid search + rerank + citation**, không phải ở model to. Trước khi tối ưu bất cứ gì: có eval set 50-100 câu.

Ba quyết định khó đảo ngược đã xử lý sẵn trong code:

1. Embedding (provider, model, dim) lưu vào metadata kho — trộn kho khác embedding là **lỗi ngay** kèm hướng dẫn, không hỏng ngầm.
2. `chunk_id` deterministic (`doc::pN::cNNN`) — re-index cùng tài liệu ra cùng id, citation cũ không chết.
3. RBAC ở tầng retrieval (lọc trong query) — model không bao giờ đọc được tài liệu không có quyền.

## 2. Cấu trúc thư mục

```
chatbot/
  config.py                 # MỌI lựa chọn model/tham số tập trung ở đây
  cli.py                    # ingest / ask / chat / eval / stats
  server.py                 # Web UI + REST API (FastAPI, http://localhost:8000)
  web/index.html            # giao diện chat (1 file, không cần build)
  list_models.py            # liệt kê model Gemini mà API key được dùng
  requirements.txt
  run_gemini.bat            # nạp nhanh cấu hình Gemini (chạy mỗi phiên cmd)
  run_offline.bat           # nạp chế độ offline (smoke-test, 0đ)
  examples/                 # tài liệu mẫu + evalset.json
  rag/
    schema.py               # Chunk (chunk_id DETERMINISTIC), DocStatus, sha256
    embed.py                # Embedder: local(bge-m3) | gemini | offline(hashing)
    net.py                  # HTTP + retry tuỳ chọn (RAG_MAX_RETRIES)
    text/
      vi.py                 # NFC + tách từ (underthesea > pyvi > regex)
      alias.py              # "NĐ 13" <-> "Nghị định 13/2023/NĐ-CP"
    ingest/
      extract.py            # phân loại + trích text theo TRANG
      chunk.py              # chunking parent-child
      pipeline.py           # orchestrate + GATE
    index/
      __init__.py           # get_store() factory (numpy | pgvector)
      store.py              # numpy backend (file cục bộ storage\)
      pg_store.py           # pgvector backend (PostgreSQL, transaction)
      bm25.py               # BM25 Okapi tự viết (cho numpy backend)
    retrieve/
      hybrid.py             # BM25 + vector + RRF (gắn điểm rrf vào chunk)
      rerank.py             # cross-encoder | llm(Gemini) | lexical
      pipeline.py           # retrieve() + build_context() (điểm liên quan)
    generate/
      llm.py                # chat + vision: ollama|openai|gemini|offline
      answer.py             # prompt có nguồn + verify citation bằng code
    chat/
      store.py              # session + messages: PostgreSQL | file JSON (theo RAG_STORE)
      pipeline.py           # condense question + prompt có lịch sử + tóm tắt dần
    advanced/               # tầng 3: classify_query | multihop | reflect | nli
    eval/harness.py         # đo hit@k, keyword_recall
```

## 3. Cài môi trường từ đầu

Mở **Command Prompt (cmd)** tại thư mục dự án:

```bat
REM 1) venv Python 3.11
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -U pip

REM 2) Mạng công ty chặn TLS (CERTIFICATE_VERIFY_FAILED)? Cài NGAY dòng này:
pip install pip-system-certs

REM 3) Cài dependency
pip install -r requirements.txt
```

`requirements.txt` gồm:

| Package | Bắt buộc? | Dùng cho |
|---|---|---|
| `numpy` | ✅ | mọi môi trường |
| `requests` | ✅ (Gemini/Ollama) | gọi API |
| `pip-system-certs` | mạng công ty | fix proxy giải mã TLS |
| `PyMuPDF` | khi ingest PDF | đọc PDF + render trang cho Vision |
| `python-docx` | khi ingest .docx | đoạn văn + bảng (→ Markdown); `.doc` cũ phải Save As `.docx` |
| `psycopg2-binary` | khi dùng pgvector | PostgreSQL |
| `underthesea` / `pyvi` | nên có | tách từ tiếng Việt (không có thì fallback regex) |
| `sentence-transformers` | chỉ khi dùng local bge-m3 | ⚠️ CẦN Hugging Face — mạng công ty chặn |

## 4. Ba môi trường chạy

| Môi trường | Embedding | LLM | Chi phí | Dùng khi |
|---|---|---|---|---|
| **Offline** | hashing (giả, 256d) | extractive | 0đ | Smoke-test luồng, không cần mạng/key |
| **Local** | bge-m3 (HF ~2GB) | Ollama | 0đ | Tài liệu MẬT (⚠️ cần Hugging Face) |
| **Cloud (Gemini)** | gemini-embedding-2 | gemini-flash-lite-latest | tốn quota | Demo, tài liệu KHÔNG mật |

**Quy tắc:** một kho chỉ dùng MỘT embedding (provider + model + dim). Đổi embedding = xoá kho rồi ingest lại (numpy: `rmdir /s /q storage`; pgvector: `DROP TABLE chunks, docs;`). Code tự chặn nếu trộn.

### Offline (thử nhanh nhất)

```bat
run_offline.bat
python cli.py ingest examples\nd13.md examples\hopdong.md
python cli.py ask "mức phạt vi phạm hợp đồng là bao nhiêu"
python cli.py eval examples\evalset.json
```

Dấu hiệu: `stats` báo `dim: 256`, ask trả `mode=extractive`.

## 5. ⭐ Cấu hình Gemini đang chạy được

### Bước 1 — Lấy API key

https://aistudio.google.com/apikey → **Create API key** (key dạng `AIza...` hoặc `AQ...`).

- Bị 429 hết credit? Tạo key trong **project MỚI chưa bật billing** → free-tier (giới hạn tốc độ).
- API restriction (Cloud Console): chọn **Gemini API** (= Generative Language API).

### Bước 2 — Đặt key (1 lần)

```bat
setx GEMINI_API_KEY "AIza...key_that"
```

Rồi **mở cmd MỚI** (`setx` chỉ có tác dụng ở cmd mở sau đó). KHÔNG commit key vào repo.

### Bước 3 — Nạp cấu hình + chạy

```bat
.venv\Scripts\activate.bat
run_gemini.bat

python cli.py ingest examples\nd13.md examples\hopdong.md
python cli.py stats
python cli.py ask "NĐ 13 định nghĩa dữ liệu cá nhân thế nào"
```

`run_gemini.bat` nạp (chạy lại mỗi phiên cmd — `set` chỉ sống trong 1 cửa sổ):

```bat
set PYTHONUTF8=1
set RAG_EMBED_PROVIDER=gemini
set RAG_EMBED_MODEL=gemini-embedding-2      REM embedding-001 đã hết quota (429)
set RAG_EMBED_DIM=1536
set RAG_LLM_PROVIDER=gemini
set RAG_LLM_MODEL=gemini-flash-lite-latest  REM LUÔN dùng alias *-latest, xem sự cố #6
set RAG_RERANKER=llm
set RAG_VISION_PROVIDER=gemini
set RAG_VISION_MODEL=gemini-flash-lite-latest
set RAG_STORE=pgvector
set RAG_MAX_RETRIES=0                       REM prod đổi thành 2
```

**Nghiệm thu đạt:** `stats` báo `dim: 1536`, ingest KHÔNG có dòng `[embed] fallback OFFLINE`, ask trả tiếng Việt tự nhiên kèm `[1]`, `grounded=True`, `mode=llm`.

### Chọn model — dùng `list_models.py`

```bat
run_gemini.bat
python list_models.py
```

In danh sách model key được dùng, chia 2 nhóm: **EMBEDDING** (cho `RAG_EMBED_MODEL`) và **GENERATE** (cho `RAG_LLM_MODEL`/`RAG_VISION_MODEL`). Hai quy tắc quan trọng:

1. **Mỗi model một bucket quota riêng** — 429 model này thì đổi model khác trong danh sách là chạy tiếp.
2. **Luôn dùng alias** (`gemini-flash-lite-latest`, `gemini-flash-latest`...) — tên bản cứng (`gemini-2.5-flash-lite`) có thể bị Google khoá với user mới (404) **dù vẫn hiện trong danh sách**.

Đổi LLM/Vision/rerank: thoải mái, không cần re-ingest. Đổi EMBEDDING: phải xoá kho + ingest lại.

## 6. Lưu vào PostgreSQL (pgvector)

```bat
docker run -d --name rag-pg -p 5432:5432 -e POSTGRES_PASSWORD=123456 -e POSTGRES_DB=rag pgvector/pgvector:pg16

pip install psycopg2-binary
set RAG_STORE=pgvector
set RAG_PG_DSN=postgresql://postgres:123456@localhost:5432/rag

python cli.py ingest examples\nd13.md examples\hopdong.md
python cli.py stats          REM báo backend: pgvector
```

Kiểm tra: `docker exec -it rag-pg psql -U postgres -d rag -c "SELECT count(*) FROM chunks;"`

- Vector: pgvector cosine `<=>` + index HNSW. Keyword: full-text search trên text đã tách từ VN.
- Ghi theo **transaction**: một tài liệu ghi đủ (doc + cha + con) hoặc không ghi gì — lỗi giữa chừng không để rác.
- Chiều vector cố định khi tạo bảng. Đổi embedding: `DROP TABLE chunks, docs;` rồi ingest lại.
- Cần BM25 Okapi thật trong DB → extension `pg_search` (ParadeDB), thay `search_bm25` trong `pg_store.py`.

## 7. Reranker — 3 chế độ

Chọn qua `RAG_RERANKER`:

| Giá trị | Cần gì | Chất lượng | Ghi chú |
|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` (mặc định) | Hugging Face | Cao nhất | Mạng công ty chặn HF → không dùng được |
| `llm` | Gemini | Gần cross-encoder | +1 lần gọi Gemini/câu, né HF hoàn toàn |
| `lexical` | Không cần gì | Thô (trùng từ) | Tức thì, miễn phí |

Mọi chế độ lỗi đều **tự fallback lexical** (in `[rerank] fallback lexical ...`, không hỏng ngầm) — hệ thống vẫn trả lời, chỉ giảm chất lượng xếp hạng; tệ nhất = chất lượng hybrid search thuần.

## 8. Vision OCR cho PDF scan

PDF scan (ảnh, 0 ký tự) → GATE báo FAIL. Bật Vision để đọc:

```bat
pip install PyMuPDF
set RAG_VISION_PROVIDER=gemini
set RAG_VISION_MODEL=gemini-flash-lite-latest
python cli.py ingest duong\dan\file_scan.pdf
```

Gemini đọc cả trang → Markdown (giữ bảng, dấu tiếng Việt, bỏ mộc/chữ ký). Tài liệu `--confidential` **KHÔNG BAO GIỜ** gửi ảnh lên Gemini (tự ép về Ollama local).

## 9. Bảng biến môi trường

| Biến | Giá trị | Mặc định |
|---|---|---|
| `RAG_OFFLINE` | `force` (ép offline) \| *(rỗng)* (auto-fallback) \| `off` (cấm fallback) | auto |
| `RAG_STORE` | `numpy` \| `pgvector` | `numpy` |
| `RAG_PG_DSN` | chuỗi kết nối PostgreSQL | `postgresql://postgres:123456@localhost:5432/rag` |
| `RAG_DATA_DIR` | thư mục kho numpy | `storage` |
| `RAG_EMBED_PROVIDER` | `local` \| `gemini` \| `offline` | `local` |
| `RAG_EMBED_MODEL` | tên model | theo provider |
| `RAG_EMBED_DIM` | số chiều | bge-m3=1024, gemini=1536, offline=256 |
| `GEMINI_API_KEY` | key `AIza...`/`AQ...` | rỗng (đừng commit) |
| `RAG_LLM_PROVIDER` | `ollama` \| `openai` \| `gemini` \| `offline` | `offline` |
| `RAG_LLM_MODEL` | tên model | `qwen2.5:7b` |
| `RAG_RERANKER` | tên model HF \| `llm` \| `lexical` | `BAAI/bge-reranker-v2-m3` |
| `RAG_VISION_PROVIDER` | `ollama` \| `openai` \| `gemini` \| `offline` | `offline` |
| `RAG_VISION_MODEL` | tên model | `qwen2-vl` |
| `RAG_MAX_RETRIES` | `0` = không retry (dev) \| `1-2` (prod, chỉ retry 429/500/503) | `0` |
| `PYTHONUTF8` | `1` — tránh UnicodeEncodeError console Windows | *(nên set)* |

> `set` chỉ sống trong cửa sổ cmd hiện tại. `setx` lưu lâu dài (áp dụng cmd mở MỚI).
> Retry KHÔNG tốn thêm phí (request lỗi không được bill); lỗi vĩnh viễn (401/404) không bao giờ retry.

## 10. Lệnh CLI

```bat
python cli.py ingest <file...> [--confidential] [--dept legal]   REM .pdf/.docx/.txt/.md, nhận glob
python cli.py ask "<câu hỏi>" [--dept X] [--no-clearance] [--advanced] [--nli]
python cli.py chat [--session <id>] [--list] [--once "<câu hỏi>"] [--dept X] [--no-clearance]
python cli.py eval examples\evalset.json [-k 5]
python cli.py log [--tail 20]      REM query log + thống kê score (cân ngưỡng tự tin)
python cli.py stats
```

**Query log:** mọi lượt hỏi (`ask`/`chat`/web) được ghi lại — kể cả khi "Không tìm thấy"
(giữ `top_score` của nguồn tốt nhất retrieval tìm được). Backend theo `RAG_STORE`:
bảng `query_log` (pgvector) hoặc `<RAG_DATA_DIR>\query_log.jsonl` (numpy).
`python cli.py log` in các lượt gần nhất + phân bố score hai nhóm trả-lời-được /
không-tìm-thấy — khi hai nhóm tách nhau, nó tự gợi ý vùng đặt ngưỡng tự tin (Todo.md mục 2).
Ghi log lỗi không bao giờ làm hỏng câu trả lời (chỉ in cảnh báo).

- `--confidential`: tài liệu mật — không gửi ảnh lên cloud khi OCR, chỉ người có clearance thấy.
- `--dept legal`: tài liệu thuộc phòng ban — chỉ query cùng `--dept` mới thấy (tài liệu không gắn dept = công khai).
- `--advanced`: bật tầng 3 (phân loại câu hỏi + multihop cho câu so sánh/nhiều vế).
- `--nli`: kiểm từng câu trả lời có được nguồn hỗ trợ không, in cảnh báo câu nghi ngờ.

### Chat đa phiên (`cli.py chat`)

Kho tài liệu **CHUNG** cho mọi phiên; cái riêng từng phiên là **lịch sử hội thoại**:

```bat
python cli.py chat                       REM tạo phiên mới, vòng lặp gõ-đáp (exit để thoát)
python cli.py chat --list                REM liệt kê phiên đã có
python cli.py chat --session abc123      REM nối lại phiên cũ, nhớ nguyên ngữ cảnh
python cli.py chat --once "câu hỏi"      REM hỏi 1 câu rồi thoát (script/test)
```

Cách hoạt động một lượt:

```
câu hỏi nối tiếp ("thế còn mức phạt?")
  → CONDENSE: LLM viết lại thành câu ĐỘC LẬP dựa trên hội thoại (in [condense] → ...)
  → retrieve(câu độc lập)              ← pipeline retrieval cũ, nguyên vẹn
  → prompt = tóm tắt hội thoại cũ + 8 lượt gần nhất + nguồn [n] + câu gốc
  → trả lời + verify citation như thường
  → lưu 2 message vào DB
```

- Lịch sử lưu theo `RAG_STORE`: pgvector → bảng `chat_sessions`/`chat_messages` (PostgreSQL);
  numpy → file JSON trong `storage\chat\`.
- RBAC (`--dept`, `--no-clearance`) đặt **lúc TẠO phiên**, cố định cả phiên — câu sau không "quên cờ" được.
- Hội thoại dài: giữ nguyên văn 8 lượt gần nhất, phần cũ hơn được LLM **tóm tắt dần** vào
  `summary` của phiên (offline thì chỉ cắt cửa sổ, không tóm tắt).
- Offline/LLM lỗi: condense bị bỏ qua (dùng câu gốc), trả lời extractive — chat vẫn chạy.

### Web UI (`server.py`)

```bat
pip install fastapi uvicorn
run_web.bat              REM = run_gemini.bat + python server.py
```

Mở **http://localhost:8000** — giao diện chat đầy đủ: sidebar danh sách phiên (tạo/xoá/nối lại),
bong bóng hội thoại, trích dẫn `[n]` tô màu, danh sách nguồn kèm `liên quan %`, badge
`✓ có nguồn`, dòng *"đã hiểu là: ..."* khi câu hỏi nối tiếp được viết lại. Tự theo
light/dark của hệ điều hành. Không cần Node/build — một file `web\index.html`.

REST API (gọi được từ ứng dụng khác):

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/api/sessions` | danh sách phiên |
| POST | `/api/sessions` | tạo phiên `{dept?, clearance?}` |
| GET | `/api/sessions/{sid}` | thông tin phiên + toàn bộ messages |
| POST | `/api/sessions/{sid}/messages` | hỏi `{question}` → answer/sources/grounded/mode/standalone |
| DELETE | `/api/sessions/{sid}` | xoá phiên (messages xoá theo) |
| GET | `/api/documents` | danh sách tài liệu + trạng thái |
| POST | `/api/documents` | upload multipart (`file`, `dept?`, `confidential?`) — ingest chạy nền |
| DELETE | `/api/documents/{doc_id}` | xoá tài liệu khỏi kho (chunks + vectors) |

### Quản lý tài liệu trên UI (nút "📄 Quản lý tài liệu")

- **Upload** `.pdf/.docx/.txt/.md` (kèm tuỳ chọn `dept`/`mật`) — file gốc lưu ở `uploads\`,
  ingest chạy **nền**: badge `⏳ đang xử lý` tự cập nhật mỗi 2s → `✓ đã nạp` (kèm số
  cha/con) hoặc `✗ lỗi` (kèm lý do, ví dụ PDF scan chưa bật Vision).
- **Trạng thái** persist ở `<RAG_DATA_DIR>\ingest_status.json` — sống qua restart server.
- **Xoá** 🗑 gỡ tài liệu khỏi kho (chunks + vectors); phiên chat cũ vẫn giữ nguyên lịch sử,
  nhưng câu hỏi mới sẽ không lấy được từ tài liệu đã xoá.
- **Upload lại cùng tên**: nội dung y hệt (trùng sha256) → bỏ qua, báo trùng; nội dung MỚI
  → **thay thế trọn vẹn bản cũ** (xoá bản cũ chỉ sau khi bản mới embed thành công —
  bản mới lỗi thì bản cũ còn nguyên).

> Server serialize mọi request qua một lock (store/embedder singleton chưa thread-safe) —
> đủ cho nội bộ/single-user; nhiều user đồng thời mới cần connection pool + worker riêng.

## 11. Đọc kết quả `ask`

```
Trình tự, thủ tục giám sát ngân hàng gồm 3 bước sau: ... [1].

Nguồn:
  [1] nhnn.pdf — trang 6 — liên quan 94% (rrf 0.0152) (nhnn::p6::P000)

grounded=True  mode=llm
```

- **`[n]`**: trích dẫn — được verify bằng code, mọi `[n]` phải trỏ đến nguồn có thật.
- **`liên quan %`**: điểm RRF (hợp nhất BM25 + vector) so với nguồn mạnh nhất của lần truy vấn. 94% nghĩa là nguồn được trích không phải nguồn retrieval mạnh nhất — tín hiệu debug hữu ích.
- **`grounded`**: `True` = mọi trích dẫn hợp lệ, hoặc trả lời trung thực "không tìm thấy".
- **`mode`**: `llm` (Gemini trả lời) | `extractive` (LLM lỗi/offline — nhặt câu từ nguồn) | `no-context` (retrieval không ra gì).

## 12. Eval — đo chất lượng

**Điều kiện tiên quyết trước khi tối ưu bất cứ gì.** Format `examples\evalset.json`:

```json
[{ "q": "câu hỏi", "expect_doc": "nd13", "keywords": ["từ khoá phải có trong context"] }]
```

```bat
python cli.py eval examples\evalset.json
```

Đo `hit@k` (đúng tài liệu trong top-k) + `keyword_recall`. Muốn biết rerank đáng giá bao nhiêu:

```bat
run_gemini.bat
python cli.py eval examples\evalset.json     REM rerank llm

set RAG_RERANKER=lexical
python cli.py eval examples\evalset.json     REM mô phỏng rerank chết hẳn
```

> Lưu ý cmd: đừng viết `set X=value && lệnh` — cmd sẽ gán cả dấu cách trước `&&` vào giá trị biến.

Lộ trình nâng chất lượng (theo thứ tự): (1) mở rộng eval 50-100 câu — thêm câu diễn đạt khác từ ngữ, câu không có đáp án, câu tra số; (2) chunk theo ranh giới `Điều X.`; (3) ngưỡng điểm tự tin → trả "không tìm thấy" sớm; (4) chỉ bật tầng 3 khi eval lộ điểm yếu cụ thể.

## 13. Xử lý sự cố

| # | Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|---|
| 1 | `No module named 'requests'` | thiếu lib | `pip install requests` (đúng venv) |
| 2 | `CERTIFICATE_VERIFY_FAILED self-signed` | proxy công ty giải mã TLS | `pip install pip-system-certs` |
| 3 | 403 Forbidden gọi Gemini | thường là billing | xem #4 |
| 4 | 429 `RESOURCE_EXHAUSTED` | hết quota **của model đó** (mỗi model 1 bucket) | `python list_models.py` → đổi model; embedding đổi = xoá kho + ingest lại; hoặc key free-tier project chưa bật billing |
| 5 | 401, URL vẫn key cũ | key set ở cửa sổ khác / setx cũ đè | `echo %GEMINI_API_KEY%`; set key + chạy CÙNG cửa sổ |
| 6 | 404 `no longer available to new users` | tên model **bản cứng** bị khoá với user mới (dù vẫn hiện trong ListModels!) | dùng alias `*-latest` |
| 7 | `WinError 10054 ... huggingface.co` | mạng công ty chặn HF | bỏ local bge-m3, dùng Gemini; hoặc chép cache model + `HF_HUB_OFFLINE=1` |
| 8 | `[rerank] fallback lexical` liên tục | reranker mặc định cần HF | `set RAG_RERANKER=llm` |
| 9 | `curl: (35) schannel` | curl siết kiểm tra cert | `curl --ssl-no-revoke` (Python không dính) |
| 10 | biến `set` "mất" liên tục | `set` chỉ sống 1 cửa sổ | chạy `run_gemini.bat` mỗi phiên; key thì `setx` |
| 11 | `[FAIL] ... 0 ký tự` khi ingest PDF | thiếu PyMuPDF HOẶC PDF scan | `pip install PyMuPDF`; scan → bật Vision (mục 8) |
| 12 | lỗi trộn chiều vector / khác model | trộn kho khác embedding | xoá kho (`rmdir /s /q storage` hoặc `DROP TABLE chunks, docs;`) rồi ingest lại |
| 13 | `UnicodeEncodeError (cp1252)` | console Windows | `cli.py` đã ép UTF-8; script khác: `set PYTHONUTF8=1` |
| 14 | ingest fail giữa chừng vì 429 (không retry) | free-tier giới hạn RPM | chờ ~1 phút, chạy lại đúng lệnh ingest — file đã xong tự bỏ qua (dedup sha256), pgvector không để rác (transaction); hoặc `set RAG_MAX_RETRIES=2` |
| 15 | `.doc chưa hỗ trợ` khi ingest | Word 97-2003 nhị phân, python-docx không đọc được | mở Word → Save As → `.docx` rồi nạp lại |

**Bài học mạng công ty:** thông `generativelanguage.googleapis.com` (Google) và PyPI, **chặn hoàn toàn Hugging Face** → mọi thứ chạy qua Gemini; PyMuPDF/psycopg2/underthesea trên PyPI cài bình thường.

## 14. Checklist dựng lại nhanh

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
python cli.py ingest examples\nd13.md examples\hopdong.md
python cli.py stats
python cli.py ask "NĐ 13 định nghĩa dữ liệu cá nhân thế nào"
python cli.py eval examples\evalset.json
```

**Nghiệm thu đạt:** `stats` = `dim: 1536` + `backend: pgvector`; ask trả tiếng Việt kèm `[1]` + `liên quan %`, `grounded=True`, `mode=llm`, không có dòng fallback OFFLINE.
