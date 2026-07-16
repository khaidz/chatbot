# Cẩm nang triển khai RAG chatbot (từ đầu → chạy được)

Tài liệu này gộp toàn bộ quá trình: kiến trúc, quyết định, các bước cài đặt, cấu hình đang
chạy được, và **nhật ký xử lý mọi lỗi đã gặp**. Mang file này + thư mục code là dựng lại được y hệt.

Môi trường tham chiếu: **Windows 11, Python 3.11, cmd**, mạng công ty có proxy chặn TLS + chặn Hugging Face.

---

## Mục lục
1. [Mục tiêu & triết lý](#1-mục-tiêu--triết-lý)
2. [Kiến trúc 4 tầng](#2-kiến-trúc-4-tầng)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài từ đầu (venv + deps)](#4-cài-từ-đầu-venv--deps)
5. [Ba môi trường chạy](#5-ba-môi-trường-chạy)
6. [⭐ Cấu hình ĐANG CHẠY ĐƯỢC (Gemini + mạng công ty)](#6--cấu-hình-đang-chạy-được-gemini--mạng-công-ty)
7. [Lưu vào database (pgvector)](#7-lưu-vào-database-pgvector)
8. [Reranker — 3 chế độ](#8-reranker--3-chế-độ)
9. [Vision OCR cho PDF scan](#9-vision-ocr-cho-pdf-scan)
10. [Bảng biến môi trường](#10-bảng-biến-môi-trường)
11. [Lệnh CLI](#11-lệnh-cli)
12. [📓 Nhật ký xử lý sự cố](#12--nhật-ký-xử-lý-sự-cố)
13. [Quyết định khó đảo ngược](#13-quyết-định-khó-đảo-ngược)
14. [Checklist dựng lại nhanh](#14-checklist-dựng-lại-nhanh)

---

## 1. Mục tiêu & triết lý

Xây RAG chatbot đọc tài liệu (nghị định, hợp đồng...) tiếng Việt, trả lời **có trích nguồn**, **không bịa**.

Nguyên tắc cốt lõi (rút từ kinh nghiệm thực chiến):
- **80% chất lượng nằm ở ingestion + hybrid search + rerank + citation**, không phải ở model to.
- "Rác vào → rác ra, kèm thái độ tự tin." Ingestion bẩn thì retrieval xịn mấy cũng vô nghĩa.
- **Model nhỏ chuyên dụng ở mọi tầng giữa, model to chỉ ở tầng cuối.**
- Tầng nâng cao (multi-hop, self-reflection, NLI, agentic) là **thuốc theo triệu chứng** — chỉ bật khi
  eval set chứng minh cần. Build cả bộ "cho đủ" = cái bẫy 6 tháng.
- **Điều kiện tiên quyết trước khi tối ưu: eval set 50-100 câu có đáp án + nguồn.**

---

## 2. Kiến trúc 4 tầng

```
TẦNG 1 — INGESTION   (rag/ingest/)   ← khó nhất, làm kỹ nhất
  file -> dedup(sha256) -> phân loại trang (text/scan/hybrid theo mật độ ký tự)
       -> text: PyMuPDF | scan: Vision OCR->markdown | hybrid: merge
       -> chunk parent-child -> embed child -> store
       -> GATE: 0 ký tự => FAIL (KHÔNG bao giờ báo 'ingested' giả)

TẦNG 2 — RETRIEVAL   (rag/retrieve/)  ← là một pipeline, không phải 1 câu search
  query -> NFC + segment tiếng Việt + alias số hiệu văn bản
        -> BM25 + vector -> RRF fusion -> RBAC filter (trong query)
        -> rerank -> child->parent -> context đánh số [n]
        -> LLM trả lời -> VERIFY CITATION bằng code (chống bịa ~0ms)

TẦNG 3 — NÂNG CAO    (rag/advanced/)  ← chỉ bật khi eval set cần
  classify_query (3.1) | multihop (3.2) | reflect (3.3) | nli (3.4)

TẦNG 4 — AGENTIC + CONNECTORS + WORKFLOW  ← CHƯA cần làm
  Phần lớn KHÔNG còn là RAG (nó "hành động": gửi mail, cam kết).
  Tách sản phẩm riêng. Chỉ làm khi: cần kết hợp nhiều nguồn khác loại,
  tài liệu rải rác Drive/SharePoint, hoặc AI phải hành động (không chỉ trả lời).
```

**Vì sao tầng 4 chưa cần:** agent gọi retrieval như một tool — retrieval chưa tốt thì agent chỉ
nhân cái sai lên và "gửi mail" cái sai đó. Phải làm tốt tầng 1-2 (đo bằng eval set) trước.

---

## 3. Cấu trúc thư mục

```
chatbot/
  config.py                 # MỌI lựa chọn model/tham số tập trung ở đây
  cli.py                    # ingest / ask / eval / stats
  requirements.txt
  run_gemini.bat            # nạp nhanh cấu hình Gemini
  examples/                 # tài liệu mẫu + evalset.json
  rag/
    schema.py               # Chunk (chunk_id DETERMINISTIC), DocStatus, sha256
    embed.py                # Embedder: local(bge-m3) | gemini | offline(hashing)
    text/
      vi.py                 # NFC + word segmentation (underthesea/pyvi)
      alias.py              # "NĐ 13" <-> "Nghị định 13/2023/NĐ-CP"
    ingest/
      extract.py            # phân loại + trích text theo TRANG
      chunk.py              # chunking parent-child
      pipeline.py           # orchestrate + GATE
    index/
      __init__.py           # get_store() factory (numpy | pgvector)
      store.py              # numpy backend (file cục bộ)
      pg_store.py           # pgvector backend (PostgreSQL)
      bm25.py               # BM25 Okapi tự viết (cho numpy backend)
    retrieve/
      hybrid.py             # BM25 + vector + RRF
      rerank.py             # cross-encoder | llm(Gemini) | lexical
      pipeline.py           # retrieve() + build_context()
    generate/
      llm.py                # chat + vision: ollama|openai|gemini|offline
      answer.py             # prompt có nguồn + verify citation bằng code
    advanced/               # tầng 3
      classify_query.py | multihop.py | reflect.py | nli.py
    eval/
      harness.py            # đo hit@k, keyword_recall
```

---

## 4. Cài từ đầu (venv + deps)

Mở **Command Prompt (cmd)** tại thư mục `chatbot`:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -U pip
```

**Mạng công ty chặn TLS** (lỗi `CERTIFICATE_VERIFY_FAILED`) — cài ngay, dùng cho cả pip/HF/requests:
```bat
pip install pip-system-certs
```

Cài dependency theo môi trường muốn dùng (xem mục 5). Tối thiểu để chạy:
```bat
pip install numpy
```

---

## 5. Ba môi trường chạy

| Môi trường | Embedding | LLM | Chi phí | Dùng khi |
|---|---|---|---|---|
| **Offline** | hashing (giả) | extractive | 0đ | Smoke-test luồng |
| **Local** | bge-m3 (HF ~2GB) | Ollama/offline | 0đ | Test thật, tài liệu MẬT |
| **Cloud (Gemini)** | gemini-embedding-001 | gemini-flash-latest | tốn credit | Demo, tài liệu KHÔNG mật |

**Quy tắc:** một kho chỉ dùng MỘT embedding provider. Đổi provider = xoá `storage\` (hoặc DB) rồi ingest lại.

### 5a. Offline
```bat
pip install numpy
set RAG_OFFLINE=force
python cli.py ingest examples\nd13.md examples\hopdong.md
python cli.py eval examples\evalset.json
```
`stats` báo `dim: 256` = offline.

### 5b. Local bge-m3 (⚠️ CẦN Hugging Face — mạng công ty này bị CHẶN)
```bat
pip install numpy sentence-transformers
set RAG_OFFLINE=
set RAG_EMBED_PROVIDER=local
set RAG_EMBED_MODEL=BAAI/bge-m3
set RAG_EMBED_DIM=1024
```
> bge-m3 (~2GB) tự tải từ Hugging Face lần đầu. **Nếu mạng chặn HF** (`WinError 10054`), thử
> `set HF_ENDPOINT=https://hf-mirror.com` hoặc chép cache model từ máy có internet sang rồi
> `set HF_HUB_OFFLINE=1`. Nếu HF chặn hoàn toàn → dùng Gemini (mục 6).

### 5c. Gemini → xem mục 6.

---

## 6. ⭐ Cấu hình ĐANG CHẠY ĐƯỢC (Gemini + mạng công ty)

Đây là cấu hình đã chạy thành công đầu-cuối trên mạng công ty (thông Google, chặn Hugging Face).

### Bước 1 — Lấy API key
https://aistudio.google.com/apikey → **Create API key** → key dạng `AIza...`
- Nếu bị `429 hết credit`: tạo key trong **project MỚI chưa bật billing** → được free-tier (giới hạn tốc độ).
- API restriction (Cloud Console): chọn **Gemini API** (= Generative Language API), KHÔNG phải Agent Platform API.

### Bước 2 — Cài
```bat
pip install numpy requests pip-system-certs
```

### Bước 3 — Đặt key (1 lần, KHÔNG ghi vào file trong repo)
```bat
setx GEMINI_API_KEY "AIza...key_that"
```
Rồi **mở cmd MỚI** (setx chỉ có tác dụng ở cmd mở sau đó).

### Bước 4 — Nạp cấu hình + chạy
Dùng file `run_gemini.bat` (nạp hết biến), hoặc set tay:
```bat
run_gemini.bat

rmdir /s /q storage 2>nul
python cli.py ingest examples\nd13.md examples\hopdong.md
python cli.py stats
python cli.py ask "NĐ 13 định nghĩa dữ liệu cá nhân thế nào"
```

Cấu hình đầy đủ (nội dung `run_gemini.bat`):
```bat
set RAG_OFFLINE=
set RAG_EMBED_PROVIDER=gemini
set RAG_EMBED_MODEL=gemini-embedding-001
set RAG_EMBED_DIM=1536
set RAG_LLM_PROVIDER=gemini
set RAG_LLM_MODEL=gemini-flash-latest
set RAG_RERANKER=llm
set RAG_VISION_PROVIDER=gemini
set RAG_VISION_MODEL=gemini-flash-latest
```

**Dấu hiệu chạy thật:** `stats` báo `dim: 1536`, ingest KHÔNG còn dòng `[embed] fallback OFFLINE`,
`ask` trả câu tiếng Việt tự nhiên kèm `[1]`, `grounded=True`.

> Lưu ý: dùng `gemini-flash-latest` (alias) chứ đừng dùng tên bản cứng như `gemini-2.5-flash`
> — bản cứng có thể bị Google khoá với user mới (404).

---

## 7. Lưu vào database (pgvector)

Mặc định vector nằm ở file `storage\vectors.npy`. Muốn DB thật:

```bat
docker run -d --name rag-pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rag pgvector/pgvector:pg16

pip install psycopg2-binary
set RAG_STORE=pgvector
set RAG_PG_DSN=postgresql://postgres:postgres@localhost:5432/rag

rmdir /s /q storage 2>nul
python cli.py ingest examples\nd13.md examples\hopdong.md
python cli.py stats          REM báo backend: pgvector
```
Kiểm tra: `docker exec -it rag-pg psql -U postgres -d rag -c "SELECT count(*) FROM chunks;"`

- Vector: pgvector cosine `<=>` + index HNSW. Keyword: Postgres full-text search trên text đã segment VN.
- Chiều vector cố định khi tạo bảng (`RAG_EMBED_DIM`). Đổi embedding = tạo DB/bảng mới.
- Cần BM25 Okapi thật trong DB → extension `pg_search` (ParadeDB), đổi `search_bm25` trong `pg_store.py`.

---

## 8. Reranker — 3 chế độ

Chọn qua `RAG_RERANKER`:

| Giá trị | Cần gì | Chất lượng | Ghi chú |
|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` (mặc định) | Hugging Face | Cao nhất | Mạng công ty chặn HF → không dùng được |
| `llm` | Gemini (đang thông) | Gần cross-encoder | +1 lần gọi Gemini/câu, né HF hoàn toàn |
| `lexical` | Không cần gì | Thô (trùng từ) | Tức thì, miễn phí |

Với mạng công ty này: **`set RAG_RERANKER=llm`** (đã có trong `run_gemini.bat`).
Cơ chế `llm`: gửi Gemini danh sách đoạn đánh số → nhận mảng JSON thứ tự liên quan → lấy top 5.
Tự fallback về lexical nếu LLM lỗi/sai định dạng.

---

## 9. Vision OCR cho PDF scan

PDF scan (ảnh, 0 ký tự text) → GATE báo FAIL. Bật Vision để đọc:
```bat
pip install PyMuPDF          REM render trang PDF thành ảnh (không đụng HF)
set RAG_VISION_PROVIDER=gemini
set RAG_VISION_MODEL=gemini-flash-latest
python cli.py ingest examples\file_scan.pdf
```
Gemini đọc cả trang → Markdown (giữ bảng, dấu tiếng Việt, bỏ mộc/chữ ký).
Tài liệu `--confidential` KHÔNG gửi ảnh lên Gemini (tự ép về Ollama local).

---

## 10. Bảng biến môi trường

| Biến | Giá trị | Mặc định |
|---|---|---|
| `RAG_OFFLINE` | `force` \| *(rỗng)* \| `off` | `auto` |
| `RAG_STORE` | `numpy` \| `pgvector` | `numpy` |
| `RAG_PG_DSN` | chuỗi kết nối PostgreSQL | `postgresql://postgres:postgres@localhost:5432/rag` |
| `RAG_EMBED_PROVIDER` | `local` \| `gemini` \| `offline` | `local` |
| `RAG_EMBED_MODEL` | tên model | theo provider |
| `RAG_EMBED_DIM` | số chiều | bge-m3=1024, gemini=1536 |
| `GEMINI_API_KEY` | key `AIza...` | rỗng (đừng commit) |
| `RAG_RERANKER` | tên model \| `llm` \| `lexical` | `BAAI/bge-reranker-v2-m3` |
| `RAG_LLM_PROVIDER` | `ollama` \| `openai` \| `gemini` \| `offline` | `offline` |
| `RAG_LLM_MODEL` | tên model | `qwen2.5:7b` |
| `RAG_VISION_PROVIDER` | `ollama` \| `openai` \| `gemini` \| `offline` | `offline` |
| `RAG_VISION_MODEL` | tên model | `qwen2-vl` |
| `RAG_DATA_DIR` | đường dẫn kho | `storage\` |

> `set` chỉ sống trong cửa sổ cmd hiện tại. `setx` lưu lâu dài (áp dụng cmd mở MỚI).

---

## 11. Lệnh CLI

```bat
python cli.py ingest <file...> [--confidential] [--dept legal]   REM .pdf/.txt/.md
python cli.py ask "<câu hỏi>" [--dept X] [--no-clearance] [--advanced] [--nli]
python cli.py eval examples\evalset.json [-k 5]
python cli.py stats
```

---

## 12. 📓 Nhật ký xử lý sự cố

Toàn bộ lỗi đã gặp khi triển khai trên mạng công ty, theo thứ tự, để lần sau khỏi mất thời gian:

| # | Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|---|
| 1 | `No module named 'requests'` | Thiếu lib (Gemini/Ollama cần) | `python -m pip install requests` (đúng interpreter) |
| 2 | `CERTIFICATE_VERIFY_FAILED self-signed` | Proxy công ty giải mã TLS, Python không tin CA nội bộ | `pip install pip-system-certs` |
| 3 | `403 Forbidden` gọi Gemini | (ban đầu tưởng key sai) thực ra do billing | xem #4 |
| 4 | `429 RESOURCE_EXHAUSTED / prepay depleted` | **Ví prepay Gemini cạn** — KHÁC với credit Free Trial GCP (₫7.8M không chi cho Gemini) | Nạp ví Gemini, HOẶC tạo key free-tier ở project chưa bật billing |
| 5 | `401 Unauthorized`, URL vẫn `key=AQ...` cũ | Key mới set ở cửa sổ khác / `setx` cũ đè | `echo %GEMINI_API_KEY%` kiểm tra; set key + chạy python CÙNG cửa sổ |
| 6 | `404 model no longer available` (`gemini-2.5-flash`) | Bản model cứng bị khoá với user mới | Dùng alias `gemini-flash-latest` |
| 7 | `WinError 10054 ... huggingface.co` | Mạng công ty **chặn Hugging Face** (cả `hf-mirror.com`) | Bỏ Local bge-m3, dùng Gemini; hoặc chép cache model + `HF_HUB_OFFLINE=1` |
| 8 | `[rerank] fallback lexical` + retry HF chậm | Reranker tải từ HF (bị chặn) | `set RAG_RERANKER=llm` (Gemini) hoặc `lexical` |
| 9 | `curl: (35) schannel ... REVOCATION` | curl siết kiểm tra thu hồi cert | `curl --ssl-no-revoke ...` (Python không dính) |
| 10 | Biến `set` "mất" liên tục | `set` chỉ sống trong 1 cửa sổ cmd | Dùng `run_gemini.bat` nạp lại mỗi phiên; key thì `setx` |
| 11 | `[FAIL] ... 0 ký tự` khi ingest PDF | Thiếu PyMuPDF, HOẶC PDF là scan | `pip install PyMuPDF`; nếu scan → bật Vision (mục 9) |
| 12 | `could not broadcast (1024) vs (256)` | Trộn kho khác chiều vector | `rmdir /s /q storage` rồi ingest lại |
| 13 | `UnicodeEncodeError` (cp1252) | Console Windows | Đã ép UTF-8 trong `cli.py`; chỗ khác: `set PYTHONUTF8=1` |

**Bài học lớn nhất về mạng công ty này:** thông `generativelanguage.googleapis.com` (Google) và PyPI,
nhưng **chặn hoàn toàn Hugging Face**. => Mọi thứ chạy qua **Gemini**, tránh mọi phụ thuộc Hugging Face
(embedding local, reranker cross-encoder). PyMuPDF/psycopg2/requests trên PyPI thì cài bình thường.

---

## 13. Quyết định khó đảo ngược (đã xử lý sẵn trong code)

1. **Embedding model** lưu vào metadata TỪNG vector → migrate từng phần. Đổi chiều = báo lỗi ngay.
2. **chunk_id deterministic** (`doc::pN::cNNN`) → citation cũ không chết khi re-index.
3. **RBAC ở tầng retrieval** (lọc trong query vector store), KHÔNG lọc sau khi model đã đọc.

---

## 14. Checklist dựng lại nhanh

Trên máy/mạng công ty mới, chạy theo thứ tự:

```bat
REM 1) venv
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -U pip

REM 2) fix TLS proxy + deps cho Gemini
pip install pip-system-certs numpy requests

REM 3) đặt key 1 lần rồi MỞ CMD MỚI
setx GEMINI_API_KEY "AIza...key_that"

REM --- mở cmd mới, vào lại thư mục + activate venv ---
.venv\Scripts\activate.bat

REM 4) nạp cấu hình Gemini
run_gemini.bat

REM 5) chạy
rmdir /s /q storage 2>nul
python cli.py ingest examples\nd13.md examples\hopdong.md
python cli.py stats
python cli.py ask "NĐ 13 định nghĩa dữ liệu cá nhân thế nào"

REM (tuỳ chọn) 6) DB thật thay file:
REM   docker run -d --name rag-pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rag pgvector/pgvector:pg16
REM   pip install psycopg2-binary
REM   set RAG_STORE=pgvector
REM   set RAG_PG_DSN=postgresql://postgres:postgres@localhost:5432/rag
```

**Nghiệm thu đạt:** `stats` = `dim: 1536` (hoặc `backend: pgvector`); `ask` trả lời tiếng Việt kèm `[1]`,
`grounded=True`, không có dòng `fallback OFFLINE`.

---

## Bước tiếp theo nên làm (không phải tầng 4)

1. **Mở rộng eval set lên 50-100 câu** từ tài liệu thật → biết retrieval đủ tốt chưa.
2. Chỉ bật **1-2 mảnh tầng 3** nếu eval lộ điểm yếu cụ thể (đừng bật cả bộ).
3. Xin IT **whitelist huggingface.co** → mở khoá bge-m3 + cross-encoder thật (thoát phụ thuộc Gemini).
4. Tầng 4 (agentic/connectors) chỉ khi có nhu cầu thật: kết hợp nhiều nguồn, sync Drive/SharePoint, AI hành động.
 