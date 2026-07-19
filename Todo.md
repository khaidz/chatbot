# TODO — việc nên làm tiếp (cập nhật 18/07/2026)

> **Chỉ chạy CLOUD (quyết định).** Đã GỠ hoàn toàn offline (hashing giả) + local/on-prem
> (Ollama/bge-m3/cross-encoder HF). Toàn hệ thống chạy **Gemini** (embed/LLM/vision); LLM/Vision
> đổi được sang **OpenAI**. Thiếu key/lỗi provider → báo lỗi rõ (không fallback giả); rerank &
> answer có dự phòng thuần-Python (lexical/extractive).
> ⇒ **KHÔNG dùng cho tài liệu mật cần ở-lại-nội-bộ** (text đi lên Google/OpenAI). Nếu sau này
> cần on-prem lại → khôi phục provider local từ lịch sử git.

Nguyên tắc chung: **đo trước, tối ưu sau** — mọi mục chất lượng (2, 3...) chỉ làm khi eval set
chứng minh cần. Đừng làm cả bộ "cho đủ".

## 0. ✅ Chỉ chạy cloud — ĐÃ XONG

- [x] ~~Gỡ toàn bộ offline/local~~: bỏ `RAG_OFFLINE`, `offline_forced/banned`, embedder hashing +
      bge-m3, provider `ollama`, cross-encoder HF, `run_offline.bat`, dòng `sentence-transformers`.
      config mặc định gemini; thiếu key → lỗi rõ. Dự phòng thuần-Python: rerank `lexical` +
      answer `extractive` khi Gemini lỗi tạm. Docs (README/handbook) đã đồng bộ.

## 1. 🥇 Mở rộng + DUYỆT eval set  ← điều kiện của mọi mục tối ưu sau

- [x] ~~Dựng lại HOÀN TOÀN (18/07/2026)~~: `examples\evalset_draft.json` — **68 câu** bám
      đúng KHO thực tế (8 doc: game, hopdongvaycamcotructuyen_namabank, lqmb94, luatthue,
      noi_quy_phong_thi, phongchongkhungbo, phongchongruatien, pl_5). 56 câu có đáp án +
      12 no-answer. Mọi keyword đã LINT khớp nguyên văn text đã lưu trong kho (68/68 pass).
      ⚠️ Lưu ý: `nhnn` KHÔNG có trong kho (chưa ingest); `luatthue`+`phongchongkhungbo` là scan (OCR).
- [x] ~~Harness hỗ trợ câu no-answer~~: `expect_doc: null` → ghi top_score riêng.
- [x] ~~Chạy eval lần đầu (18/07/2026)~~: **hit@5 = 100%, keyword_recall = 100%** trên Gemini.
      Retrieval MẠNH (kể cả 2 doc scan). Ngoặc: kho còn nhỏ + câu hỏi soạn từ tài liệu →
      100% là tín hiệu tốt, chưa chắc giữ khi kho phình lớn.
- [ ] **NGƯỜI DUYỆT LẠI** 68 câu (nhất là câu OCR có lỗi khoảng trắng ở `game`/`luatthue`);
      chốt tên chính thức (đổi `evalset_draft.json` → `evalset.json`)
- [ ] Bổ sung câu cho tài liệu mới khi ingest thêm (vd nạp `nhnn` rồi thêm câu) — hướng tới 80–100 câu

## 2. 🥈 Ngưỡng tự tin (confidence threshold) — van chặn rác trước cửa LLM

- [x] ~~Query log + thống kê~~: mọi lượt hỏi ghi `top_score/n_sources/grounded/mode` + latency
      vào `query_log` (pgvector) hoặc `query_log.jsonl` (numpy). `python cli.py log` tự gợi ý
      vùng ngưỡng khi hai nhóm (trả lời được / không tìm thấy) tách nhau.
- [x] ~~Cài van RRF `SCORE_MIN`~~ (mặc định **0.017**) trong `config.py` + `answer()`.
- [x] ~~ĐO: RRF top_score KHÔNG tách được câu lạc đề~~ (18/07/2026): eval sạch cho thấy
      no-answer đạt tới 0.0325, vượt cả answered min 0.0272 → **không ngưỡng RRF nào tách nổi**
      (RRF theo hạng nên đoạn top-1 luôn ~cùng điểm). RRF bị loại làm tín hiệu van chính.
      (May: min-answered 0.0272 > 0.017 nên `SCORE_MIN=0.017` KHÔNG hại câu hợp lệ, còn bắt
      được 2 no-answer điểm thấp 0.0154/0.0164 → giữ như bộ lọc rẻ phụ trợ.)
- [x] ~~Van OUT-OF-DOMAIN qua RERANKER (18/07/2026)~~: `rerank._llm` trả `[]` khi Gemini
      phán "không đoạn nào liên quan" → `retrieve()` trả [] → `answer()` no-context, KHÔNG gọi
      LLM sinh câu. Nghiệm thu: hit@5 vẫn 100% (không hại câu hợp lệ); **7/12 no-answer bị chặn
      (score→0)**; cộng SCORE_MIN chặn thêm 2 → **9/12** câu lạc đề bị từ chối đúng.
- [x] ~~Kiểm end-to-end 3 câu cận-domain còn lọt ở tầng retrieval (18/07/2026)~~: chạy `ask`
      thật → CẢ 3 (thuế VAT / tướng Murad / BHXH) đều bị `answer()` từ chối đúng ("Không tìm thấy",
      grounded=True) nhờ prompt rule #3 + verify citation. ⇒ **end-to-end 12/12 câu ngoài phạm vi
      bị từ chối**: 7 ở reranker (0 LLM), 2 ở SCORE_MIN (0 LLM), 3 ở LLM answer (1 call như thường).
- [~] Bước "judge relevance 0–1" → **KHÔNG làm**: sẽ tốn +1 call trên MỌI câu chỉ để tiết kiệm
      1 call trên 3 câu cận-domain vốn đã trả đúng → lỗ chi phí. Van coi như ĐỦ cho kho hiện tại.
- [ ] Theo dõi dùng thật: nếu xuất hiện câu cận-domain mà LLM answer BỊA (không tự nói "không tìm
      thấy") → lúc đó mới cân bước judge. Chưa thấy trường hợp nào.
- [ ] Chỉ khi lên on-prem: cross-encoder cho điểm relevance TUYỆT ĐỐI → van chặt hơn nữa (tuỳ chọn).

## 3. 🥉 Chunk theo cấu trúc "Điều/Khoản" (văn bản pháp luật/hợp đồng)

- [ ] Parent split ưu tiên ranh giới `^Điều \d+` thay vì cắt cứng ~1200 ký tự
      (hiện một Điều có thể bị xẻ đôi, một chunk chứa nửa Điều này + nửa Điều kia)
- [ ] Đưa số Điều vào metadata → nguồn hiển thị "nhnn.pdf — Điều 12, trang 8"
- [ ] So eval trước/sau để xác nhận đáng giữ
- Đặc biệt đáng giá với bộ tài liệu mới (hợp đồng, luật thuế, phòng chống rửa tiền — toàn văn bản có Điều/Khoản)

## Chờ eval chứng minh mới làm (đừng làm trước)

- [ ] Alias "Điều 9", "khoản 2 Điều 11" trong `text/alias.py` — chỉ khi eval lộ câu hỏi theo điều khoản bị trượt
- [ ] Bật từng mảnh tầng 3 (multihop / reflect / NLI — code sẵn, đang tắt) — chỉ bật mảnh đúng triệu chứng eval chỉ ra
- [ ] **Adaptive retrieval**: score thấp → LLM viết lại câu hỏi thử lại 1 lần → vẫn thấp mới
      trả "không tìm thấy". Bản nâng cấp của mục 2 — chỉ làm khi query log cho thấy nhiều câu
      hợp lệ bị điểm thấp.
- [ ] **Xem lại hiệu quả cache sau vài tuần dùng thật**: `python cli.py log` → tỷ lệ hit =
      M/(M+N) từ dòng "Latency TB (N lượt tươi, M lượt cache)". >15-20% = gánh quota đáng kể;
      <5% = chỉ là bảo hiểm demo (vẫn giữ, miễn phí). Nhiều câu "cùng ý khác chữ" bị miss →
      cân nhắc semantic cache (có rủi ro trả nhầm — cần cẩn trọng).

## Việc sản phẩm (không phải chất lượng RAG)

- [x] ~~Multi-user cho server~~: đã BỎ lock toàn cục + connection pool chung (`rag/db.py`,
      semaphore chờ khi hết slot), timing thread-local, singleton (store/embedder/chat/tokenizer)
      hâm nóng lúc startup, numpy/JsonChatStore tự khoá bộ nhớ. Nghiệm thu: 30 phiên pgvector đồng
      thời (60 lượt) + ingest song song qua pool 8 → 0 lỗi; 20 client HTTP đồng thời → 0 lỗi.
      Chỉnh số connection: `RAG_PG_POOL_MAX` (mặc định 16).
- [ ] Auth cho web UI nếu mở ra LAN (`server.py` hiện `host=127.0.0.1`, port 8000)
- [ ] Prod: `set RAG_MAX_RETRIES=2` (retry lỗi tạm 429/500/503)
- [ ] **Đồng bộ docs**: `README.md` + `handbook.md` vừa cập nhật cho khớp reality (3 provider,
      config default local, examples mới). Nếu đổi default/bat sau này nhớ sửa cả hai.

## Đã xong (tham chiếu)

- [x] **Báo cáo trích xuất per-trang** (`rag/ingest/report.py`) — mỗi trang bắt lỗi riêng (1 trang
      hỏng không giết cả tài liệu), lưu **dual-backend**: pgvector → bảng `ingest_reports` (query/theo
      dõi bằng SQL), numpy → file JSON. Trạng thái ok/ocr_failed/ocr_skipped/blank/error; message ingest
      nêu rõ trang hỏng; `GET /api/documents/<id>/report` + nút 🔍 trên trang Tài liệu
- [x] Answer cache exact-match (`rag/cache.py`) — key gồm dept/clearance, tự vô hiệu khi kho đổi
- [x] Query log + latency từng khâu (`rag/querylog.py`, `rag/timing.py`)
- [x] Streaming đáp án SSE (`chat_turn_stream()` + endpoint stream + UI hiện chữ dần)
- [x] Chat đa phiên nhớ ngữ cảnh (`rag/chat/`, condense + tóm tắt dần)
- [x] Chặn smalltalk trước cửa RAG (`rag/advanced/smalltalk.py`) — chào/cảm ơn/ok không tốn LLM
- [x] Đọc `.docx` (đoạn văn + bảng → Markdown)
- [x] Timeout gọi LLM/embedding rõ ràng (`RAG_TIMEOUT`/`RAG_VISION_TIMEOUT`)
