# TODO — việc nên làm tiếp (cập nhật 16/07/2026)

Nguyên tắc chung: **đo trước, tối ưu sau** — mọi mục dưới đây (trừ #1) chỉ làm khi eval set
chứng minh cần. Đừng làm cả bộ "cho đủ".

## 1. 🥇 Mở rộng eval set lên 50–100 câu  ← LÀM TRƯỚC, là điều kiện của mọi mục sau

- [x] ~~Sinh nháp 50 câu~~ (xong 16/07/2026): `examples\evalset_draft.json` — 28 câu `game`
      + 12 câu `lqmb94` + 10 câu no-answer (`expect_doc: null`). Mọi keyword đã lint
      khớp nguyên văn tài liệu trong kho. Có trường `type`: direct/paraphrase/number/no-answer.
- [x] ~~Harness hỗ trợ câu no-answer~~: `expect_doc: null` → ghi top_score riêng,
      tự gợi ý vùng ngưỡng tự tin khi 2 nhóm tách nhau.
- [ ] **NGƯỜI DUYỆT LẠI** nháp: sửa câu tối nghĩa, thêm câu theo cách người dùng thật hay hỏi
- [ ] Chạy lần đầu với Gemini: `run_gemini.bat` rồi `python cli.py eval examples\evalset_draft.json`
      → hit@5, keyword_recall, phân bố score 2 nhóm (số liệu nền cho mục 2 và 3)
- [ ] Nạp lại tài liệu nào còn dùng (nd13/hopdong/nhnn đã bị xoá khỏi kho) và bổ sung
      câu hỏi cho tài liệu mới khi kho phình ra — hướng tới 100 câu
- Lưu ý: eval cũ 8 câu (`evalset.json`) hỏi về nd13/hopdong — chỉ dùng lại khi nạp lại 2 file đó

## 2. 🥈 Ngưỡng tự tin (confidence threshold) — van chặn rác trước cửa LLM

- [x] ~~Query log~~ (xong 16/07/2026): mọi lượt hỏi ghi `top_score/n_sources/grounded/mode`
      vào bảng `query_log` (pgvector) hoặc `query_log.jsonl` (numpy).
      Xem + thống kê phân bố score: `python cli.py log` — tự gợi ý vùng ngưỡng khi
      hai nhóm (trả lời được / không tìm thấy) tách nhau.
- [ ] Tích luỹ log từ dùng thật vài tuần HOẶC chạy eval (sau khi có nhóm "câu không có
      đáp án" ở mục 1) → xem phân bố score hai nhóm tách nhau ở đâu (`python cli.py log`)
- [ ] Chốt ngưỡng từ số liệu (thang RRF hiện tại: 0.0125–0.0328; vùng dự kiến ~0.016–0.018)
- [ ] Cài vào `answer()`/`chat_turn()`: score nguồn tốt nhất < ngưỡng → trả thẳng
      "Không tìm thấy", KHÔNG gọi LLM (0 lượt Gemini, ~0ms) — `score` đã có sẵn trong pipeline
- Được gì: chống bịa từ gốc (không phó thác việc từ chối cho LLM), tiết kiệm 2 lượt
  Gemini/câu lạc đề, hành vi từ chối nhất quán 100%
- Rủi ro nếu đặt mò: ngưỡng cao quá → từ chối oan câu hợp lệ diễn đạt khác từ ngữ
  (mất recall). **Không đoán ngưỡng — phải đo.**

## 3. 🥉 Chunk theo cấu trúc "Điều/Khoản" (văn bản pháp luật/hợp đồng)

- [ ] Parent split ưu tiên ranh giới `^Điều \d+` thay vì cắt cứng ~1200 ký tự
      (hiện một Điều có thể bị xẻ đôi, một chunk chứa nửa Điều này + nửa Điều kia)
- [ ] Đưa số Điều vào metadata → nguồn hiển thị "nhnn.pdf — Điều 12, trang 8"
- [ ] So eval trước/sau để xác nhận đáng giữ
- Loại việc "80% chất lượng nằm ở ingestion"

## Chờ eval chứng minh mới làm (đừng làm trước)

- [ ] Alias "Điều 9", "khoản 2 Điều 11" trong `text/alias.py` — chỉ khi eval lộ câu hỏi theo điều khoản bị trượt
- [ ] Bật từng mảnh tầng 3 (multihop / reflect / NLI — code sẵn, đang tắt) — chỉ bật mảnh đúng triệu chứng eval chỉ ra
- [ ] Xin IT whitelist `huggingface.co` → mở khoá bge-m3 + cross-encoder thật, thoát phụ thuộc quota Gemini (việc gửi email, không phải code)

## Việc sản phẩm (không phải chất lượng RAG)

- [ ] Multi-user cho server: connection pool + bỏ lock toàn cục (hiện serialize mọi request — đủ cho nội bộ/single-user)
- [ ] Auth cho web UI nếu mở ra LAN (`host="0.0.0.0"`)
- [ ] Prod: `set RAG_MAX_RETRIES=2` (retry lỗi tạm 429/500/503)
