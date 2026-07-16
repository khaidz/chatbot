"""Eval harness — ĐIỀU KIỆN TIÊN QUYẾT trước khi tối ưu bất cứ gì.

Format evalset.json:
  {"q": "...", "expect_doc": "nd13", "keywords": ["..."]}   <- câu CÓ đáp án
  {"q": "...", "expect_doc": null,   "keywords": []}         <- câu KHÔNG có đáp án trong kho

Câu có đáp án: đo hit@k (đúng tài liệu trong top-k) + keyword_recall (từ khoá trong context).
Câu không có đáp án: retrieval luôn trả gì đó — ghi nhận top_score để cân NGƯỠNG TỰ TIN
(Todo.md mục 2): nếu phân bố score hai nhóm tách nhau, in luôn vùng ngưỡng gợi ý.
"""
import json
from pathlib import Path

from rag.retrieve.pipeline import build_context, retrieve
from rag.text.vi import normalize


def run_eval(path: str, k: int = 5) -> dict:
    # utf-8-sig: chấp nhận cả file có BOM (Notepad/PowerShell trên Windows hay chèn)
    items = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    answerable = [it for it in items if it.get("expect_doc")]
    no_answer = [it for it in items if not it.get("expect_doc")]

    print(f"Eval {len(items)} câu (k={k}) — {len(answerable)} có đáp án, "
          f"{len(no_answer)} không có đáp án trong kho:")

    hits = 0
    recalls: list[float] = []
    ans_scores: list[float] = []
    for it in answerable:
        parents = retrieve(it["q"])[:k]
        top = max((p.score for p in parents), default=0.0)
        ans_scores.append(top)
        context, _ = build_context(parents)
        ctx = normalize(context).lower()

        hit = any(p.doc_id == it["expect_doc"] for p in parents)
        hits += hit
        kws = it.get("keywords", [])
        found = sum(1 for kw in kws if normalize(kw).lower() in ctx)
        recalls.append(found / len(kws) if kws else 1.0)

        mark = "OK  " if hit else "MISS"
        print(f"  [{mark}] score={top:.4f} kw={found}/{len(kws) or '-'}  {it['q'][:58]}")

    na_scores: list[float] = []
    for it in no_answer:
        parents = retrieve(it["q"])[:k]
        top = max((p.score for p in parents), default=0.0)
        na_scores.append(top)
        print(f"  [NA  ] score={top:.4f}          {it['q'][:58]}")

    n = len(answerable) or 1
    summary = {
        "n": len(items),
        "hit_at_k": hits / n,
        "keyword_recall": sum(recalls) / n,
        "ans_score_min": min(ans_scores) if ans_scores else 0.0,
        "na_score_max": max(na_scores) if na_scores else 0.0,
    }
    print(f"\nhit@{k}: {summary['hit_at_k']:.2%}   keyword_recall: {summary['keyword_recall']:.2%}")
    if ans_scores:
        print(f"score câu có đáp án   : min={min(ans_scores):.4f} "
              f"avg={sum(ans_scores)/len(ans_scores):.4f} max={max(ans_scores):.4f}")
    if na_scores:
        print(f"score câu KHÔNG đáp án: min={min(na_scores):.4f} "
              f"avg={sum(na_scores)/len(na_scores):.4f} max={max(na_scores):.4f}")
    if ans_scores and na_scores:
        lo, hi = max(na_scores), min(ans_scores)
        if lo < hi:
            print(f"=> Hai nhóm TÁCH NHAU: ngưỡng tự tin đặt trong khoảng ({lo:.4f} — {hi:.4f})")
        else:
            print(f"=> Hai nhóm CHỒNG LẤN (max no-answer {lo:.4f} >= min answered {hi:.4f}) "
                  "— chưa đặt ngưỡng cứng được")
    if summary["hit_at_k"] < 0.8:
        print("=> retrieval còn yếu: soi lại ingestion/alias/segment TRƯỚC, đừng vội bật tầng 3.")
    return summary
