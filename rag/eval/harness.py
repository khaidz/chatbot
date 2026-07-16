"""Eval harness — ĐIỀU KIỆN TIÊN QUYẾT trước khi tối ưu bất cứ gì.

Format evalset.json: [{"q": "...", "expect_doc": "nd13", "keywords": ["...", ...]}, ...]
Đo: hit@k (đúng tài liệu trong top-k) + keyword_recall (từ khoá đáp án có trong context).
"""
import json
from pathlib import Path

from rag.retrieve.pipeline import build_context, retrieve
from rag.text.vi import normalize


def run_eval(path: str, k: int = 5) -> dict:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    hits = 0
    recalls: list[float] = []
    print(f"Eval {len(items)} câu (k={k}):")
    for it in items:
        q = it["q"]
        parents = retrieve(q)[:k]
        context, _ = build_context(parents)
        ctx = normalize(context).lower()

        hit = any(p.doc_id == it.get("expect_doc") for p in parents)
        hits += hit

        kws = it.get("keywords", [])
        found = sum(1 for kw in kws if normalize(kw).lower() in ctx)
        recall = found / len(kws) if kws else 1.0
        recalls.append(recall)

        mark = "OK  " if hit else "MISS"
        print(f"  [{mark}] hit={int(hit)} kw={found}/{len(kws) or '-'}  {q[:60]}")

    n = len(items) or 1
    summary = {"n": len(items), "hit_at_k": hits / n, "keyword_recall": sum(recalls) / n}
    print(f"\nhit@{k}: {summary['hit_at_k']:.2%}   keyword_recall: {summary['keyword_recall']:.2%}")
    if summary["hit_at_k"] < 0.8:
        print("=> retrieval còn yếu: soi lại ingestion/alias/segment TRƯỚC, đừng vội bật tầng 3.")
    return summary
