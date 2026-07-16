"""CLI: ingest / ask / eval / stats. Chạy: python cli.py <lệnh> ...

Console Windows mặc định cp1252 -> ép UTF-8 ngay tại đây (sự cố #13 trong cẩm nang).
"""
import argparse
import glob
import sys

# Ép UTF-8 cho console Windows (tránh UnicodeEncodeError cp1252)
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def cmd_ingest(args):
    from rag.ingest.pipeline import ingest_files

    paths: list[str] = []
    for pattern in args.files:  # cmd Windows không tự expand glob
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])
    results = ingest_files(paths, dept=args.dept, confidential=args.confidential)
    failed = [r for r in results if r[1].value == "failed"]
    sys.exit(1 if failed else 0)


def cmd_ask(args):
    from rag.generate.answer import answer
    from rag.retrieve.pipeline import build_context, retrieve

    dept = args.dept
    clearance = not args.no_clearance
    parents = None

    if args.advanced:
        from rag.advanced.classify_query import classify_query

        qtype = classify_query(args.question)
        print(f"[advanced] loại câu hỏi: {qtype}")
        if qtype == "smalltalk":
            print("Xin chào! Hãy hỏi tôi về nội dung các tài liệu đã nạp.")
            return
        if qtype == "multihop":
            from rag.advanced.multihop import retrieve_multihop

            parents = retrieve_multihop(args.question, dept, clearance)

    result = answer(args.question, dept=dept, clearance=clearance, parents=parents)

    if args.nli and result["sources"]:
        from rag.advanced.nli import check

        ctx_parents = parents if parents is not None else retrieve(args.question, dept, clearance)
        context, _ = build_context(ctx_parents)
        suspects = check(result["answer"], context)
        for s in suspects:
            print(f"[nli] CẢNH BÁO câu không được nguồn hỗ trợ: {s[:100]}")

    print("\n" + result["answer"])
    if result["sources"]:
        print("\nNguồn:")
        for s in result["sources"]:
            print(
                f"  [{s['n']}] {s['source']} — trang {s['page']} — "
                f"liên quan {s['rel']:.0%} (rrf {s['score']:.4f}) ({s['chunk_id']})"
            )
    print(f"\ngrounded={result['grounded']}  mode={result['mode']}")


def cmd_eval(args):
    from rag.eval.harness import run_eval

    run_eval(args.evalset, k=args.k)


def cmd_stats(args):
    from rag.embed import get_embedder
    from rag.index import get_store

    emb = get_embedder()
    print(f"embedder : {emb.provider}/{emb.model}  dim: {emb.dim}")
    for k, v in get_store().stats().items():
        print(f"{k:9}: {v}")


def main():
    ap = argparse.ArgumentParser(prog="cli.py", description="RAG chatbot tiếng Việt")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="nạp tài liệu (.pdf/.txt/.md)")
    p.add_argument("files", nargs="+")
    p.add_argument("--confidential", action="store_true", help="tài liệu MẬT (không gửi cloud)")
    p.add_argument("--dept", default="", help="phòng ban sở hữu (RBAC)")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("ask", help="hỏi đáp có trích nguồn")
    p.add_argument("question")
    p.add_argument("--dept", default="")
    p.add_argument("--no-clearance", action="store_true", help="không có quyền xem tài liệu mật")
    p.add_argument("--advanced", action="store_true", help="bật tầng 3: classify + multihop")
    p.add_argument("--nli", action="store_true", help="bật NLI check từng câu trả lời")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("eval", help="chạy eval set")
    p.add_argument("evalset")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("stats", help="thông tin kho")
    p.set_defaults(fn=cmd_stats)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
