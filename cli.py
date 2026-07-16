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

    _print_result(result)


def _print_result(result: dict):
    print("\n" + result["answer"])
    if result["sources"]:
        print("\nNguồn:")
        for s in result["sources"]:
            print(
                f"  [{s['n']}] {s['source']} — trang {s['page']} — "
                f"liên quan {s['rel']:.0%} (rrf {s['score']:.4f}) ({s['chunk_id']})"
            )
    print(f"\ngrounded={result['grounded']}  mode={result['mode']}")


def cmd_chat(args):
    from rag.chat import get_chat_store
    from rag.chat.pipeline import chat_turn

    store = get_chat_store()

    if args.list:
        sessions = store.list_sessions()
        if not sessions:
            print("Chưa có phiên chat nào.")
            return
        for s in sessions:
            print(f"  {s['session_id']}  [{s['n_messages']:>3} msg]  "
                  f"{s['created_at'][:16]}  {s['title'] or '(chưa có tiêu đề)'}")
        return

    if args.session:
        sid = args.session
        if store.get_session(sid) is None:
            print(f"[!] Không tìm thấy session '{sid}' — xem: python cli.py chat --list")
            sys.exit(1)
    else:
        sid = store.create_session(dept=args.dept, clearance=not args.no_clearance)
        print(f"[chat] phiên mới: {sid}  (nối lại: python cli.py chat --session {sid})")

    def one_turn(q: str):
        result = chat_turn(store, sid, q)
        if result["standalone"] != q:
            print(f"[condense] → {result['standalone']}")
        _print_result(result)

    if args.once:
        one_turn(args.once)
        return

    print("Gõ câu hỏi (exit/quit/thoát để dừng):")
    while True:
        try:
            q = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "thoát", "thoat"):
            break
        one_turn(q)
    print(f"\n[chat] đã lưu. Nối lại: python cli.py chat --session {sid}")


def cmd_log(args):
    from rag.querylog import read_log

    rows = read_log()
    if not rows:
        print("Chưa có lượt hỏi nào được ghi log.")
        return

    print(f"── {min(args.tail, len(rows))} lượt gần nhất (tổng {len(rows)}) ──")
    for r in rows[-args.tail:]:
        g = "✓" if r["grounded"] else "✗"
        sid = f" ({r['session_id']})" if r.get("session_id") else ""
        print(f"{r['ts'][:16]}  score={r['top_score']:.4f}  nguồn={r['n_sources']}  "
              f"{g} [{r['mode']}]{sid}  {r['question'][:55]}")
        if r.get("standalone"):
            print(f"{'':18}↳ đã hiểu là: {r['standalone'][:65]}")

    # thống kê phục vụ cân chỉnh ngưỡng tự tin (Todo.md mục 2)
    answered = [r["top_score"] for r in rows if r["n_sources"] > 0]
    notfound = [r["top_score"] for r in rows if r["n_sources"] == 0]
    print(f"\n── Thống kê ({len(rows)} lượt) ──")
    print(f"Trả lời được   : {len(answered):>4} lượt", end="")
    if answered:
        print(f"  | top_score min={min(answered):.4f} avg={sum(answered)/len(answered):.4f} max={max(answered):.4f}")
    else:
        print()
    print(f"Không tìm thấy : {len(notfound):>4} lượt", end="")
    if notfound:
        print(f"  | top_score min={min(notfound):.4f} avg={sum(notfound)/len(notfound):.4f} max={max(notfound):.4f}")
    else:
        print()
    if answered and notfound:
        lo, hi = max(notfound), min(answered)
        if lo < hi:
            print(f"=> Hai nhóm TÁCH NHAU: ngưỡng tự tin đặt trong khoảng ({lo:.4f} — {hi:.4f})")
        else:
            print(f"=> Hai nhóm CHỒNG LẤN (max not-found {lo:.4f} >= min answered {hi:.4f}) "
                  "— chưa đặt ngưỡng được, cần thêm dữ liệu/cải thiện retrieval")


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

    p = sub.add_parser("ingest", help="nạp tài liệu (.pdf/.docx/.txt/.md)")
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

    p = sub.add_parser("chat", help="hội thoại đa phiên, nhớ ngữ cảnh")
    p.add_argument("--session", default="", help="nối lại phiên cũ theo id")
    p.add_argument("--list", action="store_true", help="liệt kê các phiên đã có")
    p.add_argument("--once", default="", metavar="CÂU_HỎI",
                   help="hỏi đúng 1 câu vào phiên rồi thoát (dùng cho script/test)")
    p.add_argument("--dept", default="", help="RBAC của phiên (đặt lúc TẠO, cố định cả phiên)")
    p.add_argument("--no-clearance", action="store_true")
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("eval", help="chạy eval set")
    p.add_argument("evalset")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("log", help="xem query log + thống kê score (cân ngưỡng tự tin)")
    p.add_argument("--tail", type=int, default=20, help="số lượt gần nhất hiển thị")
    p.set_defaults(fn=cmd_log)

    p = sub.add_parser("stats", help="thông tin kho")
    p.set_defaults(fn=cmd_stats)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
