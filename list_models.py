"""Liệt kê model Gemini mà GEMINI_API_KEY hiện tại được phép dùng.

Chạy:  run_gemini.bat  (hoặc set GEMINI_API_KEY=...)  rồi  python list_models.py
In 2 nhóm: EMBEDDING (embedContent) và GENERATE (generateContent — dùng cho LLM/Vision/rerank).
"""
import sys

import requests

import config

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def main():
    if not config.GEMINI_API_KEY:
        print("[!] Thiếu GEMINI_API_KEY — chạy run_gemini.bat trước.")
        sys.exit(1)

    models = []
    page_token = ""
    while True:
        url = f"{config.GEMINI_BASE}/models?pageSize=200&key={config.GEMINI_API_KEY}"
        if page_token:
            url += f"&pageToken={page_token}"
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print(f"[!] HTTP {r.status_code}: {r.text[:400]}")
            sys.exit(1)
        data = r.json()
        models.extend(data.get("models", []))
        page_token = data.get("nextPageToken", "")
        if not page_token:
            break

    def row(m):
        name = m["name"].removeprefix("models/")
        return f"  {name:42} | in:{m.get('inputTokenLimit','?'):>8} out:{m.get('outputTokenLimit','?'):>7}"

    embed = [m for m in models if "embedContent" in m.get("supportedGenerationMethods", [])]
    gen = [m for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]

    print(f"Tổng: {len(models)} model khả dụng với key này\n")
    print(f"=== EMBEDDING ({len(embed)}) — dùng cho RAG_EMBED_MODEL ===")
    for m in sorted(embed, key=lambda x: x["name"]):
        print(row(m))
    print(f"\n=== GENERATE ({len(gen)}) — dùng cho RAG_LLM_MODEL / RAG_VISION_MODEL ===")
    for m in sorted(gen, key=lambda x: x["name"]):
        print(row(m))

    print(
        "\nGợi ý đổi model khi 429 (mỗi model một bucket quota riêng):\n"
        "  Embedding : set RAG_EMBED_MODEL=<model nhóm EMBEDDING ở trên> (+ RAG_EMBED_DIM khớp)\n"
        "              Đổi embedding = XOÁ kho rồi ingest lại — numpy: rmdir /s /q storage;\n"
        "              pgvector: DROP TABLE chunks, docs; (vector 2 model không so sánh được)\n"
        "  LLM/Vision: đổi RAG_LLM_MODEL / RAG_VISION_MODEL sang model nhóm GENERATE\n"
        "              (không cần re-ingest; ưu tiên alias *-latest / flash-lite)"
    )


if __name__ == "__main__":
    main()
