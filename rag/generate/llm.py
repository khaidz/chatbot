"""Chat + Vision cho các provider: ollama | openai | gemini | offline.

Lưu ý model Gemini: dùng alias `gemini-flash-latest`, KHÔNG dùng tên bản cứng
(gemini-2.5-flash có thể bị Google khoá với user mới -> 404).
"""
import base64

import config


def chat(prompt: str, system: str | None = None, provider: str | None = None,
         model: str | None = None) -> str:
    provider = provider or config.LLM_PROVIDER
    model = model or config.LLM_MODEL
    if config.offline_forced() or provider == "offline":
        return ""  # caller (answer.py) tự chuyển sang extractive
    if provider == "gemini":
        return _gemini_generate(model, [{"text": prompt}], system)
    if provider == "ollama":
        return _ollama_chat(model, prompt, system)
    if provider == "openai":
        return _openai_chat(model, prompt, system)
    raise ValueError(f"LLM provider chưa hỗ trợ: {provider}")


def vision_ocr(png_bytes: bytes, prompt: str, provider: str | None = None,
               model: str | None = None) -> str:
    provider = provider or config.VISION_PROVIDER
    model = model or config.VISION_MODEL
    if provider == "offline":
        return ""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    if provider == "gemini":
        parts = [{"inline_data": {"mime_type": "image/png", "data": b64}}, {"text": prompt}]
        return _gemini_generate(model, parts, None)
    if provider == "ollama":
        return _ollama_chat(model, prompt, None, images=[b64])
    if provider == "openai":
        return _openai_chat(model, prompt, None, image_b64=b64)
    raise ValueError(f"Vision provider chưa hỗ trợ: {provider}")


# ---------- backends ----------
def _gemini_generate(model: str, parts: list[dict], system: str | None) -> str:
    from rag.net import post_json

    if not config.GEMINI_API_KEY:
        raise RuntimeError("Thiếu GEMINI_API_KEY (setx GEMINI_API_KEY \"AIza...\" rồi mở cmd MỚI)")
    url = f"{config.GEMINI_BASE}/models/{model}:generateContent?key={config.GEMINI_API_KEY}"
    body: dict = {"contents": [{"role": "user", "parts": parts}]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    r = post_json(url, body, timeout=180)
    data = r.json()
    try:
        return "".join(
            p.get("text", "") for p in data["candidates"][0]["content"]["parts"]
        ).strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini trả về bất thường: {str(data)[:300]}")


def _ollama_chat(model: str, prompt: str, system: str | None, images=None) -> str:
    import requests

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    msg: dict = {"role": "user", "content": prompt}
    if images:
        msg["images"] = images
    messages.append(msg)
    r = requests.post(
        f"{config.OLLAMA_BASE}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["message"]["content"].strip()


def _openai_chat(model: str, prompt: str, system: str | None, image_b64=None) -> str:
    import requests

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if image_b64:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})
    r = requests.post(
        f"{config.OPENAI_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
        json={"model": model, "messages": messages},
        timeout=180,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"].strip()
