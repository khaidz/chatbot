"""Chat + Vision cho provider CLOUD: gemini | openai.

Lưu ý model Gemini: dùng alias `gemini-flash-latest`, KHÔNG dùng tên bản cứng
(gemini-2.5-flash có thể bị Google khoá với user mới -> 404).
"""
import base64

import config
from rag import usage


def chat(prompt: str, system: str | None = None, provider: str | None = None,
         model: str | None = None, temperature: float | None = None) -> str:
    """temperature=0 cho tác vụ cần DETERMINISTIC (condense làm key cache, rerank JSON);
    None = mặc định của provider (sinh câu trả lời)."""
    provider = provider or config.LLM_PROVIDER
    model = model or config.LLM_MODEL
    if provider == "gemini":
        return _gemini_generate(model, [{"text": prompt}], system, temperature)
    if provider == "openai":
        return _openai_chat(model, prompt, system, temperature=temperature)
    raise ValueError(f"LLM provider chưa hỗ trợ: {provider} (chỉ gemini | openai)")


def chat_stream(prompt: str, system: str | None = None, provider: str | None = None,
                model: str | None = None):
    """Generator sinh từng mẩu text (streaming). Lỗi giữa chừng -> raise,
    caller fallback chat() rồi extractive."""
    provider = provider or config.LLM_PROVIDER
    model = model or config.LLM_MODEL
    if provider == "gemini":
        yield from _gemini_stream(model, prompt, system)
    elif provider == "openai":
        yield from _openai_stream(model, prompt, system)
    else:
        raise ValueError(f"LLM provider chưa hỗ trợ: {provider} (chỉ gemini | openai)")


def _gemini_stream(model: str, prompt: str, system: str | None):
    import json

    import requests

    if not config.GEMINI_API_KEY:
        raise RuntimeError("Thiếu GEMINI_API_KEY")
    url = (f"{config.GEMINI_BASE}/models/{model}:streamGenerateContent"
           f"?alt=sse&key={config.GEMINI_API_KEY}")
    body: dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    # stream=True: timeout áp cho kết nối + KHOẢNG LẶNG giữa 2 chunk —
    # token đang chảy đều thì không bị cắt, chỉ cắt khi thật sự im lặng quá hạn.
    try:
        r = requests.post(url, json=body, stream=True, timeout=config.TIMEOUT)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Gemini stream: quá thời gian chờ ({config.TIMEOUT}s) — "
                           "mạng hoặc dịch vụ không phản hồi (chỉnh bằng RAG_TIMEOUT)")
    if r.status_code != 200:
        raise RuntimeError(f"Gemini stream HTTP {r.status_code}: {r.text[:300]}")
    # Gemini gắn usageMetadata LUỸ KẾ vào từng chunk -> giữ cái mới nhất, cộng MỘT lần
    # khi stream kết thúc (cộng mỗi chunk sẽ đếm trùng). finally: stream bị huỷ giữa
    # chừng (user đóng tab, watchdog cắt) vẫn ghi được phần token đã thực sự tiêu.
    last_um: dict = {}
    try:
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            obj = json.loads(data)
            if obj.get("usageMetadata"):
                last_um = obj["usageMetadata"]
            for part in obj.get("candidates", [{}])[0].get("content", {}).get("parts", []):
                if part.get("text"):
                    yield part["text"]
    finally:
        usage.add(last_um.get("promptTokenCount"), last_um.get("candidatesTokenCount"))


def _openai_stream(model: str, prompt: str, system: str | None):
    import json

    import requests

    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    try:
        r = requests.post(
            f"{config.OPENAI_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            # stream_options.include_usage: OpenAI CHỈ trả token khi stream nếu bật cờ này
            # (chunk cuối cùng mang "usage", không có "choices").
            json={"model": model, "messages": messages, "stream": True,
                  "stream_options": {"include_usage": True}},
            stream=True, timeout=config.TIMEOUT,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"OpenAI stream: quá thời gian chờ ({config.TIMEOUT}s) — "
                           "mạng hoặc dịch vụ không phản hồi (chỉnh bằng RAG_TIMEOUT)")
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI stream HTTP {r.status_code}: {r.text[:300]}")
    last_usage: dict = {}
    try:
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):        # chunk chốt: có "usage", "choices" rỗng
                last_usage = obj["usage"]
            choices = obj.get("choices") or [{}]
            piece = choices[0].get("delta", {}).get("content", "")
            if piece:
                yield piece
    finally:
        usage.add(last_usage.get("prompt_tokens"), last_usage.get("completion_tokens"))


def vision_ocr(png_bytes: bytes, prompt: str, provider: str | None = None,
               model: str | None = None) -> str:
    provider = provider or config.VISION_PROVIDER
    model = model or config.VISION_MODEL
    b64 = base64.b64encode(png_bytes).decode("ascii")
    if provider == "gemini":
        parts = [{"inline_data": {"mime_type": "image/png", "data": b64}}, {"text": prompt}]
        return _gemini_generate(model, parts, None, None)
    if provider == "openai":
        return _openai_chat(model, prompt, None, image_b64=b64)
    raise ValueError(f"Vision provider chưa hỗ trợ: {provider} (chỉ gemini | openai)")


# ---------- backends ----------
def _gemini_generate(model: str, parts: list[dict], system: str | None,
                     temperature: float | None = None) -> str:
    from rag.net import post_json

    if not config.GEMINI_API_KEY:
        raise RuntimeError("Thiếu GEMINI_API_KEY (setx GEMINI_API_KEY \"AIza...\" rồi mở cmd MỚI)")
    url = f"{config.GEMINI_BASE}/models/{model}:generateContent?key={config.GEMINI_API_KEY}"
    body: dict = {"contents": [{"role": "user", "parts": parts}]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if temperature is not None:
        body["generationConfig"] = {"temperature": temperature}
    r = post_json(url, body)  # timeout = RAG_TIMEOUT (mặc định 30s)
    data = r.json()
    um = data.get("usageMetadata") or {}
    usage.add(um.get("promptTokenCount"), um.get("candidatesTokenCount"))
    try:
        return "".join(
            p.get("text", "") for p in data["candidates"][0]["content"]["parts"]
        ).strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini trả về bất thường: {str(data)[:300]}")


def _openai_chat(model: str, prompt: str, system: str | None, image_b64=None,
                 temperature: float | None = None) -> str:
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
    body: dict = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature
    timeout = config.VISION_TIMEOUT if image_b64 else config.TIMEOUT
    try:
        r = requests.post(
            f"{config.OPENAI_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            json=body,
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"OpenAI: quá thời gian chờ ({timeout}s) — "
                           "mạng hoặc dịch vụ không phản hồi (chỉnh bằng RAG_TIMEOUT)")
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    u = data.get("usage") or {}
    usage.add(u.get("prompt_tokens"), u.get("completion_tokens"))
    return data["choices"][0]["message"]["content"].strip()
