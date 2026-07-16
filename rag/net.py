"""HTTP POST cho Gemini, retry TUỲ CHỌN qua RAG_MAX_RETRIES (mặc định 0 = không retry).

Chỉ retry lỗi TẠM: 429 (throttle) / 500 / 503 (lỗi phía Google). Lỗi khác (401 key sai,
404 model khoá...) fail ngay — retry vô ích. Backoff: theo retryDelay Google trả về
nếu có, không thì 2s, 4s...; trần 30s để đường serving không treo user quá lâu.
"""
import re
import time

import config

_RETRYABLE = {429, 500, 503}
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


def post_json(url: str, body: dict, timeout: int | None = None, tag: str = "Gemini"):
    """Trả về requests.Response chắc chắn status 200, hoặc raise RuntimeError.
    timeout=None -> dùng RAG_TIMEOUT (mặc định 30s); quá hạn -> lỗi rõ, không treo user."""
    import requests

    timeout = timeout or config.TIMEOUT
    last = None
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            r = requests.post(url, json=body, timeout=timeout)
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"{tag}: quá thời gian chờ ({timeout}s) — mạng hoặc dịch vụ không "
                "phản hồi (chỉnh bằng RAG_TIMEOUT)"
            )
        if r.status_code == 200:
            return r
        last = r
        if r.status_code not in _RETRYABLE or attempt == config.MAX_RETRIES:
            break
        m = _RETRY_DELAY_RE.search(r.text)
        parsed = float(m.group(1)) if m else 0.0
        delay = min(max(parsed, 2.0 * (attempt + 1)), 30.0)
        print(f"[{tag}] HTTP {r.status_code} — retry {attempt + 1}/{config.MAX_RETRIES} "
              f"sau {delay:.0f}s...")
        time.sleep(delay)

    hint = ""
    if last.status_code == 429:
        hint = (" | 429 = hết quota model này — đổi model (python list_models.py), "
                "chờ reset quota, hoặc prod: set RAG_MAX_RETRIES=2.")
    raise RuntimeError(f"{tag} HTTP {last.status_code}: {last.text[:300]}{hint}")
