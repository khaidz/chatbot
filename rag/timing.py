"""Đo latency từng khâu của MỘT lượt hỏi (retrieve/rerank/llm/total).

Dùng bộ nhớ THREAD-LOCAL: mỗi luồng (mỗi request web đồng thời, hoặc CLI) có bảng
số liệu riêng, không giẫm lên nhau. log_query() gọi take() để gom số liệu của ĐÚNG
lượt hỏi đang chạy trên luồng này vào query log.
"""
import threading
import time

_local = threading.local()


def _bucket() -> dict[str, int]:
    d = getattr(_local, "t", None)
    if d is None:
        d = _local.t = {}
    return d


def record(stage: str, ms: float):
    _bucket()[stage] = int(ms)


def take() -> dict[str, int]:
    d = _bucket()
    out = dict(d)
    d.clear()
    return out


class span:
    """with span("llm_ms"): ...  -> tự record thời gian chạy khối lệnh."""

    def __init__(self, stage: str):
        self.stage = stage

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        record(self.stage, (time.perf_counter() - self.t0) * 1000)
        return False
