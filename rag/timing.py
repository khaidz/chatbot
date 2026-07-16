"""Đo latency từng khâu của MỘT lượt hỏi (retrieve/rerank/llm/total).

Dùng dict module-level: an toàn vì server serialize mọi lượt qua 1 lock,
CLI thì tuần tự sẵn. log_query() gọi take() để gom số liệu vào query log.
"""
import time

_t: dict[str, int] = {}


def record(stage: str, ms: float):
    _t[stage] = int(ms)


def take() -> dict[str, int]:
    d = dict(_t)
    _t.clear()
    return d


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
