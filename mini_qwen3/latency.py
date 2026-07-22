from functools import wraps
from time import perf_counter
from collections import defaultdict


class LatencyTracker:
    _timings: dict[str, list[float]] = defaultdict(list)

    @classmethod
    def record(cls, label: str, elapsed: float) -> None:
        cls._timings[label].append(elapsed)

    @classmethod
    def summary(cls) -> dict[str, dict[str, float]]:
        result = {}
        for label, times in cls._timings.items():
            result[label] = {
                "count": len(times),
                "total": sum(times),
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            }
        return result

    @classmethod
    def print_summary(cls) -> None:
        for label, stats in cls.summary().items():
            print(f"  {label}: {stats['mean']*1000:.3f}ms avg ({stats['count']} calls, "
                  f"{stats['total']*1000:.1f}ms total)")

    @classmethod
    def reset(cls) -> None:
        cls._timings.clear()


def latency(label: str | None = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = label or func.__qualname__
            start = perf_counter()
            result = func(*args, **kwargs)
            LatencyTracker.record(name, perf_counter() - start)
            return result
        return wrapper
    return decorator


class latency_context:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *args):
        LatencyTracker.record(self.label, perf_counter() - self.start)
