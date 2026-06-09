"""Fixed-width streaming histogram for latency in milliseconds.

Why a histogram and not a list of raw samples?
    A 5-minute run at 200 req/s is 60 000 samples. Storing each as a
    float8 is fine in isolation, but we want *live* percentiles every
    500ms — sorting 60k floats on every snapshot is wasteful, and at
    higher concurrency the sample array grows fast.

    A bucket-based histogram makes ``record`` O(1) and percentile
    computation O(B) where B is the number of buckets (tunable; default
    ~6000). It also bounds memory regardless of run duration.

Bucket layout: linear, ``bucket_ms`` wide (default 5ms). The array
auto-grows in fixed chunks when a sample lands beyond the current max
so we don't have to size for the worst case up front.

Accuracy: error is bounded by half a bucket width. At 5ms buckets the
worst-case percentile error is ~2.5ms — well below what matters for
LLM latency observability.
"""
from __future__ import annotations

import array
import math


class LatencyHistogram:
    __slots__ = (
        "_bucket_ms", "_counts", "_grow_chunk",
        "_total", "_sum_ms", "_min_ms", "_max_ms",
    )

    def __init__(
        self,
        *,
        bucket_ms: int = 5,
        initial_max_ms: int = 30_000,
        grow_chunk: int = 500,
    ) -> None:
        if bucket_ms < 1:
            raise ValueError("bucket_ms must be >= 1")
        self._bucket_ms = bucket_ms
        self._grow_chunk = grow_chunk
        # 'Q' = unsigned 64-bit; counts can grow large under load.
        n_buckets = max(1, initial_max_ms // bucket_ms)
        self._counts = array.array("Q", [0] * n_buckets)
        self._total = 0
        self._sum_ms = 0.0
        self._min_ms = math.inf
        self._max_ms = 0.0

    # ---- recording ------------------------------------------------------- #

    def record(self, latency_ms: float) -> None:
        if latency_ms < 0:
            return
        idx = int(latency_ms / self._bucket_ms)
        if idx >= len(self._counts):
            # Extend in chunks; reallocating on every overflow would be O(n²).
            extra = idx - len(self._counts) + self._grow_chunk
            self._counts.extend([0] * extra)
        self._counts[idx] += 1
        self._total += 1
        self._sum_ms += latency_ms
        if latency_ms < self._min_ms:
            self._min_ms = latency_ms
        if latency_ms > self._max_ms:
            self._max_ms = latency_ms

    # ---- queries --------------------------------------------------------- #

    def percentile(self, p: float) -> float:
        """Return the p-th percentile in ms. ``p`` is a fraction in [0, 1]."""
        if self._total == 0:
            return 0.0
        if p <= 0:
            return self._min_ms if self._min_ms != math.inf else 0.0
        if p >= 1:
            return self._max_ms

        target = self._total * p
        cumulative = 0
        for i, c in enumerate(self._counts):
            if c == 0:
                continue
            cumulative += c
            if cumulative >= target:
                # Bucket midpoint; error bounded by half a bucket width.
                return (i + 0.5) * self._bucket_ms
        return self._max_ms  # shouldn't reach here but safe-guard

    def snapshot(self) -> dict[str, float | int]:
        """Cheap snapshot of common stats. Computed lazily so callers
        can call this every ~500ms without sweating.
        """
        if self._total == 0:
            return {
                "count": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0,
                "mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0,
            }
        return {
            "count": self._total,
            "p50": self.percentile(0.50),
            "p90": self.percentile(0.90),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
            "mean_ms": self._sum_ms / self._total,
            "min_ms": self._min_ms,
            "max_ms": self._max_ms,
        }

    @property
    def total(self) -> int:
        return self._total
