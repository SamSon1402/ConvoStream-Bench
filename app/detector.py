"""Bottleneck detector.

Two heuristics:

1. **P99 spike**: current P99 exceeds an EWMA baseline by a configurable
   multiplier. EWMA smooths the baseline so a single bad sample doesn't
   permanently anchor "normal" to a high value.

2. **Error burst**: per-target error rate exceeds 5% in the most recent
   window.

We don't alert before ``min_samples`` recordings on a given target —
early-run percentiles are too noisy to trust. This avoids the classic
"alert fires for 1.5s every time a benchmark starts" antipattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import PercentileSnapshot, TargetName


@dataclass
class TargetBaseline:
    p99_baseline_ms: float = 0.0
    samples_seen: int = 0


class BottleneckDetector:
    def __init__(
        self,
        *,
        spike_multiplier: float = 1.6,
        smoothing_alpha: float = 0.2,
        min_samples: int = 50,
        error_rate_threshold: float = 0.05,
    ) -> None:
        self._spike_multiplier = spike_multiplier
        self._alpha = smoothing_alpha
        self._min_samples = min_samples
        self._error_rate_threshold = error_rate_threshold
        self._baselines: dict[TargetName, TargetBaseline] = {}

    def observe(self, snap: PercentileSnapshot) -> str | None:
        """Update the baseline for this target and return an alert
        string if either heuristic fires, else None.
        """
        baseline = self._baselines.setdefault(snap.target, TargetBaseline())
        baseline.samples_seen = snap.count

        # Don't try to alert on too-fresh data.
        if snap.count < self._min_samples:
            self._update_baseline(baseline, snap.p99_ms)
            return None

        # Error burst first — usually more urgent than tail latency.
        if snap.count > 0:
            error_rate = snap.error_count / snap.count
            if error_rate > self._error_rate_threshold:
                return (
                    f"error burst on {snap.target.value}: "
                    f"{error_rate * 100:.1f}% errors in last window"
                )

        # P99 spike vs baseline.
        if baseline.p99_baseline_ms > 0:
            ratio = snap.p99_ms / baseline.p99_baseline_ms
            if ratio > self._spike_multiplier:
                msg = (
                    f"P99 spike on {snap.target.value}: "
                    f"{snap.p99_ms:.0f}ms vs baseline {baseline.p99_baseline_ms:.0f}ms "
                    f"({ratio:.2f}×)"
                )
                # Update baseline more slowly during a spike so we don't
                # accept the spike as new normal too quickly.
                self._update_baseline(baseline, snap.p99_ms, alpha_override=0.05)
                return msg

        self._update_baseline(baseline, snap.p99_ms)
        return None

    def _update_baseline(
        self,
        baseline: TargetBaseline,
        new_p99: float,
        *,
        alpha_override: float | None = None,
    ) -> None:
        a = alpha_override if alpha_override is not None else self._alpha
        if baseline.p99_baseline_ms == 0:
            baseline.p99_baseline_ms = new_p99
        else:
            baseline.p99_baseline_ms = (1 - a) * baseline.p99_baseline_ms + a * new_p99
