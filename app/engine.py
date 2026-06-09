"""Benchmark engine.

The core loop is a producer-consumer pattern over an ``asyncio.Semaphore``:

* The producer loops until the deadline, acquiring the semaphore before
  each task spawn — so we *never* exceed ``concurrency`` inflight calls.
* Each consumer task issues one request, records the latency, and
  releases the semaphore.

Why not ``asyncio.gather`` of N tasks?
    Gather fires all N at once which a) spikes the connection pool and
    b) doesn't give us a steady inflight cap if calls vary in duration.
    A semaphore + continuous producer keeps load *steady* at the target
    concurrency.

Spike scheduling is a tiny separate coroutine that flips the
semaphore's permit count partway through the run.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from uuid import uuid4

from .detector import BottleneckDetector
from .histogram import LatencyHistogram
from .schemas import (
    PercentileSnapshot,
    RunConfig,
    RunResult,
    RunSnapshot,
    RunState,
    TargetName,
    Workload,
    WorkloadPayload,
)
from .targets import Target, TargetError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Workload generation                                                         #
# --------------------------------------------------------------------------- #

_CHAT_PROMPTS = [
    "Summarize the refund policy in two sentences.",
    "I'd like to cancel my subscription — what are my options?",
    "Translate this into French: 'we'll follow up tomorrow.'",
    "Generate a polite apology for a late delivery.",
]

_AGENT_SYSTEM = (
    "You are a voice agent. Follow the tool-calling protocol strictly. "
    "Confirm intent, then act, then summarize."
)


def _build_payload(workload: Workload) -> WorkloadPayload:
    """Translate a workload type into an actual request payload.

    Different workload types stress the providers differently:
        chat       — single short turn (TTFB-dominated)
        agent      — longer system prompt + multi-turn (system overhead)
        streaming  — same as chat, but stream=True (SSE tail behavior)
        mixed      — random pick (more representative of real traffic)
    """
    if workload is Workload.MIXED:
        workload = random.choice([Workload.CHAT, Workload.AGENT, Workload.STREAMING])

    user_msg = random.choice(_CHAT_PROMPTS)

    if workload is Workload.CHAT:
        return WorkloadPayload(
            kind=Workload.CHAT,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=128,
        )
    if workload is Workload.AGENT:
        return WorkloadPayload(
            kind=Workload.AGENT,
            messages=[
                {"role": "system", "content": _AGENT_SYSTEM},
                {"role": "user", "content": "Help me reschedule my Friday booking."},
                {"role": "assistant", "content": "Sure — what time works?"},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=256,
        )
    if workload is Workload.STREAMING:
        return WorkloadPayload(
            kind=Workload.STREAMING,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=128,
            stream=True,
        )
    raise ValueError(f"unsupported workload: {workload}")


# --------------------------------------------------------------------------- #
# Engine                                                                      #
# --------------------------------------------------------------------------- #

class BenchmarkRun:
    """Lifecycle wrapper around a single benchmark execution.

    State transitions: pending → running → (completed | cancelled | failed).
    The run holds its own histograms; the snapshot method copies them out
    cheaply for the WebSocket broadcaster.
    """

    def __init__(self, config: RunConfig, targets: list[Target]) -> None:
        self.run_id = str(uuid4())
        self.config = config
        self._targets = targets
        self.started_at = datetime.now(timezone.utc)
        self.finished_at: datetime | None = None
        self.state = RunState.PENDING

        self._histograms: dict[TargetName, LatencyHistogram] = {
            t.name: LatencyHistogram() for t in targets
        }
        self._error_counts: dict[TargetName, int] = {t.name: 0 for t in targets}

        self._sem = asyncio.Semaphore(config.concurrency)
        self._stop = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()
        self._main_task: asyncio.Task | None = None

        self._detector = BottleneckDetector()
        self.alerts: list[str] = []

    # ---- lifecycle ------------------------------------------------------- #

    def start(self) -> None:
        if self.state is not RunState.PENDING:
            raise RuntimeError("run already started")
        self.state = RunState.RUNNING
        self._main_task = asyncio.create_task(self._run(), name=f"run-{self.run_id}")

    async def cancel(self) -> None:
        self._stop.set()
        if self._main_task:
            try:
                await asyncio.wait_for(self._main_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._main_task.cancel()
        self.state = RunState.CANCELLED
        self.finished_at = datetime.now(timezone.utc)

    # ---- inner loop ------------------------------------------------------ #

    async def _run(self) -> None:
        cfg = self.config
        deadline = time.monotonic() + cfg.duration_s
        spike_task: asyncio.Task | None = None
        if cfg.spike_at_s is not None and cfg.spike_concurrency:
            spike_task = asyncio.create_task(self._inject_spike())

        try:
            i = 0
            while time.monotonic() < deadline and not self._stop.is_set():
                # acquire blocks until a slot is free → enforces inflight cap
                await self._sem.acquire()
                target = self._targets[i % len(self._targets)]
                i += 1
                task = asyncio.create_task(self._send_one(target))
                self._tasks.add(task)
                # discard completed tasks so the set doesn't grow forever
                task.add_done_callback(self._tasks.discard)

            # drain remaining inflight requests
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self.state = RunState.COMPLETED
        except Exception:
            logger.exception("run %s failed", self.run_id)
            self.state = RunState.FAILED
        finally:
            if spike_task and not spike_task.done():
                spike_task.cancel()
            self.finished_at = datetime.now(timezone.utc)

    async def _send_one(self, target: Target) -> None:
        payload = _build_payload(self.config.workload)
        start = time.perf_counter()
        try:
            await target.call(payload)
            latency_ms = (time.perf_counter() - start) * 1000
            self._histograms[target.name].record(latency_ms)
        except TargetError:
            self._error_counts[target.name] += 1
        except Exception:
            # unexpected — log and count, don't crash the engine
            logger.exception("unexpected error from target %s", target.name.value)
            self._error_counts[target.name] += 1
        finally:
            self._sem.release()

    async def _inject_spike(self) -> None:
        cfg = self.config
        assert cfg.spike_at_s is not None and cfg.spike_concurrency is not None
        await asyncio.sleep(cfg.spike_at_s)
        if self._stop.is_set():
            return
        # Boost concurrency by topping up the semaphore.
        boost = cfg.spike_concurrency - cfg.concurrency
        if boost > 0:
            for _ in range(boost):
                self._sem.release()
            logger.info("run %s: spike injected +%d slots", self.run_id, boost)

    # ---- snapshots ------------------------------------------------------- #

    def snapshot(self) -> RunSnapshot:
        """Cheap to call repeatedly — the WS broadcaster does so every 500ms."""
        elapsed = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        per_target: list[PercentileSnapshot] = []
        latest_alert: str | None = None

        for tname, hist in self._histograms.items():
            s = hist.snapshot()
            snap = PercentileSnapshot(
                target=tname,
                count=int(s["count"]),
                error_count=self._error_counts[tname],
                p50_ms=s["p50"],
                p90_ms=s["p90"],
                p95_ms=s["p95"],
                p99_ms=s["p99"],
                mean_ms=s["mean_ms"],
                min_ms=s["min_ms"],
                max_ms=s["max_ms"],
            )
            per_target.append(snap)
            alert = self._detector.observe(snap)
            if alert:
                latest_alert = alert
                if alert not in self.alerts:
                    self.alerts.append(alert)

        return RunSnapshot(
            run_id=self.run_id,
            state=self.state,
            elapsed_s=elapsed,
            targets=per_target,
            bottleneck=latest_alert,
        )

    def result(self) -> RunResult:
        """Final result — call once state is terminal."""
        snap = self.snapshot()
        total = sum(t.count for t in snap.targets)
        errors = sum(t.error_count for t in snap.targets)

        # Winner = lowest P95 among targets with enough samples to count.
        eligible = [t for t in snap.targets if t.count >= 30]
        winner = min(eligible, key=lambda t: t.p95_ms).target if eligible else None

        return RunResult(
            run_id=self.run_id,
            state=self.state,
            config=self.config,
            started_at=self.started_at,
            finished_at=self.finished_at,
            total_requests=total,
            total_errors=errors,
            targets=snap.targets,
            winner=winner,
            alerts=self.alerts,
        )
