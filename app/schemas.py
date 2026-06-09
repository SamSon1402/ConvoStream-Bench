"""Pydantic models — request bodies, snapshots, and the run lifecycle."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Workload(str, Enum):
    CHAT = "chat"
    AGENT = "agent"
    STREAMING = "streaming"
    MIXED = "mixed"


class TargetName(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    SYNTHETIC = "synthetic"


class RunState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RunConfig(BaseModel):
    """Single benchmark run parameters.

    ``concurrency`` is the *max inflight* — the engine paces with a
    semaphore so we never exceed it. ``duration_s`` is wall-clock; the
    producer stops accepting new work after that, then drains.
    """

    targets: list[TargetName] = Field(default_factory=lambda: [TargetName.SYNTHETIC])
    workload: Workload = Workload.CHAT
    concurrency: int = Field(default=32, ge=1, le=1024)
    duration_s: int = Field(default=60, ge=1, le=3600)
    # Optional: schedule a spike at this offset (seconds from start)
    spike_at_s: int | None = Field(default=None, ge=0)
    spike_concurrency: int | None = Field(default=None, ge=1, le=2048)


class PercentileSnapshot(BaseModel):
    """One provider's stats at a point in time. Computed from the histogram."""

    target: TargetName
    count: int
    error_count: int
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


class RunSnapshot(BaseModel):
    """What the WebSocket emits every ~500ms while a run is live."""

    run_id: str
    state: RunState
    elapsed_s: float
    targets: list[PercentileSnapshot]
    bottleneck: str | None = None  # human-readable alert, if any


class RunCreated(BaseModel):
    run_id: str
    state: RunState
    config: RunConfig
    started_at: datetime


class RunResult(BaseModel):
    """Final result returned by GET /v1/runs/{id} once complete."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    state: RunState
    config: RunConfig
    started_at: datetime
    finished_at: datetime | None
    total_requests: int
    total_errors: int
    targets: list[PercentileSnapshot]
    winner: TargetName | None  # best p95 among targets
    alerts: list[str] = Field(default_factory=list)


# Internal — workload payloads. Kept here so the engine and targets
# agree on the shape of what gets sent.
class WorkloadPayload(BaseModel):
    kind: Workload
    messages: list[dict]
    max_tokens: int = 128
    temperature: float = 0.7
    stream: bool = False
