"""FastAPI application — REST routes for run lifecycle + a WebSocket for
live percentile telemetry.

Run lifecycle::

    POST   /v1/runs               start a run, returns run_id
    GET    /v1/runs/{id}          fetch result (final once state is terminal)
    DELETE /v1/runs/{id}          cancel a running benchmark
    WS     /v1/runs/{id}/stream   subscribe to live snapshots (~500ms cadence)
    GET    /healthz               liveness for K8s

Runs are kept in an in-memory dict here for the demo; production would
swap this for Redis or a real key-value store with TTL.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from .config import get_settings
from .engine import BenchmarkRun
from .schemas import RunConfig, RunCreated, RunResult, RunSnapshot, RunState, TargetName
from .targets import (
    AnthropicTarget,
    MistralTarget,
    OpenAITarget,
    SyntheticTarget,
    Target,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Target wiring                                                               #
# --------------------------------------------------------------------------- #

def _resolve_targets(names: list[TargetName], http: httpx.AsyncClient) -> list[Target]:
    """Build target instances for the requested names.

    Real-provider targets need credentials; if a key is missing for one,
    we raise rather than silently substituting — the user asked for it,
    they should know it's unavailable.
    """
    s = get_settings()
    out: list[Target] = []
    for name in names:
        if name is TargetName.SYNTHETIC:
            out.append(SyntheticTarget())
        elif name is TargetName.OPENAI:
            if not s.openai_api_key:
                raise HTTPException(400, "OPENAI_API_KEY not configured")
            out.append(OpenAITarget(s.openai_api_key, http))
        elif name is TargetName.ANTHROPIC:
            if not s.anthropic_api_key:
                raise HTTPException(400, "ANTHROPIC_API_KEY not configured")
            out.append(AnthropicTarget(s.anthropic_api_key, http))
        elif name is TargetName.MISTRAL:
            if not s.mistral_api_key:
                raise HTTPException(400, "MISTRAL_API_KEY not configured")
            out.append(MistralTarget(s.mistral_api_key, http))
    return out


# --------------------------------------------------------------------------- #
# Lifespan                                                                    #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logging.basicConfig(
        level=s.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # One client shared across all targets and all runs — connection
    # pooling matters when running thousands of requests/second.
    http = httpx.AsyncClient(
        timeout=httpx.Timeout(s.request_timeout_s, connect=s.connect_timeout_s),
        limits=httpx.Limits(max_keepalive_connections=128, max_connections=512),
    )
    app.state.http = http
    # In-memory run registry. Production: Redis with TTL.
    app.state.runs = {}  # type: dict[str, BenchmarkRun]

    try:
        yield
    finally:
        # Cancel any still-running benchmarks on shutdown.
        for run in app.state.runs.values():
            if run.state is RunState.RUNNING:
                await run.cancel()
        await http.aclose()


app = FastAPI(
    title="ConvoStream-Bench",
    version="0.1.0",
    description=(
        "Concurrent LLM benchmarking with streaming-histogram percentiles, "
        "live WebSocket telemetry, and bottleneck detection."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #

@app.get("/healthz", tags=["ops"])
async def healthz(request: Request) -> dict[str, Any]:
    runs: dict[str, BenchmarkRun] = request.app.state.runs
    return {
        "ok": True,
        "active_runs": sum(1 for r in runs.values() if r.state is RunState.RUNNING),
        "total_runs": len(runs),
    }


@app.post("/v1/runs", response_model=RunCreated, tags=["runs"])
async def create_run(config: RunConfig, request: Request) -> RunCreated:
    """Start a benchmark and return immediately with a ``run_id``.

    The run executes in the background. Subscribe to ``/v1/runs/{id}/stream``
    for live snapshots or poll ``GET /v1/runs/{id}`` for the final result.
    """
    s = get_settings()
    if config.concurrency > s.max_concurrency:
        raise HTTPException(400, f"concurrency exceeds max ({s.max_concurrency})")
    if config.duration_s > s.max_duration_s:
        raise HTTPException(400, f"duration exceeds max ({s.max_duration_s}s)")

    http: httpx.AsyncClient = request.app.state.http
    targets = _resolve_targets(config.targets, http)
    if not targets:
        raise HTTPException(400, "no targets resolved")

    run = BenchmarkRun(config, targets)
    request.app.state.runs[run.run_id] = run
    run.start()

    logger.info(
        "run_started id=%s targets=%s concurrency=%d duration=%ds",
        run.run_id,
        [t.value for t in config.targets],
        config.concurrency,
        config.duration_s,
    )
    return RunCreated(
        run_id=run.run_id,
        state=run.state,
        config=config,
        started_at=run.started_at,
    )


@app.get("/v1/runs/{run_id}", response_model=RunResult, tags=["runs"])
async def get_run(run_id: str, request: Request) -> RunResult:
    run = _get_run_or_404(request, run_id)
    return run.result()


@app.delete("/v1/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["runs"])
async def cancel_run(run_id: str, request: Request):
    run = _get_run_or_404(request, run_id)
    if run.state is not RunState.RUNNING:
        return
    await run.cancel()


@app.websocket("/v1/runs/{run_id}/stream")
async def stream_run(websocket: WebSocket, run_id: str):
    """Push :class:`RunSnapshot` every ~500ms until the run terminates.

    The cadence is fixed by config; a subscriber that can't keep up will
    be disconnected by the WS layer on backpressure.
    """
    await websocket.accept()
    runs: dict[str, BenchmarkRun] = websocket.app.state.runs
    run = runs.get(run_id)
    if run is None:
        await websocket.close(code=1008, reason="run not found")
        return

    interval = get_settings().snapshot_interval_ms / 1000
    try:
        # Stream until the run is in a terminal state, then send one
        # final snapshot so the subscriber sees the closing numbers.
        while run.state is RunState.RUNNING:
            snap: RunSnapshot = run.snapshot()
            await websocket.send_json(snap.model_dump(mode="json"))
            await asyncio.sleep(interval)
        await websocket.send_json(run.snapshot().model_dump(mode="json"))
    except WebSocketDisconnect:
        logger.info("ws_client_disconnected run=%s", run_id)
    finally:
        # The WS context manager closes the socket; nothing to do here.
        pass


# --------------------------------------------------------------------------- #

def _get_run_or_404(request: Request, run_id: str) -> BenchmarkRun:
    run = request.app.state.runs.get(run_id)
    if run is None:
        raise HTTPException(404, f"run {run_id} not found")
    return run
