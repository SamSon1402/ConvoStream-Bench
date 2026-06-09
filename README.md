# ConvoStream-Bench

Concurrent LLM benchmarking with streaming-histogram percentiles, live WebSocket telemetry, and bottleneck detection. The observability tool that tells you *where* your voice agent gets slow before your users do.

Companion to the live demo at `samson1402.github.io/convostream-bench`.

## What's in the box

```
app/
├── main.py        FastAPI app + REST + WebSocket endpoint
├── config.py      pydantic-settings; env-driven config
├── schemas.py     Pydantic models for run config, snapshots, results
├── histogram.py   Streaming linear-bucket histogram + percentile math
├── targets.py    OpenAI / Anthropic / Mistral / synthetic adapters
├── detector.py    Bottleneck detector (smoothed P99 + error burst)
└── engine.py      Producer-consumer benchmark engine
```

## Endpoints

| Method | Path                            | Purpose                                |
|--------|---------------------------------|----------------------------------------|
| POST   | `/v1/runs`                      | Start a benchmark; returns `run_id`    |
| GET    | `/v1/runs/{id}`                 | Final result (or partial, while live)  |
| DELETE | `/v1/runs/{id}`                 | Cancel a running benchmark             |
| WS     | `/v1/runs/{id}/stream`          | Subscribe to live snapshots (~500ms)   |
| GET    | `/healthz`                      | Liveness for K8s                       |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # provider keys are OPTIONAL — synthetic target works without them
uvicorn app.main:app --reload
```

Start a 30-second run against the synthetic target (no API costs):

```bash
curl -sX POST http://localhost:8000/v1/runs -H 'content-type: application/json' -d '{
  "targets": ["synthetic"],
  "workload": "chat",
  "concurrency": 32,
  "duration_s": 30
}' | jq

# subscribe to live snapshots (needs `websocat` or similar)
websocat ws://localhost:8000/v1/runs/<run_id>/stream
```

## Why a histogram, not a list of samples

A 5-minute run at 200 req/s is 60 000 samples. We want **live** percentiles every 500ms — sorting 60k floats every snapshot is wasteful, and memory grows linearly with run duration.

`LatencyHistogram` is a linear-bucket histogram: `record()` is O(1), `percentile()` is O(B) where B is the number of buckets (default ~6000 → ~48 KB per target, fixed). The bucket width (5ms by default) bounds the percentile error at ~2.5ms.

It's not HDR — log-linear bucketing would be more memory-efficient at the extreme tails. Linear was the right trade-off for the 0–30 s range we actually care about for LLM latency.

## Why a semaphore, not `asyncio.gather`

The naive approach — `await asyncio.gather(*[send() for _ in range(N)])` — fires N concurrent tasks once and then waits. Two problems:

1. All N requests open connections simultaneously, spiking the pool.
2. As fast requests finish and slow ones drag, *inflight* concurrency drifts below the target.

The engine instead acquires a semaphore *before each spawn*, then releases on completion. Result: inflight stays pinned at `concurrency` for the whole run, which is what you actually want when measuring "what does the system do at 64 concurrent calls?"

## Bottleneck detection

Two heuristics, run on every snapshot:

* **P99 spike** — current P99 exceeds an EWMA baseline by ≥ 1.6×. The baseline updates slowly during a spike (α = 0.05 instead of 0.2) so a sustained spike doesn't get accepted as "new normal".
* **Error burst** — per-target error rate > 5% in the most recent window.

Neither fires before 50 recorded samples per target — early-run percentiles are too noisy to alert on.

## The four workload types

| Workload    | What it stresses                         |
|-------------|-------------------------------------------|
| `chat`      | TTFB-dominated single-turn               |
| `agent`     | Longer system prompt + multi-turn        |
| `streaming` | `stream=true`; SSE tail behavior         |
| `mixed`     | Random pick — most representative        |

## Synthetic target

The `synthetic` target returns after a controllable sleep with log-normal-ish jitter and an injectable error rate. It exists so the engine itself can be exercised in CI and during local dev without spending real money on every load test.

## Design notes worth pointing at

- **One shared `httpx.AsyncClient`** across targets and runs. HTTP/2 + keepalive pooling only kicks in if the client is shared.
- **Run lifecycle is explicit** — `pending → running → (completed | cancelled | failed)`. State transitions live on the run object, not in scattered booleans.
- **Tasks tracked in a set, discarded on completion** — avoids unbounded growth during long runs while still letting the producer cancel them on shutdown.
- **Snapshot is cheap** — copying out four percentiles per target costs ~O(B) and the WS broadcaster calls it every 500ms without breaking a sweat.
- **Spike injection releases extra semaphore permits** — the cleanest way to bump concurrency mid-run without touching the producer loop.

## Deliberately out of scope

- Multi-host load generation (one process; for true scale, run N instances behind a coordinator)
- Persistent run history (in-memory dict here; would be Redis/Postgres with TTL in prod)
- Per-token streaming latency breakdown (TTFB + inter-token gaps) — possible but adds provider-specific SSE parsers
- Authentication and per-tenant quotas
