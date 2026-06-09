"""Application settings.

Most knobs (concurrency, duration, workload) are per-run and live in the
request schema; this file holds *gateway-wide* defaults and credentials.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "convostream-bench"
    log_level: str = "INFO"

    # Provider credentials. Optional — when none are set, only the
    # synthetic target is available (useful for engine smoke tests).
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    mistral_api_key: str | None = None

    # Engine
    max_concurrency: int = 1024
    max_duration_s: int = 3600

    # WebSocket broadcast cadence
    snapshot_interval_ms: int = 500

    # Bottleneck detector
    p99_spike_multiplier: float = 1.6     # current P99 vs rolling baseline
    p99_smoothing_alpha: float = 0.2      # EWMA factor for baseline
    spike_min_samples: int = 50           # don't alert before this many recordings

    # HTTP client (used by real targets)
    request_timeout_s: float = 30.0
    connect_timeout_s: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
