"""Bench targets — what we send load at.

Each target has the same async ``call(payload)`` interface so the engine
can hit them polymorphically. The real targets actually exercise the
provider HTTP API; the ``SyntheticTarget`` returns immediately after a
controllable sleep, which lets the engine be smoke-tested without
spending money on real LLM calls.
"""
from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod

import httpx

from .schemas import TargetName, WorkloadPayload


class TargetError(Exception):
    """Any failure of a target call — network, timeout, 4xx, 5xx."""


class Target(ABC):
    name: TargetName

    @abstractmethod
    async def call(self, payload: WorkloadPayload) -> None:
        """Issue one request. Raises ``TargetError`` on failure.

        The return value is intentionally None — we only care about
        latency and success/failure for benchmarking, not the content.
        """
        ...


# --------------------------------------------------------------------------- #
# Synthetic — no network, used to validate the engine itself                  #
# --------------------------------------------------------------------------- #

class SyntheticTarget(Target):
    """Returns after a configurable sleep. Useful for self-tests + CI."""

    name = TargetName.SYNTHETIC

    def __init__(
        self,
        *,
        mean_ms: float = 250.0,
        jitter_ms: float = 80.0,
        error_rate: float = 0.005,
    ) -> None:
        self._mean_ms = mean_ms
        self._jitter_ms = jitter_ms
        self._error_rate = error_rate

    async def call(self, payload: WorkloadPayload) -> None:
        # Larger payloads cost more, like a real provider would
        size_factor = 1 + len(payload.messages) * 0.05
        # Log-normal-ish jitter to mimic real tail behavior
        delay_ms = (
            self._mean_ms * size_factor
            + random.gauss(0, self._jitter_ms)
            + (random.expovariate(0.01) if random.random() < 0.05 else 0)
        )
        await asyncio.sleep(max(1.0, delay_ms) / 1000)

        if random.random() < self._error_rate:
            raise TargetError("synthetic injected error")


# --------------------------------------------------------------------------- #
# Real provider targets — all HTTP POST to the chat-completion endpoint       #
# --------------------------------------------------------------------------- #

class OpenAITarget(Target):
    name = TargetName.OPENAI

    def __init__(self, api_key: str, http: httpx.AsyncClient, *, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key
        self._http = http
        self._model = model

    async def call(self, payload: WorkloadPayload) -> None:
        try:
            resp = await self._http.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": payload.messages,
                    "max_tokens": payload.max_tokens,
                    "temperature": payload.temperature,
                    "stream": payload.stream,
                },
            )
            resp.raise_for_status()
        except Exception as exc:
            raise TargetError(f"openai: {exc}") from exc


class AnthropicTarget(Target):
    name = TargetName.ANTHROPIC

    def __init__(self, api_key: str, http: httpx.AsyncClient, *, model: str = "claude-3-5-haiku-20241022") -> None:
        self._api_key = api_key
        self._http = http
        self._model = model

    async def call(self, payload: WorkloadPayload) -> None:
        # Anthropic separates system from user/assistant turns.
        system = "\n".join(m["content"] for m in payload.messages if m["role"] == "system") or None
        turns = [m for m in payload.messages if m["role"] != "system"]
        body: dict = {
            "model": self._model,
            "messages": turns,
            "max_tokens": payload.max_tokens,
            "temperature": payload.temperature,
            "stream": payload.stream,
        }
        if system:
            body["system"] = system

        try:
            resp = await self._http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                json=body,
            )
            resp.raise_for_status()
        except Exception as exc:
            raise TargetError(f"anthropic: {exc}") from exc


class MistralTarget(Target):
    name = TargetName.MISTRAL

    def __init__(self, api_key: str, http: httpx.AsyncClient, *, model: str = "mistral-small-latest") -> None:
        self._api_key = api_key
        self._http = http
        self._model = model

    async def call(self, payload: WorkloadPayload) -> None:
        try:
            resp = await self._http.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": payload.messages,
                    "max_tokens": payload.max_tokens,
                    "temperature": payload.temperature,
                    "stream": payload.stream,
                },
            )
            resp.raise_for_status()
        except Exception as exc:
            raise TargetError(f"mistral: {exc}") from exc
