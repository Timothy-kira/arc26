"""Async OpenAI-compatible chat client for a locally served model (vLLM).

All games share one client and one concurrency semaphore so vLLM's continuous
batching stays saturated without unbounded queueing.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import httpx


@dataclass
class LLMConfig:
    base_url: str = field(default_factory=lambda: os.getenv("ARC26_LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    model: str = field(default_factory=lambda: os.getenv("ARC26_LLM_MODEL", "qwen3.8-27b"))
    api_key: str = field(default_factory=lambda: os.getenv("ARC26_LLM_API_KEY", "EMPTY"))
    max_concurrency: int = field(default_factory=lambda: int(os.getenv("ARC26_LLM_CONCURRENCY", "16")))
    timeout_s: float = 300.0
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 2048
    thinking_max_tokens: int = 8192


@dataclass
class LLMStats:
    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


class ChatModel(Protocol):
    stats: LLMStats

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: bool = False,
        json_schema: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str: ...


def image_part(png_b64: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}}


def user_message(text: str, images: Optional[list[str]] = None) -> dict[str, Any]:
    if not images:
        return {"role": "user", "content": text}
    return {"role": "user", "content": [*(image_part(b) for b in images), {"type": "text", "text": text}]}


class OpenAICompatClient:
    def __init__(self, cfg: Optional[LLMConfig] = None) -> None:
        self.cfg = cfg or LLMConfig()
        self.stats = LLMStats()
        self._sem = asyncio.Semaphore(self.cfg.max_concurrency)
        self._client: Optional[httpx.AsyncClient] = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.cfg.base_url,
                timeout=httpx.Timeout(self.cfg.timeout_s),
                headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                trust_env=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def healthy(self) -> bool:
        try:
            r = await self._http().get("/models", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: bool = False,
        json_schema: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": max_tokens
            or (self.cfg.thinking_max_tokens if thinking else self.cfg.max_tokens),
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if json_schema is not None and not thinking:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": json_schema},
            }
        async with self._sem:
            t0 = time.monotonic()
            try:
                r = await self._http().post("/chat/completions", json=body)
                r.raise_for_status()
            except Exception:
                self.stats.failures += 1
                raise
            finally:
                self.stats.seconds += time.monotonic() - t0
        data = r.json()
        self.stats.calls += 1
        usage = data.get("usage") or {}
        self.stats.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.stats.completion_tokens += int(usage.get("completion_tokens") or 0)
        msg = data["choices"][0]["message"]
        return msg.get("content") or ""
