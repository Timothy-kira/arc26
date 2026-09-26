"""Scripted chat model for CPU tests and for the LLM-less fallback path."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Optional

from .client import LLMStats

Handler = Callable[[list[dict[str, Any]], Optional[dict[str, Any]]], str]


class MockLLM:
    def __init__(self, handler: Optional[Handler] = None, delay_s: float = 0.0) -> None:
        self.handler = handler or (lambda messages, schema: "{}")
        self.delay_s = delay_s
        self.stats = LLMStats()
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: bool = False,
        json_schema: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        self.calls.append(messages)
        self.stats.calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        out = self.handler(messages, json_schema)
        return out if isinstance(out, str) else json.dumps(out)

    async def healthy(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


_DEFAULTS: dict[str, Any] = {
    "HypothesesOut": {"hypotheses": [{"statement": "the avatar moves with arrows", "kind": "control", "confidence": 0.5, "test": "press ACTION1"}], "notes": ""},
    "PlanOut": {"goal": "explore", "actions": [], "mode": "explore"},
    "CharterOut": {"system_addendum": "", "priorities": {}, "avoid": [], "max_level_actions": 0},
    "LevelSummaryOut": {"task_intent": "reach the goal", "approach": "explore every action systematically", "key_insight": "none"},
    "GateOut": {"plan": "", "selected": []},
    "SkillOpsOut": {"reasoning": "", "ops": []},
}


def schema_default_handler(messages: list[dict[str, Any]], schema: Optional[dict[str, Any]]) -> str:
    """Return a minimal valid reply for whichever role schema is requested."""
    title = (schema or {}).get("title", "")
    return json.dumps(_DEFAULTS.get(title, {}))
