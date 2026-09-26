"""IO schemas for the LLM roles. Lenient on input so a 27B model's near-misses still parse."""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

_ACT = re.compile(r"(\d)")


def _coerce_action(v: Any) -> int:
    if isinstance(v, str):
        s = v.strip().upper()
        if s in ("UP",):
            return 1
        if s in ("DOWN",):
            return 2
        if s in ("LEFT",):
            return 3
        if s in ("RIGHT",):
            return 4
        if s.startswith("CLICK"):
            return 6
        m = _ACT.search(s)
        if not m:
            raise ValueError(f"not an action: {v!r}")
        v = int(m.group(1))
    v = int(v)
    if not 1 <= v <= 7:
        raise ValueError("action must be 1..7")
    return v


class Hypothesis(BaseModel):
    statement: str
    kind: str = "mechanic"
    confidence: float = 0.5
    test: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5


class HypothesesOut(BaseModel):
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    notes: str = ""


class ActionSpec(BaseModel):
    action: int
    x: Optional[int] = None
    y: Optional[int] = None
    repeat: int = 1

    @field_validator("action", mode="before")
    @classmethod
    def _a(cls, v: Any) -> int:
        return _coerce_action(v)

    @field_validator("x", "y", mode="before")
    @classmethod
    def _coord(cls, v: Any) -> Optional[int]:
        if v is None or v == "":
            return None
        return max(0, min(63, int(v)))

    @field_validator("repeat", mode="before")
    @classmethod
    def _rep(cls, v: Any) -> int:
        try:
            return max(1, min(10, int(v)))
        except (TypeError, ValueError):
            return 1

    def key(self) -> tuple:
        if self.action == 6:
            return (6, self.x if self.x is not None else 32, self.y if self.y is not None else 32)
        return (self.action,)


def _priorities(v: Any) -> dict[int, float]:
    out: dict[int, float] = {}
    if isinstance(v, dict):
        for k, val in v.items():
            try:
                out[_coerce_action(k)] = max(0.0, min(2.0, float(val)))
            except (TypeError, ValueError):
                continue
    return out


class PlanOut(BaseModel):
    goal: str = ""
    confirmed: list[str] = Field(default_factory=list)
    refuted: list[str] = Field(default_factory=list)
    new_hypotheses: list[Hypothesis] = Field(default_factory=list)
    actions: list[ActionSpec] = Field(default_factory=list)
    priorities: dict[int, float] = Field(default_factory=dict)
    avoid: list[ActionSpec] = Field(default_factory=list)
    expect: str = ""
    mode: Literal["execute", "explore"] = "explore"

    @field_validator("priorities", mode="before")
    @classmethod
    def _p(cls, v: Any) -> dict[int, float]:
        return _priorities(v)

    @field_validator("mode", mode="before")
    @classmethod
    def _mode(cls, v: Any) -> str:
        return "execute" if str(v).lower().startswith("exec") else "explore"


class CharterOut(BaseModel):
    system_addendum: str = ""
    priorities: dict[int, float] = Field(default_factory=dict)
    avoid: list[ActionSpec] = Field(default_factory=list)
    max_level_actions: int = 0

    @field_validator("priorities", mode="before")
    @classmethod
    def _p(cls, v: Any) -> dict[int, float]:
        return _priorities(v)


class LevelSummaryOut(BaseModel):
    task_intent: str
    approach: str
    key_insight: str
