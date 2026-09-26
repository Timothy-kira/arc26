"""Per-node verdicts and host decisions (Raven ``dag_verdict.py`` / ``dag_adjudication.py``).

A judge rules on every finished node; a judge error or timeout counts as
accomplished (fail-open). A node ruled not accomplished is suspended and the host
decides: ``continue`` (re-run with a message), ``abandon`` (fail it, dependents
skip) or ``replan`` (stop this run and hand back a successor graph).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Optional

from pydantic import BaseModel

from .graph import DagNode, DagSpec


class Verdict(BaseModel):
    outcome: Literal["accomplished", "not_accomplished"] = "accomplished"
    category: str = ""
    what_is_missing: str = ""
    evidence: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome == "accomplished"


ACCOMPLISHED = Verdict()


@dataclass
class Decision:
    kind: Literal["continue", "abandon", "replan"]
    message: str = ""
    spec: Optional[DagSpec] = None

    @classmethod
    def cont(cls, message: str) -> "Decision":
        return cls("continue", message)

    @classmethod
    def abandon(cls, message: str = "") -> "Decision":
        return cls("abandon", message)

    @classmethod
    def replan(cls, spec: DagSpec, message: str = "") -> "Decision":
        return cls("replan", message, spec)


Judge = Callable[[DagNode, str], Awaitable[Verdict]]
Adjudicator = Callable[[DagNode, str, Verdict, int], Awaitable[Decision]]


async def abandon_all(node: DagNode, output: str, verdict: Verdict, attempt: int) -> Decision:
    return Decision.abandon("no adjudicator configured")
