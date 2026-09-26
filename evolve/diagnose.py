"""Diagnose failing trajectories into WHY x WHERE labels and a living failure map.

Multi-label per trajectory, exactly one ``dominant`` (weight 1.0, others 0.5), as in
Raven's ``nodes/taxonomy.py`` + ``nodes/diagnose.py``. Unknown WHYs are accepted as
``other:<name>`` so the taxonomy can grow (induced classes).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from arc_harness.llm.client import ChatModel
from arc_harness.llm.semantic import call_json

from .state import WHITELIST, WHY_SEED

SYSTEM = (
    "You diagnose why an agent failed ARC-AGI-3 games, so that its harness (prompts, explorer "
    "settings, charter logic, skills) can be improved. Be concrete and evidence-based."
)


class Label(BaseModel):
    why: str
    where: str
    dominant: bool = False
    reasoning: str = ""
    fix_hint: str = ""


class DiagnosisOut(BaseModel):
    labels: list[Label] = Field(default_factory=list, max_length=4)


def _prompt(game: str, text: str) -> str:
    whys = "\n".join(f"- {k}: {v}" for k, v in WHY_SEED.items())
    wheres = ", ".join(WHITELIST)
    return (
        f"Game {game} trajectory (report, notebook, last plans):\n{text}\n\n"
        f"Failure classes (WHY):\n{whys}\n- other:<short_name> if none fits\n\n"
        f"Editable locations (WHERE): {wheres}\n\n"
        "Label the failure with 1-4 {why, where, dominant, reasoning, fix_hint}; exactly one label is "
        "dominant. fix_hint must be a general harness change, never game-specific. "
        'Reply with ONLY JSON {"labels": [...]}.'
    )


async def diagnose(llm: ChatModel, trajectories: list[tuple[str, str]], fmap_path: Path) -> dict:
    fmap = json.loads(fmap_path.read_text()) if fmap_path.exists() else {"why_distribution": {}, "cells": {}}
    for game, text in trajectories:
        try:
            out = await call_json(llm, DiagnosisOut, SYSTEM, _prompt(game, text), retries=2, max_tokens=1500)
        except Exception:
            continue
        labels = out.labels
        if labels and not any(lb.dominant for lb in labels):
            labels[0].dominant = True
        for lb in labels:
            why = lb.why if lb.why in WHY_SEED or lb.why.startswith("other:") else f"other:{lb.why[:30]}"
            w = 1.0 if lb.dominant else 0.5
            fmap["why_distribution"][why] = fmap["why_distribution"].get(why, 0.0) + w
            cell = fmap["cells"].setdefault(f"{lb.where}::{why}", {"candidates": []})
            cell["candidates"].append({"game": game, "reasoning": lb.reasoning[:600], "fix_hint": lb.fix_hint[:400]})
    fmap_path.parent.mkdir(parents=True, exist_ok=True)
    fmap_path.write_text(json.dumps(fmap, indent=1))
    return fmap


def choose_whys(fmap: dict, attempts: dict[str, int], k: int) -> list[str]:
    """count x 0.55^failed_attempts (Raven ``rerank_whys`` without the fixability table)."""
    dist = fmap.get("why_distribution", {})
    ranked = sorted(dist, key=lambda w: -dist[w] * (0.55 ** attempts.get(w, 0)))
    return ranked[:k]


def why_brief(fmap: dict, why: str, max_items: int = 6) -> str:
    items = []
    for key, cell in fmap.get("cells", {}).items():
        if key.endswith(f"::{why}"):
            where = key.split("::")[0]
            for c in cell["candidates"][-max_items:]:
                items.append(f"- [{where}] {c['game']}: {c['reasoning']} | fix hint: {c['fix_hint']}")
    desc = WHY_SEED.get(why, why)
    return f"Target failure class {why}: {desc}\nEvidence:\n" + "\n".join(items[:max_items])
