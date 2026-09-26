"""Read the engine's own scorecard (the official RHAE implementation) into a flat summary.

This module is part of the evolver's immutable kernel: candidates may not edit it.
"""

from __future__ import annotations

from typing import Any


def summarize_scorecard(card: Any) -> dict[str, Any]:
    if card is None:
        return {"games": {}, "totals": {}}
    data = card.model_dump() if hasattr(card, "model_dump") else dict(card)
    games: dict[str, Any] = {}
    for env in data.get("environments", []):
        runs = env.get("runs") or []
        best = max(runs, key=lambda r: (r.get("score") or 0.0), default={})
        games[env["id"]] = {
            "score": float(env.get("score") or 0.0),
            "levels_completed": int(env.get("levels_completed") or 0),
            "level_count": int(env.get("level_count") or 0),
            "actions": int(env.get("actions") or 0),
            "resets": int(env.get("resets") or 0),
            "level_scores": best.get("level_scores"),
            "level_actions": best.get("level_actions"),
            "level_baseline_actions": best.get("level_baseline_actions"),
        }
    n = max(len(games), 1)
    totals = {
        "score": float(data.get("score") or 0.0),
        "mean_game_score": sum(g["score"] for g in games.values()) / n,
        "games": len(games),
        "games_with_progress": sum(g["levels_completed"] > 0 for g in games.values()),
        "levels_completed": sum(g["levels_completed"] for g in games.values()),
        "levels_total": sum(g["level_count"] for g in games.values()),
        "actions": sum(g["actions"] for g in games.values()),
    }
    return {"games": games, "totals": totals}


def game_score(summary: dict[str, Any], game_id: str) -> float:
    for k, v in summary.get("games", {}).items():
        if k == game_id or k.startswith(game_id):
            return float(v["score"])
    return 0.0
