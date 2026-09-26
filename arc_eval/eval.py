"""Competition-faithful local evaluation: one play per game, RESET rules as on the gateway.

    python -m arc_eval.eval --set official --agent explorer --max-actions 2000
    python -m arc_eval.eval --set community --agent explorer --limit 60 --seed 0

Per game: levels completed, actions per level and (official set) the RHAE game score,
computed like ``arc_agi.scorecard``: level score min(115, 100 * (human/agent)^2) for completed
levels, 0 otherwise, averaged with level-index weights.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc_eval.datasets import GameInfo, games  # noqa: E402


def rhae(level_actions: list[int], baseline: tuple[int, ...]) -> float:
    """``EnvironmentScoreCalculator.to_score``: level-index weights, and the average may not
    exceed 100 x (weight of scoring levels) / (total weight)."""
    num = den = scoring = 0.0
    for i, base in enumerate(baseline):
        w = i + 1
        if i < len(level_actions) and level_actions[i] > 0:
            num += w * min(115.0, 100.0 * (base / level_actions[i]) ** 2)
            scoring += w
        den += w
    return min(num / den, 100.0 * scoring / den) if den else 0.0


def play(info: GameInfo, agent: str, max_actions: int, seed: int) -> dict[str, Any]:
    logging.disable(logging.CRITICAL)
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState

    from arc_mcp.explorer import play_explore
    from arc_mcp.session import Session

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(info.env_dir))
    card = arc.open_scorecard()
    s = Session(arc.make(info.game_id, scorecard_id=card))
    obs = s.start()
    t0 = time.time()
    if agent == "explorer":
        play_explore(s, max_actions=max_actions)
    elif agent == "random":
        rng = random.Random(seed)
        while s.actions < max_actions and s.last.state != GameState.WIN:
            if s.last.state == GameState.GAME_OVER:
                s.step(0)
                continue
            a = rng.choice([x for x in s.last.available_actions if x] or [1])
            s.step(a, rng.randrange(64), rng.randrange(64))
    else:
        raise ValueError(agent)
    out: dict[str, Any] = {
        "game": info.game_id, "tags": list(info.tags), "levels": len(s.level_actions),
        "win_levels": s.win_levels, "actions": s.actions, "level_actions": s.level_actions,
        "deaths": s.deaths, "resets": s.resets, "state": s.last.state.name,
        "ms_per_action": round(1000 * (time.time() - t0) / max(1, s.actions), 2),
    }
    if info.baseline:
        out["score"] = round(rhae(s.level_actions, info.baseline), 3)
        try:  # cross-check against the engine's own scorecard
            c = arc.get_scorecard(card).model_dump()
            env = next(e for e in c["environments"] if e["id"] == info.game_id)
            out["engine_score"] = round(float(env["score"]), 3)
        except Exception:
            pass
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="official", choices=["official", "community"])
    p.add_argument("--agent", default="explorer", choices=["explorer", "random"])
    p.add_argument("--max-actions", type=int, default=2000)
    p.add_argument("--limit", type=int, default=0, help="play a random subset of this many games")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    gs = games(a.set)
    if a.limit:
        gs = sorted(random.Random(a.seed).sample(gs, min(a.limit, len(gs))), key=lambda g: g.game_id)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(play, gs, [a.agent] * len(gs), [a.max_actions] * len(gs), [a.seed] * len(gs)))
    for r in res:
        print(f"{r['game'][:12]:12} lv {r['levels']}/{r['win_levels']} acts {r['actions']:5} "
              f"per-level {r['level_actions']} deaths {r['deaths']} {r.get('score', '')} {r['ms_per_action']}ms")
    summary = {
        "set": a.set, "agent": a.agent, "max_actions": a.max_actions, "games": len(res),
        "levels": sum(r["levels"] for r in res), "levels_total": sum(r["win_levels"] for r in res),
        "games_with_a_level": sum(r["levels"] > 0 for r in res),
    }
    if any("score" in r for r in res):
        summary["mean_score"] = round(sum(r.get("score", 0) for r in res) / len(res), 3)
    print(json.dumps(summary))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "games": res}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
