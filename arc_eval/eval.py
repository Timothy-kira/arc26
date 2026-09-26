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


def play(info: GameInfo, agent: str, max_actions: int, seed: int, max_seconds: float = 300.0) -> dict[str, Any]:
    logging.disable(logging.CRITICAL)
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState

    import numpy as np

    from arc_mcp.explorer import play_explore
    from arc_mcp.session import Session

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(info.env_dir))
    card = arc.open_scorecard()
    s = Session(arc.make(info.game_id, scorecard_id=card))
    obs = s.start()
    t0 = time.time()
    real_step = s.step

    def capped(*args, **kw):
        if time.time() - t0 > max_seconds:
            s.max_actions_hit = True
            raise TimeoutError
        return real_step(*args, **kw)

    s.step = capped
    try:
        _run(agent, s, info, max_actions, seed)
    except TimeoutError:
        pass
    return _result(info, s, t0, arc, card)


def _run(agent, s, info, max_actions, seed):
    import numpy as np
    from arcengine import GameState

    from arc_mcp.explorer import play_explore

    if agent == "explorer":
        play_explore(s, max_actions=max_actions)
    elif agent == "memory":  # the MCP's own curiosity policy (arc_explore) with the rule model
        from arc_mcp.memory import GameMemory
        m = GameMemory(s, baseline=list(info.baseline) if info.baseline else None)
        while s.actions < max_actions and s.last.state != GameState.WIN:
            m.explore(min(50, max_actions - s.actions), stop_on_event=False)
    elif agent == "novelty":  # explore, then walk the avatar to every colour it has not entered yet
        from arc_mcp.memory import GameMemory
        from arc_mcp.planner import plan
        m = GameMemory(s, baseline=list(info.baseline) if info.baseline else None)
        tried: dict[int, set] = {}
        while s.actions < max_actions and s.last.state != GameState.WIN:
            lvl = s.last.levels_completed
            m.explore(min(30, max_actions - s.actions))
            if s.last.levels_completed != lvl or not m.model.avatar():
                continue
            g = s.last.grid
            bg = int(np.bincount(g.ravel()).argmax())
            av = m.model.avatar()[0][0]
            cand = [c for c in np.unique(g).tolist() if c not in (bg, av) and m.model.entered.get(c, 0) == 0
                    and c not in tried.setdefault(lvl, set())]
            if not cand:
                continue
            c = min(cand, key=lambda c: int((g == c).sum()))  # rarest colour first
            tried[lvl].add(c)
            plan(m, {"reach": {"color": int(c)}}, max_actions=min(80, max_actions - s.actions))
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


def _result(info, s, t0, arc, card):
    out: dict[str, Any] = {
        "game": info.game_id, "tags": list(info.tags), "levels": len(s.level_actions),
        "win_levels": s.win_levels, "actions": s.actions, "level_actions": s.level_actions,
        "deaths": s.deaths, "resets": s.resets, "state": s.last.state.name,
        "ms_per_action": round(1000 * (time.time() - t0) / max(1, s.actions), 2),
        "timed_out": bool(getattr(s, "max_actions_hit", False)),
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
    p.add_argument("--agent", default="explorer", choices=["explorer", "memory", "novelty", "random"])
    p.add_argument("--max-actions", type=int, default=2000)
    p.add_argument("--limit", type=int, default=0, help="play a random subset of this many games")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-seconds", type=float, default=300.0, help="wall-time cap per game")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    gs = games(a.set)
    if a.limit:
        gs = sorted(random.Random(a.seed).sample(gs, min(a.limit, len(gs))), key=lambda g: g.game_id)
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(play, g, a.agent, a.max_actions, a.seed, a.max_seconds) for g in gs]
        for f in futs:
            r = f.result()
            res.append(r)
            print(f"{r['game'][:12]:12} lv {r['levels']}/{r['win_levels']} acts {r['actions']:5} "
                  f"per-level {r['level_actions']} deaths {r['deaths']} {r.get('score', '')} {r['ms_per_action']}ms"
                  + (" TIMEOUT" if r["timed_out"] else ""), flush=True)
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
