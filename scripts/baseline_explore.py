"""Run the LLM-free explorer on every local game and print the official scorecard."""

from __future__ import annotations

import argparse
import json
import time

from arc_harness.env.arcade import ArcEnv
from arc_harness.explore.explorer import play_explore
from arc_harness.scoring import summarize_scorecard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="all")
    ap.add_argument("--max-actions", type=int, default=600)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    env = ArcEnv("offline")
    games = env.list_games() if args.games == "all" else [
        g for g in env.list_games() if any(g.startswith(p) for p in args.games.split(","))
    ]
    env.open(tags=["baseline-explore"])
    t0 = time.time()
    for gid in games:
        s = env.make(gid)
        ex = play_explore(s, args.max_actions)
        g = ex.graph.stats() if ex.graph else {}
        won = [lv for lv, log in ex.levels.items() if log.won]
        print(f"{gid:16s} tags={','.join(s.tags) or '-':14s} actions={s.actions_taken:4d} "
              f"levels_won={len(won)} states={g.get('states')} hud={g.get('hud_masked_cells')}")
    summary = summarize_scorecard(env.scorecard())
    print(json.dumps(summary["totals"], indent=2))
    print(f"wall {time.time() - t0:.1f}s")
    if args.out:
        import os

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
