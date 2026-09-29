"""Play a set of games with the agent loop, one process per game.

    python -m arc_agent.run --out-dir runs/a1 --games ls20,vc33,tu93 --conc 3 \
        --base-url https://host/v1 --model m --api-key-file .secrets/dots_api_key --game-seconds 2700

Offline it plays the public games from --env-dir. With --gateway (the Kaggle rerun) it opens the
one scorecard the gateway allows, every game joins it, and it is closed at the end.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def one(game: str, k: int, a: dict[str, Any], card: Optional[str], seconds: float) -> dict[str, Any]:
    from arc_agent.llm import LLM
    from arc_agent.loop import play

    logging.disable(logging.CRITICAL)
    key = Path(a["api_key_file"]).read_text().strip() if a["api_key_file"] else "EMPTY"
    from arc_agent.prompt import HURRY

    llm = LLM(a["base_url"], a["model"], key, thinking=not a["no_thinking"], call_seconds=a["call_seconds"], hurry=HURRY)
    out = Path(a["out_dir"]) / "games" / f"{game}_k{k}"
    try:
        return play(game, out, llm, a["env_dir"], a["gateway"], card, seconds, a["max_actions"], a["think_policy"])
    except Exception as exc:  # one broken game must not stop the batch
        return {"game": game, "k": k, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--games", default="all")
    p.add_argument("--k", type=int, default=1, help="plays per game (offline)")
    p.add_argument("--conc", type=int, default=3)
    p.add_argument("--env-dir", default=str(ROOT / "data" / "environment_files"))
    p.add_argument("--gateway", default=None)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-file", default="")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--think-policy", default="model", choices=["always", "model", "exception"],
                   help="which steps think: every step, those the model does not mark routine (default), or "
                        "only when needed (first steps, failed prediction, new level, game over, model asks)")
    p.add_argument("--call-seconds", type=float, default=180.0, help="wall-clock limit of one LLM call")
    p.add_argument("--game-seconds", type=float, default=2700.0)
    p.add_argument("--hours", type=float, default=8.0, help="whole-run budget; games share what is left")
    p.add_argument("--max-actions", type=int, default=2000)
    a = vars(p.parse_args())
    out = Path(a["out_dir"]).resolve()
    a["out_dir"] = str(out)
    out.mkdir(parents=True, exist_ok=True)

    from arc_agi import Arcade, OperationMode

    quiet = logging.getLogger("arc.engine")
    card = None
    if a["gateway"]:
        arc = Arcade(arc_api_key="test-key-123", arc_base_url=a["gateway"].rstrip("/"),
                     operation_mode=OperationMode.ONLINE, logger=quiet)
        card = arc.open_scorecard(tags=["arc26-agent"])
    else:
        arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=a["env_dir"], logger=quiet)
    all_games = sorted(e.game_id for e in arc.get_environments())
    wanted = all_games if a["games"] == "all" else a["games"].split(",")
    games = [g for g in all_games if any(g == w or g.startswith(w) for w in wanted)]
    jobs = [(g, k) for k in range(1 if a["gateway"] else a["k"]) for g in games]
    t_end = time.time() + a["hours"] * 3600
    results = []
    try:
        with ProcessPoolExecutor(max_workers=a["conc"]) as pool:
            futs = {}
            waves = max(1.0, len(jobs) / a["conc"])
            share = max(300.0, min(a["game_seconds"], (t_end - time.time()) / waves))
            for g, k in jobs:
                futs[pool.submit(one, g, k, a, card, share)] = (g, k)
            for f in as_completed(futs):
                r = f.result()
                r.setdefault("k", futs[f][1])
                results.append(r)
                sys.stderr.write(f"[run] {r.get('game')} k{r.get('k')} levels {r.get('levels')}/{r.get('win_levels')} "
                                 f"actions {r.get('actions')} {r.get('error', '')}\n")
    finally:
        if card:
            for _ in range(3):  # closing finalizes the submission on the gateway
                try:
                    arc.close_scorecard(card)
                    break
                except Exception as exc:
                    sys.stderr.write(f"close_scorecard failed: {exc}\n")
    (out / "results.json").write_text(json.dumps(results, indent=1))
    from arc_agent.report import report

    print(report(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
