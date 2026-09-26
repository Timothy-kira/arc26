"""Compare run summaries (summary.json from arc_harness.run) game by game.

Usage: python scripts/compare_runs.py runs/baseline/summary.json runs/kaggle_dev/dev_run/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: str) -> dict:
    return json.loads(Path(p).read_text())


def main() -> None:
    runs = [(Path(p).parent.name or p, load(p)) for p in sys.argv[1:]]
    games = sorted({g for _, r in runs for g in r.get("games", {})})
    head = f"{'game':16s}" + "".join(f"{name[:18]:>20s}" for name, _ in runs)
    print(head)
    for g in games:
        row = f"{g:16s}"
        for _, r in runs:
            v = r["games"].get(g)
            row += f"{'-':>20s}" if v is None else f"{v['levels_completed']:>3d}/{v['level_count']:<3d} {v['score']:7.3f} {v['actions']:5d}"
        print(row)
    print()
    for name, r in runs:
        t = r.get("totals", {})
        reps = r.get("reports") or []
        rounds = sum(x.get("llm_rounds", 0) for x in reps)
        replans = sum(x.get("replans", 0) for x in reps)
        errors = [x["game_id"] for x in reps if x.get("error")]
        skills = sorted({s for x in reps for s in x.get("skills_written", [])})
        llm = r.get("llm") or {}
        per_call = llm.get("seconds", 0) / max(llm.get("calls", 0), 1)
        print(
            f"{name}: score={t.get('score', 0):.3f} levels={t.get('levels_completed')}/{t.get('levels_total')} "
            f"actions={t.get('actions')} rounds={rounds} replans={replans} errors={errors} "
            f"llm_calls={llm.get('calls', 0)} llm_fail={llm.get('failures', 0)} s/call={per_call:.1f} "
            f"tok_out={llm.get('completion_tokens', 0)} wall={r.get('wall_seconds', 0):.0f}s skills_written={len(skills)}"
        )


if __name__ == "__main__":
    main()
