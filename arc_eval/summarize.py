"""Summarise an LLM run directory produced by ``arc_runner/batch.py``.

    python -m arc_eval.summarize runs/t2_goal8 [--set official]

Per game: levels, actions per level against the human baseline (offline metadata, never shown to
the agent), RHAE as the scorecard computes it, LLM tokens and wall time (from the goal sessions),
and the exploration DAG: cells, surprises (flagged nodes), revisions, surprises left unrevised.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc_eval.datasets import games  # noqa: E402
from arc_eval.eval import rhae  # noqa: E402


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


REPL_FUNCS = ("act(", "show(", "changes(", "node(", "rerun(", "reset(", "history", "frames", "np.")


def usage(ws: Path) -> dict:
    """What the agent actually used: MiniMax Code tool calls (from its session transcripts), turns
    without any tool call, tokens, and REPL features (from the DAG)."""
    tools: Counter = Counter()
    text_only = turns = tokens = 0
    for f in ws.glob(".mcode-data/v2/sessions/*/*/*/*/messages.jsonl"):
        for line in open(f, errors="replace"):
            try:
                m = json.loads(line).get("message") or {}
            except ValueError:
                continue
            if m.get("role") != "assistant":
                continue
            turns += 1
            calls = [c.get("name", "?") for c in m.get("content") or [] if c.get("type") == "toolCall"]
            if not calls:
                text_only += 1
            for c in calls:
                tools[c.replace("mcp__arc__", "")] += 1
            tokens += int((m.get("usage") or {}).get("totalTokens") or 0)
    dag = load(ws / "dag.json") or []
    repl: Counter = Counter()
    for n in dag:
        code = n.get("code", "")
        for fn in REPL_FUNCS:
            if fn in code:
                repl[fn.rstrip("(.")] += 1
        repl["def"] += code.count("def ")
        repl["expect"] += bool(n.get("expect"))
        repl["revises"] += n.get("revises") is not None
        repl["explicit_parents"] += bool(n.get("parents")) and n.get("parents") != [n["id"] - 1]
    return {"tools": tools, "llm_turns": turns, "text_only_turns": text_only, "tokens": tokens, "repl": repl,
            "cells": len(dag)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--set", default="official")
    a = ap.parse_args()
    run = Path(a.run_dir)
    base = {g.game_id: g.baseline for g in games(a.set)}
    allu: list = []
    rows, tot = [], {"levels": 0, "levels_total": 0, "score": 0.0, "tokens": 0, "cells": 0, "flags": 0, "revs": 0}
    for ws in sorted((run / "ws").glob("*_k*")):
        game = ws.name.rsplit("_k", 1)[0]
        led = load(ws / "ledger.json") or {}
        dag = load(ws / "dag.json") or []
        final = load(run / f"{ws.name}.json") or {}
        la = led.get("level_actions") or []
        b = base.get(game)
        score = rhae(la, b) if b else None
        ratios = [f"{n}/{b[i]}" for i, n in enumerate(la)] if b else [str(n) for n in la]
        use = usage(ws)
        tokens = use["tokens"] or sum(int(e.get("tokens_used") or 0) for e in final.get("exec", []))
        allu.append((game, use))
        flags = [n["id"] for n in dag if n.get("flag")]
        revised = {n.get("revises") for n in dag if n.get("revises") is not None}
        rows.append((game, len(la), led.get("win_levels"), led.get("actions"), ratios, score, tokens,
                     led.get("elapsed_s"), len(dag), len(flags), len(revised), len([f for f in flags if f not in revised])))
        tot["levels"] += len(la)
        tot["levels_total"] += int(led.get("win_levels") or 0)
        tot["score"] += score or 0.0
        tot["tokens"] += tokens
        tot["cells"] += len(dag)
        tot["flags"] += len(flags)
        tot["revs"] += len(revised)
    print(f"{'game':14} lv    acts  actions/human per level        RHAE   tokens   time  cells surpr revis open")
    for g, lv, wl, acts, ratios, sc, tok, el, cells, fl, rv, op in rows:
        print(f"{g[:14]:14} {lv}/{wl:<3} {acts or 0:5}  {' '.join(ratios)[:28]:28} {(round(sc, 3) if sc is not None else '-'):>6} "
              f"{tok:8} {int(el or 0):5}s {cells:5} {fl:5} {rv:5} {op:4}")
    tools: Counter = Counter()
    repl: Counter = Counter()
    turns = text_only = 0
    for _, u in allu:
        tools.update(u["tools"])
        repl.update(u["repl"])
        turns += u["llm_turns"]
        text_only += u["text_only_turns"]
    cells = sum(u["cells"] for _, u in allu)
    print(f"\nFEATURE USE  llm turns {turns}, turns without a tool call {text_only} ({text_only / max(1, turns):.0%})")
    print("  tool calls: " + ", ".join(f"{k} {v}" for k, v in tools.most_common()))
    print(f"  REPL cells {cells}: " + ", ".join(f"{k} {v}" for k, v in repl.most_common()))
    n = max(1, len(rows))
    print(json.dumps({"games": len(rows), "levels": tot["levels"], "levels_total": tot["levels_total"],
                      "mean_score": round(tot["score"] / n, 3), "tokens": tot["tokens"], "cells": tot["cells"],
                      "surprises": tot["flags"], "revisions": tot["revs"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
