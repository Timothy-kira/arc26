"""What the model reads every step and the JSON it must answer with.

Every step is a fresh conversation: system prompt + one user message holding the whole graph, the
previous step's handoff and prediction (checked against what happened), the diff of the last
action and the current frame (image, objects, hex grid). The answer is one JSON object with
``graph_update``, ``action``, ``prediction`` and ``handoff``; the loop validates it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from arc_agent.graph import EDGE_TYPES, NODE_TYPES

SYSTEM = f"""You are playing an unknown turn-based game. Nobody tells you the rules or the goal: you find
them by acting and observing. The screen is a 64x64 grid of colours 0-15 (hex digits 0-f). The game
has several levels; clearing a level shows the next one. Your score for a level is (human actions /
your actions)^2: every action counts, so think before acting and never waste moves.

Actions: 1-4 are directional (their meaning differs per game: find out), 5 is interact, 6 is a click
at (x, y) on the grid, 7 is undo, 0 is RESET (restarts the level; only after GAME_OVER). Only the
action ids listed as available work now.

You have no memory of earlier steps except two things, and you must read both completely before
deciding anything:
- THE GRAPH: every observation, rule, hypothesis, goal, plan and question recorded so far (by you) and
  every action with its checked outcome (by the game loop). Edges link them symbolically.
- THE HANDOFF: the note you left yourself last step, and the PREDICTION you made for that step, with
  the verdict of whether it came true.

The conversation may continue across steps: then each new turn brings only what changed (the check,
the diff, the new graph nodes, the frame). It is compacted from time to time (and at every new level
or GAME_OVER): a fresh segment opens with the whole graph and your last handoff, and nothing else
survives. So whatever you will need later must be in the graph or the handoff.

Each step, answer with exactly one JSON object (you may think first, but the reply must end with the
JSON in a ```json block):
{{
  "graph_update": {{
    "nodes": [{{"type": one of {list(NODE_TYPES[:6])}, "text": "short, specific, with coordinates/colours"}}],
    "edges": [{{"src": "n0" or an existing id, "dst": ..., "rel": one of {list(EDGE_TYPES)}}}],
    "status": [{{"id": existing id, "status": "confirmed" | "refuted" | "done" | "open"}}]
  }},
  "action": {{"id": <int>, "x": <int, only for 6>, "y": <int, only for 6>}},
  "prediction": {{
    "board_changes": true | false,
    "moves": [{{"color": <int>, "dx": <int>, "dy": <int>}}],
    "level_up": true | false,
    "game_over": true | false,
    "text": "what exactly you expect to see after this action"
  }},
  "handoff": "note to your next step: current goal, plan for the next few actions, what is still uncertain",
  "think_next": true | false,
  "need_grid": false
}}

Rules for the graph:
- Add at least one new node every step (new nodes are "n0", "n1", ... in this update's edges).
- When the last prediction was WRONG, add a node saying what actually happens and link it to the rule or
  hypothesis it contradicts with "refutes" (and mark that node refuted). When it was RIGHT, mark the rule
  it tested confirmed.
- Keep nodes short and factual. Prefer rules that generalise ("ACTION1 moves the blue 5x5 up by 5").
- Record the goal as soon as you have a hypothesis about what clears a level.

Thinking time is the scarcest resource you have: every minute you think is a minute of game time.
- When the handoff holds a plan and the last prediction was RIGHT, just take the next planned action;
  do not re-derive the map or re-check rules that the graph marks confirmed.
- When you work something out (a map of walls and corridors, a route, the effect of a button), write
  the result into the graph so later steps can read it instead of working it out again.
- Think hard only when a prediction was WRONG, a level just started, or the plan is finished.
- "think_next": true when the next step needs fresh analysis, false when it is a routine continuation of
  the plan in your handoff (it may then run without deliberate reasoning). If this step's prediction
  fails, a level ends or the game is over, the next step thinks anyway.

A region marked SIDE EFFECT changed away from what moved: something the move touched (a switch, a
key, a door, a counter, a legend) reacted. These are usually the mechanism of the level: record what
triggered it and what it changed.

Playing well: first learn what each action does (one test each is usually enough), find what you
control and what the goal is, then move straight to it. The HUD (a bar or counter that changes every
action) is usually a move budget, not the board. A predicted move (color, dx, dy) counts as seen when an
object of that colour moved by it, or when an edge of that colour's extent moved by it (so a bar that
grows by 2 upwards is {{"color": c, "dx": 0, "dy": -2}}); dx is right, dy is down, in grid cells."""


HURRY = ("Your previous attempt at this step thought past the time limit and was discarded. Decide now: keep "
         "the reasoning short, rely on the graph and the handoff, and answer with the JSON object.")


def status_text(obs: dict[str, Any]) -> str:
    return (f"level {obs['level'] + 1}/{obs['win_levels']} | state {obs['state']} | actions this level "
            f"{obs['level_actions']} | total {obs['actions']} | available actions {obs['available']} | "
            f"step {obs['step']} | time left {obs['time_left']:.0f}s")


def user_message(obs: dict[str, Any], graph_text: str, handoff: str, verdict: str, diff: str,
                 objects: str, grid: str, image_b64: Optional[str], problems: list[str], lattice: str = "") -> dict:
    """Ordered for the server's prefix cache: the graph (which mostly grows at its end) comes right after
    the fixed system prompt, and everything that changes every step (handoff, check, diff, status,
    frame) follows it."""
    parts = ["THE GRAPH:", graph_text, "",
             f"THE HANDOFF (from your previous step):\n{handoff or '(none: first step)'}", "",
             f"PREDICTION CHECK for the last action:\n{verdict or '(no previous action)'}", "",
             f"WHAT THE LAST ACTION CHANGED:\n{diff or '(no previous action)'}"]
    if problems:
        parts += ["", "PROBLEMS WITH YOUR LAST ANSWER (fix them this step):", *problems]
    parts += ["", f"STATUS: {status_text(obs)}", "", "CURRENT FRAME objects:", objects]
    if lattice:
        parts += ["", "CURRENT FRAME as a cell map (use it for maps and paths; the full grid follows):", lattice]
    parts += ["", "CURRENT FRAME grid (hex; rows and columns that are only background are left out, labels "
              "are absolute y and x):", grid, "", "Answer with the JSON object now."]
    content: list[dict] = [{"type": "text", "text": "\n".join(parts)}]
    if image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    return {"role": "user", "content": content}


def step_message(obs: dict[str, Any], new_nodes: str, verdict: str, diff: str, objects: str,
                 lattice: str, grid: Optional[str], image_b64: Optional[str], problems: list[str],
                 rows: str = "") -> dict:
    """The next turn of a continuing conversation: only what is new since the last answer. The graph,
    the handoff and the first full frame (grid and objects) are already in the conversation; every
    token here stays in it until the next compaction, so the turn is kept short."""
    parts = [f"NEW GRAPH NODES:\n{new_nodes}", f"CHECK: {verdict}", f"DIFF:\n{diff}"]
    if problems:
        parts += ["PROBLEMS WITH YOUR LAST ANSWER (fix them now):", *problems]
    parts += [f"STATUS: {status_text(obs)}"]
    if lattice:
        parts += ["CELL MAP:", lattice]
    if grid:
        parts += ["FULL GRID (hex, labels absolute):", grid, "OBJECTS:", objects]
    elif rows:
        parts += ["CHANGED ROWS (hex, full width; other rows unchanged; need_grid for the whole frame):", rows]
    parts += ["Answer with the JSON object (graph_update required)."]
    content: list[dict] = [{"type": "text", "text": "\n".join(parts)}]
    if image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    return {"role": "user", "content": content}


def parse(text: str) -> Optional[dict]:
    """The last JSON object in the reply (a ```json block if there is one)."""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cands = blocks[::-1] or [text[text.find("{"): text.rfind("}") + 1]]
    for c in cands:
        try:
            d = json.loads(c)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            continue
    return None


def validate(d: Optional[dict], available: list[int], state: str) -> list[str]:
    """Problems that make an answer unusable (the loop asks again with these)."""
    if d is None:
        return ["the reply had no parsable JSON object"]
    errs = []
    nodes = ((d.get("graph_update") or {}).get("nodes") or [])
    if not any(isinstance(n, dict) and str(n.get("text", "")).strip() for n in nodes):
        errs.append("graph_update.nodes must add at least one node")
    a = d.get("action") or {}
    try:
        aid = int(a.get("id"))
    except (TypeError, ValueError):
        return errs + ["action.id must be an integer"]
    if state == "GAME_OVER" and aid != 0:
        errs.append("the game is over: the only useful action is 0 (RESET)")
    elif aid not in available and not (aid == 0 and state == "GAME_OVER"):
        errs.append(f"action {aid} is not available now; available: {available}")
    if aid == 6 and not (isinstance(a.get("x"), int) and isinstance(a.get("y"), int)):
        errs.append("action 6 needs integer x and y")
    if not isinstance(d.get("prediction"), dict):
        errs.append("prediction must be an object")
    if not str(d.get("handoff", "")).strip():
        errs.append("handoff must be a non-empty note")
    return errs


def check_prediction(pred: dict, facts: dict) -> tuple[Optional[bool], str]:
    """Compare a prediction with the facts of the action (diff, level-up, game over)."""
    checks: list[tuple[str, bool]] = []
    if "board_changes" in pred and isinstance(pred["board_changes"], bool):
        checks.append((f"board changes: expected {pred['board_changes']}, got {facts['board_changed'] > 0}",
                       pred["board_changes"] == (facts["board_changed"] > 0)))
    actual = {(m["color"], m["dx"], m["dy"]) for m in facts["moves"]}
    ext: dict[int, list] = {}
    for c, *e in facts.get("extents") or []:
        ext.setdefault(int(c), []).append(e)

    def seen(c: int, dx: int, dy: int) -> bool:
        """A matched object moved by (dx, dy), or an edge of the colour's extent did (a bar that grows
        or shrinks, one of several identical tokens)."""
        if (c, dx, dy) in actual:
            return True
        for l, t, r, b in ext.get(c, []):
            if (dx in (l, r) or dx == 0 and 0 in (l, r)) and (dy in (t, b) or dy == 0 and 0 in (t, b)):
                return True
        return False

    for m in pred.get("moves") or []:
        try:
            key = (int(m["color"]), int(m["dx"]), int(m["dy"]))
        except (KeyError, TypeError, ValueError):
            continue
        hit = seen(*key)
        checks.append((f"move colour {key[0]} by ({key[1]:+d},{key[2]:+d}): {'seen' if hit else 'not seen'}", hit))
    for k in ("level_up", "game_over"):
        if isinstance(pred.get(k), bool):
            checks.append((f"{k}: expected {pred[k]}, got {facts[k]}", pred[k] == facts[k]))
    if not checks:
        return None, "no checkable prediction was made"
    ok = all(c for _, c in checks)
    return ok, ("RIGHT" if ok else "WRONG") + ": " + "; ".join(t for t, _ in checks) + \
        (f". Expected: {pred.get('text', '')}" if pred.get("text") else "")
