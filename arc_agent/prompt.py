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
  "handoff": "note to your next step: current goal, plan for the next few actions, what is still uncertain"
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

Playing well: first learn what each action does (one test each is usually enough), find what you
control and what the goal is, then move straight to it. The HUD (a bar or counter that changes every
action) is usually a move budget, not the board. Moves given by "moves" are shape-matched objects with
their displacement in grid cells (dx right, dy down)."""


def status_text(obs: dict[str, Any]) -> str:
    return (f"level {obs['level'] + 1}/{obs['win_levels']} | state {obs['state']} | actions this level "
            f"{obs['level_actions']} | total {obs['actions']} | available actions {obs['available']} | "
            f"step {obs['step']} | time left {obs['time_left']:.0f}s")


def user_message(obs: dict[str, Any], graph_text: str, handoff: str, verdict: str, diff: str,
                 objects: str, grid: str, image_b64: Optional[str], problems: list[str], lattice: str = "") -> dict:
    parts = [f"STATUS: {status_text(obs)}", "", "THE GRAPH:", graph_text, "",
             f"THE HANDOFF (from your previous step):\n{handoff or '(none: first step)'}", "",
             f"PREDICTION CHECK for the last action:\n{verdict or '(no previous action)'}", "",
             f"WHAT THE LAST ACTION CHANGED:\n{diff or '(no previous action)'}"]
    if problems:
        parts += ["", "PROBLEMS WITH YOUR LAST ANSWER (fix them this step):", *problems]
    parts += ["", "CURRENT FRAME objects:", objects]
    if lattice:
        parts += ["", "CURRENT FRAME as a cell map (use it for maps and paths; the full grid follows):", lattice]
    parts += ["", "CURRENT FRAME grid (hex, row y then columns x):", grid,
              "", "Answer with the JSON object now."]
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
    for m in pred.get("moves") or []:
        try:
            key = (int(m["color"]), int(m["dx"]), int(m["dy"]))
        except (KeyError, TypeError, ValueError):
            continue
        checks.append((f"move colour {key[0]} by ({key[1]:+d},{key[2]:+d}): {'seen' if key in actual else 'not seen'}",
                       key in actual))
    for k in ("level_up", "game_over"):
        if isinstance(pred.get(k), bool):
            checks.append((f"{k}: expected {pred[k]}, got {facts[k]}", pred[k] == facts[k]))
    if not checks:
        return None, "no checkable prediction was made"
    ok = all(c for _, c in checks)
    return ok, ("RIGHT" if ok else "WRONG") + ": " + "; ".join(t for t, _ in checks) + \
        (f". Expected: {pred.get('text', '')}" if pred.get("text") else "")
