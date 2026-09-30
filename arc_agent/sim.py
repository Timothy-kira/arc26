"""The model's simulator of the game: code it writes, checked against the real game, searched for plans.

The model writes Python (a ```python block in its reply) that defines

    step(grid, action, x, y) -> grid   the frame after an action (grid: 64x64 numpy int array; action
                                       1-7; x, y only for 6)
    goal(grid) -> bool                 optional: the frame clears the level (needed for a search)
    key(grid) -> hashable              optional: what makes two frames the same state (default: the whole
                                       frame; leave out a move counter so the search does not blow up)
    clicks(grid) -> [(x, y), ...]      optional: where ACTION6 is worth trying (default: one cell of each
                                       object of the frame the search starts from)

The loop checks every real action against ``step`` (cells it got wrong, HUD cells apart), replays the
level's recent transitions when new code arrives, and on request searches the simulator (BFS) for a
route from the current frame to ``goal``; the route is played while the simulator keeps matching the
real frames, so a wrong simulator costs one action, not the whole route.

The code runs in a separate process with a time limit (it may loop or crash; that is reported, not
fatal).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import deque
from typing import Any, Optional

import numpy as np

HELP = """THE SIMULATOR (part of every answer, like the graph update): Python in a ```python block after
your JSON, replacing the previous version. numpy is available as np. Define
  step(grid, action, x, y) -> grid   the next frame (grid: 64x64 int numpy array, index grid[y, x]; action
                                     1-7; x, y only for 6). Return a new array; do not modify the input.
  goal(grid) -> bool                 the frame clears the level (your current goal hypothesis)
  key(grid) -> hashable              optional: the state that matters (e.g. the avatar position and the
                                     board without the move counter); default the whole frame
  clicks(grid) -> [(x, y), ...]      optional: where ACTION6 is worth trying
The game loop enforces it:
- from the 3rd action of a level an answer without a simulator is sent back; so is one that keeps a
  simulator that got the last action wrong (send the corrected version);
- a new version is replayed on the level's recent real actions and rejected if it crashes or gets more
  cells wrong than the current one;
- every real action is checked against step() (HUD cells are not counted) and the result is linked to
  the simulator's node in the graph;
- once step() has predicted 3 actions in a row exactly, the loop searches it (BFS) for the shortest
  route to goal() before each step and shows the route; "plan": true plays it while the real frames
  match step() (it stops at the first mismatch, a new level or GAME_OVER); without a route your
  "action" is played."""


def _load(code: str) -> dict:
    ns: dict[str, Any] = {"np": np, "numpy": np}
    exec(compile(code, "<simulator>", "exec"), ns)  # noqa: S102 (the model's own simulator, in a child process)
    if not callable(ns.get("step")):
        raise ValueError("the simulator must define step(grid, action, x, y)")
    return ns


def _step(ns: dict, g: np.ndarray, a: int, x: Optional[int], y: Optional[int]) -> np.ndarray:
    out = np.asarray(ns["step"](g.copy(), a, x, y))
    if out.shape != g.shape:
        raise ValueError(f"step() returned shape {out.shape}, expected {g.shape}")
    return out.astype(np.int16)


def _wrong(pred: np.ndarray, real: np.ndarray, mask: Optional[np.ndarray]) -> dict:
    d = pred != real
    hud = int((d & mask).sum()) if mask is not None else 0
    if mask is not None:
        d &= ~mask
    ys, xs = np.nonzero(d)
    out: dict[str, Any] = {"wrong": int(d.sum()), "hud_wrong": hud}
    if len(ys):
        out["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        out["cells"] = [[int(x), int(y), int(pred[y, x]), int(real[y, x])] for y, x in list(zip(ys, xs))[:8]]
    return out


def _default_clicks(g: np.ndarray, limit: int = 48) -> list[tuple[int, int]]:
    from arc_agent.vision import objects

    pts = []
    for o in sorted(objects(g), key=lambda o: o["size"])[:limit]:
        x0, y0, x1, y1 = o["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ys, xs = np.nonzero(g[y0:y1 + 1, x0:x1 + 1] == o["color"])
        i = int(np.argmin((xs + x0 - cx) ** 2 + (ys + y0 - cy) ** 2))
        pts.append((int(xs[i] + x0), int(ys[i] + y0)))
    return pts


def _search(ns: dict, g0: np.ndarray, actions: list[int], seconds: float, max_nodes: int, max_len: int) -> dict:
    if not callable(ns.get("goal")):
        return {"found": False, "reason": "the simulator defines no goal(grid)"}
    key = ns.get("key") if callable(ns.get("key")) else (lambda g: g.tobytes())
    base_clicks = _default_clicks(g0) if 6 in actions and not callable(ns.get("clicks")) else []
    t_end = time.time() + seconds
    if ns["goal"](g0.copy()):
        return {"found": True, "path": [], "nodes": 1}
    seen = {key(g0)}
    q = deque([(g0, [])])
    nodes = 0
    while q:
        g, path = q.popleft()
        if len(path) >= max_len:
            continue
        moves: list[tuple[int, Optional[int], Optional[int]]] = [(a, None, None) for a in actions if a not in (0, 6, 7)]
        if 6 in actions:
            pts = ns["clicks"](g.copy()) if callable(ns.get("clicks")) else base_clicks
            moves += [(6, int(x), int(y)) for x, y in pts]
        for a, x, y in moves:
            nodes += 1
            if nodes > max_nodes or time.time() > t_end:
                return {"found": False, "reason": f"no route within {nodes - 1} simulated actions "
                        f"({'time' if time.time() > t_end else 'node'} limit), depth {len(path)}"}
            h = _step(ns, g, a, x, y)
            k = key(h)
            if k in seen:
                continue
            seen.add(k)
            p = path + [[a, x, y]]
            if ns["goal"](h.copy()):
                return {"found": True, "path": p, "nodes": nodes}
            q.append((h, p))
    return {"found": False, "reason": f"no route: all {len(seen)} reachable states explored, none meets goal()"}


def _worker() -> None:
    req = json.loads(sys.stdin.read())
    try:
        ns = _load(req["code"])
        op = req["op"]
        mask = np.array(req["mask"], dtype=bool) if req.get("mask") is not None else None
        if op == "check":
            pred = _step(ns, np.array(req["grid"], dtype=np.int16), req["action"], req.get("x"), req.get("y"))
            res = _wrong(pred, np.array(req["after"], dtype=np.int16), mask)
        elif op == "replay":
            rows = []
            for t in req["transitions"]:
                pred = _step(ns, np.array(t["grid"], dtype=np.int16), t["action"], t.get("x"), t.get("y"))
                rows.append(_wrong(pred, np.array(t["after"], dtype=np.int16), mask)["wrong"])
            res = {"wrong": rows}
        elif op == "search":
            res = _search(ns, np.array(req["grid"], dtype=np.int16), req["actions"], req["seconds"],
                          req.get("max_nodes", 40000), req.get("max_len", 60))
        else:
            res = {"error": f"unknown op {op}"}
    except Exception as exc:  # the model's code: report its error
        import traceback

        tb = traceback.format_exc().strip().splitlines()
        res = {"error": f"{type(exc).__name__}: {exc}", "trace": [l for l in tb if "<simulator>" in l][-3:]}
    sys.stdout.write(json.dumps(res))


def run(code: str, op: str, seconds: float, **req: Any) -> dict:
    """One operation on the model's code in a child process; a hang or crash comes back as an error."""
    for k in ("grid", "after", "mask"):
        if isinstance(req.get(k), np.ndarray):
            req[k] = req[k].astype(int).tolist()
    for t in req.get("transitions") or []:
        for k in ("grid", "after"):
            t[k] = np.asarray(t[k]).astype(int).tolist()
    try:
        p = subprocess.run([sys.executable, "-m", "arc_agent.sim"], input=json.dumps({"code": code, "op": op, "seconds": seconds, **req}),
                           capture_output=True, text=True, timeout=seconds + 10)
        return json.loads(p.stdout) if p.stdout.strip() else {"error": (p.stderr.strip().splitlines() or ["no output"])[-1]}
    except subprocess.TimeoutExpired:
        return {"error": f"the simulator ran past {seconds + 10:.0f}s"}


def error_text(res: dict) -> str:
    return "SIMULATOR ERROR: " + res["error"] + ("".join("\n  " + l.strip() for l in res.get("trace") or []))


def check_text(res: dict) -> str:
    if "error" in res:
        return error_text(res)
    if not res["wrong"]:
        return "SIMULATOR: step() predicted this frame exactly" + \
            (f" (HUD cells apart: {res['hud_wrong']})" if res.get("hud_wrong") else "")
    cells = "; ".join(f"({x},{y}) predicted {p} got {r}" for x, y, p, r in res.get("cells") or [])
    return f"SIMULATOR: step() got {res['wrong']} board cells wrong, within x {res['bbox'][0]}-{res['bbox'][2]} " \
           f"y {res['bbox'][1]}-{res['bbox'][3]}: {cells}"


if __name__ == "__main__":
    ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    _worker()
