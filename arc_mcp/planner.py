"""Model-based planning: search with the learned rules, execute, verify, repair.

Goals (the JSON the agent passes to ``arc_plan``):
  {"reach": {"color": 11}}          move the avatar until it overlaps a cell of that colour
  {"reach": {"x": 30, "y": 12}}     ... until its box covers that cell
  {"click_all": {"color": 9}}       click every object of that colour once
  {"avoid": [8, 12]}                (optional, with reach) colours never to step on

``reach`` runs BFS over avatar positions using each direction action's learned vector. Cells
of colours the avatar was blocked by (and never entered) are walls; colours touched right
before a GAME_OVER are avoided. After each step the real avatar position is compared with the
prediction; a mismatch is recorded as a new wall and the plan is recomputed.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np
from arcengine import GameState

from .model import cname, kind
from .perception import background_color, objects


def avatar_box(mem, grid: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Bounding box of the avatar and the parts that move with it (touching or within 1 cell)."""
    av = mem.model.avatar()
    if not av:
        return None
    k = av[0]
    parts = mem.model.parts(k)
    objs = objects(grid, background_color(grid))
    main = next((o for o in objs if kind(o) == k), None)
    if main is None:
        return None
    t, l, b, r = main.top, main.left, main.bottom, main.right
    for o in objs:
        if kind(o) in parts and o.top <= b + 2 and o.bottom >= t - 2 and o.left <= r + 2 and o.right >= l - 2:
            t, l, b, r = min(t, o.top), min(l, o.left), max(b, o.bottom), max(r, o.right)
    return t, l, b, r


def colour_id(name: str) -> Optional[int]:
    try:
        return int(name.rsplit("#", 1)[1])
    except (IndexError, ValueError):
        return None


def walls_and_hazards(mem, extra_avoid: list[int]) -> tuple[set[int], set[int]]:
    walls: set[int] = set()
    for st in mem.model.actions.values():
        for name, n in st.blocked_by.items():
            c = colour_id(name)
            if c is not None and mem.model.entered.get(c, 0) == 0:
                walls.add(c)
    walls |= mem.blocked_colours
    hazards = set(extra_avoid)
    for note in mem.model.deaths:
        if "moved onto:" in note:
            for part in note.split("moved onto:")[1].strip(" )").split(","):
                c = colour_id(part.strip())
                if c is not None:
                    hazards.add(c)
    return walls, hazards


def bfs(grid, box, vecs, goal, walls, hazards, avatar_colours, max_nodes=20000):
    h, w = grid.shape
    top, left, bottom, right = box
    bh, bw = bottom - top, right - left

    def cells(t, l):
        return grid[t:t + bh + 1, l:l + bw + 1]

    def ok(t, l):
        if t < 0 or l < 0 or t + bh >= h or l + bw >= w:
            return False
        vals = set(int(v) for v in np.unique(cells(t, l))) - avatar_colours
        return not (vals & walls) and not (vals & hazards)

    start = (top, left)
    prev = {start: None}
    q = deque([start])
    while q and len(prev) < max_nodes:
        cur = q.popleft()
        if goal(cur[0], cur[1], bh, bw) and cur != start:
            path = []
            while prev[cur] is not None:
                cur, a = prev[cur]
                path.append(a)
            return list(reversed(path))
        for a, (dx, dy) in vecs.items():
            nxt = (cur[0] + dy, cur[1] + dx)
            if nxt not in prev and ok(*nxt):
                prev[nxt] = (cur, a)
                q.append(nxt)
    return None


def plan(mem, goal: dict[str, Any], max_actions: int = 60) -> str:
    if "click_all" in goal:
        return click_all(mem, int(goal["click_all"]["color"]), max_actions)
    if "reach" not in goal:
        return "unknown goal: use {\"reach\": {\"color\": c}} / {\"reach\": {\"x\":..,\"y\":..}} / {\"click_all\": {\"color\": c}}"
    av = mem.model.avatar()
    if not av:
        return "no avatar known yet: explore the direction actions first (arc_explore)"
    k, vecs = av
    vecs = {a: v for a, v in vecs.items() if v != (0, 0)}
    tgt = goal["reach"]
    log, used, replans = [], 0, 0
    while used < max_actions and mem.obs.state == GameState.NOT_FINISHED:
        grid = mem.obs.grid
        box = avatar_box(mem, grid)
        if box is None:
            return "avatar not visible on screen"
        if "color" in tgt:
            c = int(tgt["color"])
            goal_fn = lambda t, l, bh, bw: bool((grid[t:t + bh + 1, l:l + bw + 1] == c).any())
        else:
            gx, gy = int(tgt["x"]), int(tgt["y"])
            goal_fn = lambda t, l, bh, bw: t <= gy <= t + bh and l <= gx <= l + bw
        walls, hazards = walls_and_hazards(mem, list(goal.get("avoid", [])))
        path = bfs(grid, box, vecs, goal_fn, walls, hazards, mem.model.avatar_colours())
        if not path:
            log.append(f"no path (walls {sorted(walls)}, avoided {sorted(hazards)})")
            break
        for a in path:
            dx, dy = vecs[a]
            want = (box[0] + dy, box[1] + dx)
            obs, note = mem.step((a,), "plan")
            used += 1
            if note:
                log.append(note)
            if obs.state != GameState.NOT_FINISHED or "LEVEL" in note:
                return summarise(mem, log, used, replans, done=True)
            nb = avatar_box(mem, obs.grid)
            if nb is None or (nb[0], nb[1]) != want:
                # prediction failed: whatever is in front is treated as a wall from now on
                region = grid[max(0, want[0]):want[0] + box[2] - box[0] + 1, max(0, want[1]):want[1] + box[3] - box[1] + 1]
                front = set(int(v) for v in np.unique(region)) - mem.model.avatar_colours()
                if nb is not None and (nb[0], nb[1]) == (box[0], box[1]):
                    mem.blocked_colours |= front
                    log.append(f"ACTION{a} blocked (in front: {', '.join(cname(c) for c in sorted(front)) or 'background'})")
                else:
                    log.append(f"ACTION{a} did something unexpected; replanning")
                replans += 1
                break
            box = nb
            if used >= max_actions:
                break
        else:
            if goal_fn(box[0], box[1], box[2] - box[0], box[3] - box[1]):
                log.append("goal reached")
                break
        if replans > 8:
            log.append("too many surprises; stopping")
            break
    return summarise(mem, log, used, replans)


def click_all(mem, colour: int, max_actions: int) -> str:
    log, used = [], 0
    for _ in range(max_actions):
        grid = mem.obs.grid
        todo = [o for o in objects(grid, background_color(grid)) if o.color == colour and (o.top, o.left) not in mem.clicked]
        if not todo or mem.obs.state != GameState.NOT_FINISHED:
            break
        o = min(todo, key=lambda o: (o.top, o.left))
        ys, xs = np.where(grid[o.top:o.bottom + 1, o.left:o.right + 1] == colour)
        x, y = o.left + int(xs[len(xs) // 2]), o.top + int(ys[len(ys) // 2])
        mem.clicked.add((o.top, o.left))
        _, note = mem.step((6, x, y), "plan")
        used += 1
        if note:
            log.append(note)
            if "LEVEL" in note:
                break
    return summarise(mem, log or [f"clicked {used} {cname(colour)} objects"], used, 0)


def summarise(mem, log: list[str], used: int, replans: int, done: bool = False) -> str:
    return "\n".join([f"plan used {used} actions, {replans} replans"] + log[-8:])
