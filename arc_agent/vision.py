"""What the agent sees of a frame and of the change one action made.

ARC-AGI-3 frames are 64x64 grids of 16 colours and one action usually changes only a few cells,
so the observation is built around the diff: which cells changed, grouped into regions, which
objects moved and by how much, and which changes belong to the HUD (a bar or counter that
changes on almost every action) rather than to the board.
"""

from __future__ import annotations

import base64
import io
from collections import Counter, deque
from typing import Optional

import numpy as np

HEX = "0123456789abcdef"
NAMES = ["white", "offwhite", "lightgray", "gray", "darkgray", "black", "magenta", "pink",
         "red", "blue", "lightblue", "yellow", "orange", "maroon", "green", "purple"]
PALETTE = np.array([
    (0xFF, 0xFF, 0xFF), (0xCC, 0xCC, 0xCC), (0x99, 0x99, 0x99), (0x66, 0x66, 0x66),
    (0x33, 0x33, 0x33), (0x00, 0x00, 0x00), (0xE5, 0x3A, 0xA3), (0xFF, 0x7B, 0xCC),
    (0xF9, 0x3C, 0x31), (0x1E, 0x93, 0xFF), (0x88, 0xD8, 0xF1), (0xFF, 0xDC, 0x00),
    (0xFF, 0x85, 0x1B), (0x92, 0x12, 0x31), (0x4F, 0xCC, 0x30), (0xA3, 0x56, 0xD6),
], dtype=np.uint8)


def cname(c: int) -> str:
    return f"{NAMES[c]}#{c}"


def png(grid: np.ndarray, scale: int = 4) -> str:
    """The frame as a base64 PNG, each cell ``scale`` x ``scale`` pixels."""
    from PIL import Image

    rgb = PALETTE[np.clip(np.asarray(grid), 0, 15).astype(np.int64)]
    img = Image.fromarray(rgb, "RGB").resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def hex_grid(grid: np.ndarray) -> str:
    """One line per row, one hex digit per cell, with column and row labels."""
    g = np.asarray(grid)
    head = "    " + "".join(str(x // 10) for x in range(g.shape[1])) + "\n    " + "".join(str(x % 10) for x in range(g.shape[1]))
    return head + "\n" + "\n".join(f"{y:2d}  " + "".join(HEX[int(v)] for v in row) for y, row in enumerate(g))


def background(grid: np.ndarray) -> int:
    return int(np.bincount(np.asarray(grid).ravel(), minlength=16).argmax())


def objects(grid: np.ndarray) -> list[dict]:
    """4-connected same-colour components other than the background: colour, cell count, bbox
    (x0, y0, x1, y1)."""
    g = np.asarray(grid)
    h, w = g.shape
    seen = g == background(g)
    out = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0, x0]:
                continue
            c = int(g[y0, x0])
            q = deque([(y0, x0)])
            seen[y0, x0] = True
            ys, xs = [], []
            while q:
                y, x = q.popleft()
                ys.append(y)
                xs.append(x)
                for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and g[yy, xx] == c:
                        seen[yy, xx] = True
                        q.append((yy, xx))
            out.append({"color": c, "size": len(ys), "bbox": (min(xs), min(ys), max(xs), max(ys))})
    return out


def objects_text(grid: np.ndarray, limit: int = 24) -> str:
    """Objects grouped by colour and shape size, rarest first (single objects usually matter most)."""
    kinds: dict = {}
    for o in objects(grid):
        x0, y0, x1, y1 = o["bbox"]
        kinds.setdefault((o["color"], x1 - x0 + 1, y1 - y0 + 1), []).append(o)
    order = sorted(kinds.items(), key=lambda kv: (len(kv[1]), -kv[1][0]["size"]))
    lines = [f"background {cname(background(grid))}; {sum(len(v) for v in kinds.values())} objects "
             f"(colour WxH: top-left x,y)"]
    for (c, w, h), group in order[:limit]:
        pos = " ".join(f"({o['bbox'][0]},{o['bbox'][1]})" for o in group[:8]) + (" ..." if len(group) > 8 else "")
        lines.append(f"  {cname(c)} {w}x{h}" + (f" x{len(group)}" if len(group) > 1 else "") + f": {pos}")
    if len(order) > limit:
        lines.append(f"  ... {len(order) - limit} more kinds")
    return "\n".join(lines)


def regions(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None, gap: int = 2) -> list[dict]:
    """Changed cells between two frames grouped into regions (cells within ``gap`` of each other),
    largest first: bbox (x0, y0, x1, y1), cell count, dominant colours before and after."""
    d = np.asarray(a) != np.asarray(b)
    if mask is not None:
        d &= ~mask
    if not d.any():
        return []
    h, w = d.shape
    seen = np.zeros_like(d)
    out = []
    for y0, x0 in zip(*np.where(d)):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        cells = []
        while q:
            y, x = q.popleft()
            cells.append((y, x))
            for yy in range(max(0, y - gap), min(h, y + gap + 1)):
                for xx in range(max(0, x - gap), min(w, x + gap + 1)):
                    if d[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        q.append((yy, xx))
        ys, xs = [c[0] for c in cells], [c[1] for c in cells]
        out.append({"bbox": (min(xs), min(ys), max(xs), max(ys)), "cells": len(cells),
                    "before": Counter(int(a[y, x]) for y, x in cells).most_common(2),
                    "after": Counter(int(b[y, x]) for y, x in cells).most_common(2)})
    return sorted(out, key=lambda r: -r["cells"])


def moves(a: np.ndarray, b: np.ndarray, limit: int = 6) -> list[dict]:
    """Objects that moved: matched by colour and shape size, nearest new position wins."""
    def kinds(g):
        out: dict = {}
        for o in objects(g):
            x0, y0, x1, y1 = o["bbox"]
            out.setdefault((o["color"], x1 - x0 + 1, y1 - y0 + 1), []).append((x0, y0))
        return out

    ka, kb = kinds(a), kinds(b)
    res = []
    for k, pa in ka.items():
        pb = kb.get(k, [])
        gone = [p for p in pa if p not in pb]
        new = [p for p in pb if p not in pa]
        for p in gone:
            if not new:
                break
            q = min(new, key=lambda q: abs(q[0] - p[0]) + abs(q[1] - p[1]))
            new.remove(q)
            res.append({"color": k[0], "w": k[1], "h": k[2], "from": p, "to": q, "dx": q[0] - p[0], "dy": q[1] - p[1]})
    return sorted(res, key=lambda m: -(m["w"] * m["h"]))[:limit]


class HudTracker:
    """The HUD: a bar that changes on almost every action of a level (usually the move budget). It is
    a thin band (rows or columns) where a small change lands on 3+ actions at different positions: a
    bar that shrinks one cell at a time changes each cell only once, and cells that merely change
    often are the avatar's path, not the HUD. Its changes are reported apart from the board so they
    do not read as the action's effect."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.hits: dict[tuple, set] = {}

    def update(self, a: np.ndarray, b: np.ndarray) -> None:
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != (64, 64) or b.shape != (64, 64):
            return
        self.steps += 1
        for r in regions(a, b):
            if r["cells"] > 8:
                continue
            x0, y0, x1, y1 = r["bbox"]
            if y1 - y0 <= 2:
                self.hits.setdefault(("row", y0, y1), set()).add((self.steps, x0))
            if x1 - x0 <= 2:
                self.hits.setdefault(("col", x0, x1), set()).add((self.steps, y0))

    def bands(self) -> list[tuple]:
        return [k for k, v in self.hits.items() if len({s for s, _ in v}) >= 3 and len({p for _, p in v}) >= 3]

    def mask(self) -> np.ndarray:
        m = np.zeros((64, 64), dtype=bool)
        for kind, p0, p1 in self.bands():
            if kind == "row":
                m[p0:p1 + 1, :] = True
            else:
                m[:, p0:p1 + 1] = True
        return m


def diff_text(a: np.ndarray, b: np.ndarray, hud: Optional[np.ndarray] = None, limit: int = 6) -> tuple[str, dict]:
    """Human-readable diff of one action and the facts used to check a prediction."""
    a, b = np.asarray(a), np.asarray(b)
    hud = hud if hud is not None else np.zeros(a.shape, dtype=bool)
    total = int((a != b).sum())
    board = int(((a != b) & ~hud).sum())
    regs = regions(a, b, hud)
    mv = moves(a, b)
    facts = {"changed": total, "board_changed": board, "hud_changed": total - board,
             "moves": [{"color": m["color"], "dx": m["dx"], "dy": m["dy"]} for m in mv]}
    if total == 0:
        return "no cell changed", facts
    lines = [f"{total} cells changed ({board} on the board, {total - board} in the HUD)"]
    for r in regs[:limit]:
        x0, y0, x1, y1 = r["bbox"]
        bef = ",".join(cname(c) for c, _ in r["before"])
        aft = ",".join(cname(c) for c, _ in r["after"])
        lines.append(f"  region x={x0}..{x1} y={y0}..{y1}: {r['cells']} cells {bef} -> {aft}")
    if len(regs) > limit:
        lines.append(f"  ... {len(regs) - limit} smaller regions")
    for m in mv:
        lines.append(f"  moved {cname(m['color'])} {m['w']}x{m['h']} ({m['from'][0]},{m['from'][1]}) -> "
                     f"({m['to'][0]},{m['to'][1]}) d=({m['dx']:+d},{m['dy']:+d})")
    return "\n".join(lines), facts


def anim_text(frames: list[np.ndarray], prev: np.ndarray) -> str:
    """How many cells changed between consecutive animation frames of one action."""
    if len(frames) <= 1:
        return ""
    seq = [prev] + list(frames)
    parts = []
    for i in range(1, len(seq)):
        n = int((seq[i - 1] != seq[i]).sum())
        parts.append(str(n))
    return f"animation: {len(frames)} frames, cells changed per frame: {' '.join(parts)}"
