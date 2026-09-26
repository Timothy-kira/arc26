"""Perception format handed to the model after each acting cell: a 4x image of the frame, the
connected components (objects) and the diff between the animation frames of the last action.
Plain numpy + Pillow; used by the REPL kernel (``objects()``, ``anim()``, ``look()``)."""

from __future__ import annotations

import base64
import io
from collections import Counter, deque
from typing import Optional

import numpy as np

PALETTE = np.array(
    [
        (0xFF, 0xFF, 0xFF), (0xCC, 0xCC, 0xCC), (0x99, 0x99, 0x99), (0x66, 0x66, 0x66),
        (0x33, 0x33, 0x33), (0x00, 0x00, 0x00), (0xE5, 0x3A, 0xA3), (0xFF, 0x7B, 0xCC),
        (0xF9, 0x3C, 0x31), (0x1E, 0x93, 0xFF), (0x88, 0xD8, 0xF1), (0xFF, 0xDC, 0x00),
        (0xFF, 0x85, 0x1B), (0x92, 0x12, 0x31), (0x4F, 0xCC, 0x30), (0xA3, 0x56, 0xD6),
    ],
    dtype=np.uint8,
)
NAMES = ["white", "offwhite", "lightgray", "gray", "darkgray", "black", "magenta", "pink",
         "red", "blue", "lightblue", "yellow", "orange", "maroon", "green", "purple"]


def png4x(grid: np.ndarray, scale: int = 4) -> str:
    """The frame as a PNG (each cell scale x scale pixels), base64."""
    from PIL import Image

    rgb = PALETTE[np.clip(np.asarray(grid), 0, 15).astype(np.int64)]
    img = Image.fromarray(rgb, "RGB").resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def background(grid: np.ndarray) -> int:
    return int(np.bincount(np.asarray(grid).ravel(), minlength=16).argmax())


def objects(grid: np.ndarray, bg: Optional[int] = None, min_size: int = 1) -> list[dict]:
    """4-connected same-colour components except the background:
    {color, size, bbox: (y0, x0, y1, x1), center: (x, y)} sorted by colour then position."""
    g = np.asarray(grid)
    h, w = g.shape
    bg = background(g) if bg is None else bg
    seen = g == bg
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
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and g[yy, xx] == c:
                        seen[yy, xx] = True
                        q.append((yy, xx))
            if len(ys) < min_size:
                continue
            out.append({"color": c, "size": len(ys), "bbox": (min(ys), min(xs), max(ys), max(xs)),
                        "center": ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)})
    return sorted(out, key=lambda o: (o["color"], o["bbox"]))


def objects_text(grid: np.ndarray, limit: int = 18) -> str:
    """Objects grouped by (colour, width, height): one line per kind, rare kinds first."""
    objs = objects(grid)
    bg = background(grid)
    kinds: dict = {}
    for o in objs:
        y0, x0, y1, x1 = o["bbox"]
        kinds.setdefault((o["color"], x1 - x0 + 1, y1 - y0 + 1), []).append(o)
    order = sorted(kinds.items(), key=lambda kv: (len(kv[1]), kv[1][0]["size"]))
    lines = [f"background {NAMES[bg]}#{bg}; {len(objs)} objects in {len(kinds)} kinds (colour WxH: centres x,y)"]
    for (c, w, h), group in order[:limit]:
        cs = " ".join(f"({o['center'][0]},{o['center'][1]})" for o in group[:6]) + (" ..." if len(group) > 6 else "")
        lines.append(f"  {NAMES[c]}#{c} {w}x{h}" + (f" x{len(group)}" if len(group) > 1 else "") + f": {cs}")
    if len(order) > limit:
        lines.append(f"  ... {len(order) - limit} more kinds: objects() lists all")
    return "\n".join(lines)


def anim(frames: list, prev: Optional[np.ndarray] = None) -> list[dict]:
    """Diff between consecutive animation frames of one action (frame 0 against ``prev``)."""
    seq = ([np.asarray(prev)] if prev is not None else []) + [np.asarray(f) for f in frames]
    out = []
    for i in range(1, len(seq)):
        d = seq[i - 1] != seq[i]
        n = int(d.sum())
        if n:
            ys, xs = np.where(d)
            out.append({"frame": i - (0 if prev is not None else -1), "changed": n,
                        "bbox": (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))})
        else:
            out.append({"frame": i, "changed": 0, "bbox": None})
    return out


def anim_text(frames: list, prev: Optional[np.ndarray] = None, limit: int = 10) -> str:
    if len(frames) <= 1:
        return ""
    parts = []
    for a in anim(frames, prev)[:limit]:
        b = a["bbox"]
        parts.append(f"f{a['frame']}: {a['changed']} cells" + (f" in y={b[0]}..{b[2]} x={b[1]}..{b[3]}" if b else ""))
    return f"animation of the last action, {len(frames)} frames: " + "; ".join(parts)


def change_regions(a: np.ndarray, b: np.ndarray, gap: int = 2) -> list[dict]:
    """Changed cells between two frames grouped into regions (cells within ``gap`` of each other),
    largest first: {bbox: (y0, x0, y1, x1), cells, before: {colour: n}, after: {colour: n}}."""
    a, b = np.asarray(a), np.asarray(b)
    d = a != b
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
        out.append({"bbox": (min(ys), min(xs), max(ys), max(xs)), "cells": len(cells),
                    "before": dict(Counter(int(a[y, x]) for y, x in cells).most_common(3)),
                    "after": dict(Counter(int(b[y, x]) for y, x in cells).most_common(3))})
    return sorted(out, key=lambda r: -r["cells"])


def regions_text(a: np.ndarray, b: np.ndarray, limit: int = 6) -> str:
    regs = change_regions(a, b)
    if not regs:
        return "no cells changed"
    lines = [f"changed regions ({len(regs)}; several regions = side effects worth a look):"]
    for r in regs[:limit]:
        y0, x0, y1, x1 = r["bbox"]
        bef = ",".join(f"{NAMES[c]}#{c}" for c in r["before"])
        aft = ",".join(f"{NAMES[c]}#{c}" for c in r["after"])
        lines.append(f"  {r['cells']} cells at x={x0}..{x1} y={y0}..{y1}: {bef} -> {aft}")
    if len(regs) > limit:
        lines.append(f"  ... {len(regs) - limit} smaller regions")
    return "\n".join(lines)


def moves(a: np.ndarray, b: np.ndarray, limit: int = 6) -> list[dict]:
    """Objects that moved between two frames: matched by colour and size, nearest position wins.
    [{color, size: (w, h), from: (x, y), to: (x, y), d: (dx, dy)}]"""
    def kinds(g):
        out: dict = {}
        for o in objects(g):
            y0, x0, y1, x1 = o["bbox"]
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
            res.append({"color": k[0], "size": (k[1], k[2]), "from": p, "to": q, "d": (q[0] - p[0], q[1] - p[1])})
    return sorted(res, key=lambda m: -(m["size"][0] * m["size"][1]))[:limit]


def moves_text(a: np.ndarray, b: np.ndarray) -> str:
    ms = moves(a, b)
    if not ms:
        return ""
    return "moved objects: " + "; ".join(
        f"{NAMES[m['color']]}#{m['color']} {m['size'][0]}x{m['size'][1]} ({m['from'][0]},{m['from'][1]})->"
        f"({m['to'][0]},{m['to'][1]}) d=({m['d'][0]:+d},{m['d'][1]:+d})" for m in ms)
