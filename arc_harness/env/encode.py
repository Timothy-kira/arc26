"""Frame encoding: grids -> objects, diffs, hashes and compact text for the LLM.

A raw 64x64 grid is ~4k tokens; the LLM instead sees a background colour, a list
of connected-component objects and an object-level diff against the previous
frame. A PNG rendering is available for the VLM path.
"""

from __future__ import annotations

import base64
import hashlib
import io
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

COLOR_NAMES = [
    "white", "offwhite", "lightgray", "gray", "darkgray", "black", "magenta", "pink",
    "red", "blue", "lightblue", "yellow", "orange", "maroon", "green", "purple",
]

PALETTE = np.array(
    [
        (0xFF, 0xFF, 0xFF), (0xCC, 0xCC, 0xCC), (0x99, 0x99, 0x99), (0x66, 0x66, 0x66),
        (0x33, 0x33, 0x33), (0x00, 0x00, 0x00), (0xE5, 0x3A, 0xA3), (0xFF, 0x7B, 0xCC),
        (0xF9, 0x3C, 0x31), (0x1E, 0x93, 0xFF), (0x88, 0xD8, 0xF1), (0xFF, 0xDC, 0x00),
        (0xFF, 0x85, 0x1B), (0x92, 0x12, 0x31), (0x4F, 0xCC, 0x30), (0xA3, 0x56, 0xD6),
    ],
    dtype=np.uint8,
)


LARGE_REGION = 300


@dataclass(frozen=True)
class Obj:
    color: int
    size: int
    top: int
    left: int
    bottom: int
    right: int
    shape: str

    @property
    def center(self) -> tuple[int, int]:
        """(x, y) in engine click coordinates: x = column, y = row."""
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    def describe(self) -> str:
        x, y = self.center
        return (
            f"{COLOR_NAMES[self.color]}#{self.color} {self.width}x{self.height} "
            f"at x={self.left}..{self.right} y={self.top}..{self.bottom} "
            f"(center {x},{y}, {self.size}px)"
        )


def frame_hash(grid: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
    g = grid if mask is None else np.where(mask, -1, grid)
    return hashlib.blake2b(np.ascontiguousarray(g, dtype=np.int8).tobytes(), digest_size=8).hexdigest()


def background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _shape_key(mask: np.ndarray) -> str:
    return hashlib.blake2b(
        np.packbits(mask).tobytes() + bytes(mask.shape), digest_size=6
    ).hexdigest()


def objects(grid: np.ndarray, background: Optional[int] = None, min_size: int = 1) -> list[Obj]:
    """4-connected same-colour components, excluding the background colour."""
    h, w = grid.shape
    bg = background_color(grid) if background is None else background
    seen = np.zeros((h, w), dtype=bool)
    seen[grid == bg] = True
    out: list[Obj] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0]:
                continue
            color = int(grid[r0, c0])
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and grid[rr, cc] == color:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            if len(cells) < min_size:
                continue
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            top, bottom, left, right = min(rs), max(rs), min(cs), max(cs)
            m = np.zeros((bottom - top + 1, right - left + 1), dtype=bool)
            for r, c in cells:
                m[r - top, c - left] = True
            out.append(Obj(color, len(cells), top, left, bottom, right, _shape_key(m)))
    return out


@dataclass
class Diff:
    changed_cells: int
    bbox: Optional[tuple[int, int, int, int]]  # top, left, bottom, right
    moved: list[tuple[Obj, Obj]] = field(default_factory=list)
    appeared: list[Obj] = field(default_factory=list)
    disappeared: list[Obj] = field(default_factory=list)
    recolored: list[tuple[Obj, Obj]] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return self.changed_cells == 0

    def describe(self, limit: int = 8) -> str:
        if self.is_noop:
            return "no change"
        parts = [f"{self.changed_cells} cells changed in box y={self.bbox[0]}..{self.bbox[2]} x={self.bbox[1]}..{self.bbox[3]}"]
        for a, b in self.moved[:limit]:
            parts.append(
                f"moved {COLOR_NAMES[a.color]} {a.width}x{a.height} "
                f"({a.center[0]},{a.center[1]})->({b.center[0]},{b.center[1]})"
            )
        for a, b in self.recolored[:limit]:
            parts.append(f"recolored {COLOR_NAMES[a.color]}->{COLOR_NAMES[b.color]} at {a.center}")
        big = [o for o in self.appeared + self.disappeared if o.size > LARGE_REGION]
        for o in [o for o in self.appeared if o.size <= LARGE_REGION][:limit]:
            parts.append(f"appeared {o.describe()}")
        for o in [o for o in self.disappeared if o.size <= LARGE_REGION][:limit]:
            parts.append(f"disappeared {o.describe()}")
        if big:
            parts.append(f"{len(big)} large region(s) reshaped")
        return "; ".join(parts)


def diff(prev: np.ndarray, cur: np.ndarray, mask: Optional[np.ndarray] = None) -> Diff:
    changed = prev != cur
    if mask is not None:
        changed &= ~mask
    n = int(changed.sum())
    if n == 0:
        return Diff(0, None)
    rows = np.where(changed.any(axis=1))[0]
    cols = np.where(changed.any(axis=0))[0]
    bbox = (int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1]))
    bg = background_color(cur)
    a = objects(prev, bg)
    b = objects(cur, bg)
    sa, sb = set(a), set(b)
    gone = [o for o in a if o not in sb]
    new = [o for o in b if o not in sa]
    d = Diff(n, bbox)
    used_new: set[int] = set()
    for o in gone:
        match = next(
            (i for i, p in enumerate(new) if i not in used_new and p.shape == o.shape and p.color == o.color),
            None,
        )
        if match is not None:
            used_new.add(match)
            d.moved.append((o, new[match]))
            continue
        match = next(
            (
                i
                for i, p in enumerate(new)
                if i not in used_new and p.shape == o.shape and (p.top, p.left) == (o.top, o.left)
            ),
            None,
        )
        if match is not None:
            used_new.add(match)
            d.recolored.append((o, new[match]))
            continue
        d.disappeared.append(o)
    d.appeared = [p for i, p in enumerate(new) if i not in used_new]
    return d


def describe_scene(grid: np.ndarray, max_objects: int = 40, mask: Optional[np.ndarray] = None) -> str:
    bg = background_color(grid)
    objs = objects(grid, bg)
    if mask is not None:
        objs = [o for o in objs if not mask[o.top : o.bottom + 1, o.left : o.right + 1].all()]
    counts = Counter(o.color for o in objs)
    objs = sorted(objs, key=lambda o: (counts[o.color], o.size))
    lines = [f"background={COLOR_NAMES[bg]}#{bg}; {len(objs)} objects"]
    for o in objs[:max_objects]:
        lines.append("- " + o.describe())
    if len(objs) > max_objects:
        lines.append(f"- ... {len(objs) - max_objects} more (mostly repeated colours)")
    return "\n".join(lines)


def grid_text(grid: np.ndarray, step: int = 1) -> str:
    """Hex-digit rows (0-f); ``step>1`` subsamples for a cheaper overview."""
    g = grid[::step, ::step]
    return "\n".join("".join("0123456789abcdef"[int(v)] for v in row) for row in g)


def to_png_b64(grid: np.ndarray, scale: int = 4) -> str:
    from PIL import Image

    rgb = PALETTE[np.clip(grid, 0, 15).astype(np.int64)]
    img = Image.fromarray(rgb, "RGB").resize(
        (grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def iter_centers(objs: Iterable[Obj]) -> list[tuple[int, int]]:
    return [o.center for o in objs]
