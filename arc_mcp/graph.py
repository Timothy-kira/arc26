"""Deterministic, LLM-free exploration: a per-level state graph over masked frame hashes.

Most ARC-AGI-3 games are deterministic, so ``(state, action) -> next state`` edges
learned once can be replayed. The explorer tries every untried ``(state, action)``
pair, walking known edges (BFS) to reach the nearest frontier, and never repeats a
no-op or a known-fatal action. HUD strips (step/energy bars along the frame edge
that change on every action) are detected and masked so that states repeat.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Hashable, Optional

import numpy as np

from .perception import frame_hash, objects, background_color
from .config import DEFAULT, ExploreConfig

ActionKey = tuple  # (action_id,) or (6, x, y)


def key_to_call(key: ActionKey) -> tuple[int, Optional[int], Optional[int]]:
    if key[0] == 6:
        return 6, int(key[1]), int(key[2])
    return int(key[0]), None, None


def key_name(key: ActionKey) -> str:
    return f"CLICK({key[1]},{key[2]})" if key[0] == 6 else f"ACTION{key[0]}"


class HudDetector:
    """Masks full rows/columns in the edge band that change on most transitions."""

    def __init__(self, cfg: ExploreConfig = DEFAULT, shape: tuple[int, int] = (64, 64)) -> None:
        self.cfg = cfg
        self.shape = shape
        self.transitions = 0
        h, w = shape
        self.row_changes = np.zeros(h, dtype=np.int32)
        self.col_changes = np.zeros(w, dtype=np.int32)
        self._mask = np.zeros(shape, dtype=bool)

    def observe(self, prev: np.ndarray, cur: np.ndarray) -> None:
        if prev.shape != self.shape or cur.shape != self.shape:
            return
        changed = prev != cur
        if not changed.any():
            return
        self.transitions += 1
        self.row_changes += changed.any(axis=1)
        self.col_changes += changed.any(axis=0)
        self._recompute()

    def _recompute(self) -> None:
        cfg = self.cfg
        if self.transitions < cfg.hud_min_transitions:
            return
        h, w = self.shape
        b = cfg.hud_band
        mask = np.zeros(self.shape, dtype=bool)
        rate_r = self.row_changes / self.transitions
        rate_c = self.col_changes / self.transitions
        edge_rows = list(range(b)) + list(range(h - b, h))
        edge_cols = list(range(b)) + list(range(w - b, w))
        interior_rate = max(
            float(np.median(rate_r[b : h - b])) if h > 2 * b else 0.0,
            float(np.median(rate_c[b : w - b])) if w > 2 * b else 0.0,
        )
        thresh = max(cfg.hud_change_rate, interior_rate + 0.25)
        for r in edge_rows:
            if rate_r[r] >= thresh:
                mask[r, :] = True
        for c in edge_cols:
            if rate_c[c] >= thresh:
                mask[:, c] = True
        self._mask = mask

    @property
    def mask(self) -> np.ndarray:
        return self._mask


@dataclass
class Edge:
    dst: Optional[str]
    noop: bool = False
    game_over: bool = False
    level_up: bool = False
    count: int = 1


@dataclass
class Node:
    key: str
    grid: np.ndarray
    candidates: list[ActionKey]
    edges: dict[ActionKey, Edge] = field(default_factory=dict)

    def untried(self, banned: set[ActionKey] = frozenset()) -> list[ActionKey]:
        return [a for a in self.candidates if a not in self.edges and a not in banned]


def click_targets(grid: np.ndarray, mask: Optional[np.ndarray], cfg: ExploreConfig = DEFAULT) -> list[ActionKey]:
    """One click per distinct object (centre cell), rare colours and small objects first."""
    bg = background_color(grid)
    objs = objects(grid, bg)
    if mask is not None and mask.any():
        objs = [o for o in objs if not mask[o.center[1], o.center[0]]]
    colour_count: dict[int, int] = {}
    for o in objs:
        colour_count[o.color] = colour_count.get(o.color, 0) + 1
    objs.sort(key=lambda o: (o.size > cfg.click_large_object_px, colour_count[o.color], o.size))
    seen: set[tuple[int, int]] = set()
    out: list[ActionKey] = []
    for o in objs:
        x, y = o.center
        if grid[y, x] != o.color:
            # centre of a hollow/concave shape: click a real cell of the object instead
            ys, xs = np.where(grid[o.top : o.bottom + 1, o.left : o.right + 1] == o.color)
            if len(xs):
                i = len(xs) // 2
                x, y = o.left + int(xs[i]), o.top + int(ys[i])
        if (x, y) in seen:
            continue
        seen.add((x, y))
        out.append((6, x, y))
        if len(out) >= cfg.max_click_targets:
            break
    return out


class StateGraph:
    """Exploration graph for one level of one game."""

    def __init__(self, available: list[int], cfg: ExploreConfig = DEFAULT, hud: Optional[HudDetector] = None) -> None:
        self.cfg = cfg
        self.available = [a for a in available if a != 0]
        self.hud = hud or HudDetector(cfg)
        self.nodes: dict[str, Node] = {}
        self.start: Optional[str] = None
        self.win_edge: Optional[tuple[str, ActionKey]] = None
        self.banned: set[ActionKey] = set()
        self.priority: dict[int, float] = {}

    def key(self, grid: np.ndarray) -> str:
        return frame_hash(grid, self.hud.mask if self.hud.mask.any() else None)

    def _candidates(self, grid: np.ndarray) -> list[ActionKey]:
        simple = [a for a in self.available if a not in (6,)]
        simple.sort(key=lambda a: -self.priority.get(a, self.cfg.undo_priority if a == 7 else 1.0))
        cands: list[ActionKey] = [(a,) for a in simple]
        if 6 in self.available:
            cands += click_targets(grid, self.hud.mask, self.cfg)
        return cands

    def visit(self, grid: np.ndarray) -> Node:
        k = self.key(grid)
        node = self.nodes.get(k)
        if node is None:
            node = Node(k, grid.copy(), self._candidates(grid))
            self.nodes[k] = node
        if self.start is None:
            self.start = k
        return node

    def rehash(self) -> None:
        """Re-key nodes after the HUD mask changes so previously distinct states merge."""
        old = self.nodes
        remap = {k: self.key(n.grid) for k, n in old.items()}
        self.nodes = {}
        for k, n in old.items():
            nk = remap[k]
            if nk in self.nodes:
                tgt = self.nodes[nk]
                for a, e in n.edges.items():
                    tgt.edges.setdefault(a, e)
            else:
                n.key = nk
                self.nodes[nk] = n
        for n in self.nodes.values():
            for e in n.edges.values():
                if e.dst is not None:
                    e.dst = remap.get(e.dst, e.dst)
        if self.start:
            self.start = remap.get(self.start, self.start)
        if self.win_edge:
            self.win_edge = (remap.get(self.win_edge[0], self.win_edge[0]), self.win_edge[1])

    def record(
        self,
        prev_grid: np.ndarray,
        action: ActionKey,
        cur_grid: np.ndarray,
        game_over: bool = False,
        level_up: bool = False,
    ) -> Edge:
        before = self.hud.mask.copy()
        self.hud.observe(prev_grid, cur_grid)
        if not np.array_equal(before, self.hud.mask):
            self.rehash()
        src = self.visit(prev_grid)
        noop = self.key(prev_grid) == self.key(cur_grid) and not level_up and not game_over
        dst = None if (game_over or level_up) else self.visit(cur_grid).key
        edge = src.edges.get(action)
        if edge is None:
            edge = Edge(dst, noop, game_over, level_up)
            src.edges[action] = edge
        else:
            edge.count += 1
            edge.dst, edge.noop, edge.game_over, edge.level_up = dst, noop, game_over, level_up
        if level_up:
            self.win_edge = (src.key, action)
        return edge

    def _neighbors(self, k: str):
        node = self.nodes.get(k)
        if not node:
            return
        for a, e in node.edges.items():
            if e.dst and not e.noop and not e.game_over and e.dst in self.nodes:
                yield a, e.dst

    def path(self, src: str, goal) -> Optional[list[ActionKey]]:
        """BFS over safe known edges; ``goal`` is a predicate on node keys."""
        if src not in self.nodes:
            return None
        prev: dict[str, tuple[str, ActionKey]] = {}
        q = deque([src])
        seen = {src}
        while q:
            k = q.popleft()
            if goal(k):
                out: list[ActionKey] = []
                while k != src:
                    pk, a = prev[k]
                    out.append(a)
                    k = pk
                return out[::-1]
            for a, nk in self._neighbors(k):
                if nk not in seen:
                    seen.add(nk)
                    prev[nk] = (k, a)
                    q.append(nk)
        return None

    def plan_to_frontier(self, cur: str) -> Optional[list[ActionKey]]:
        """Shortest known path to a state with an untried action, plus that action."""
        banned = self.banned

        def has_untried(k: str) -> bool:
            return bool(self.nodes[k].untried(banned))

        p = self.path(cur, has_untried)
        if p is None:
            return None
        end = cur
        for a in p:
            end = self.nodes[end].edges[a].dst  # type: ignore[assignment]
        nxt = self.nodes[end].untried(banned)[0]
        return (p + [nxt])[: self.cfg.max_path_len]

    def winning_path(self) -> Optional[list[ActionKey]]:
        """Shortest known start -> win path (the efficient replay for this level)."""
        if not self.win_edge or not self.start:
            return None
        wk, wa = self.win_edge
        p = self.path(self.start, lambda k: k == wk)
        return None if p is None else p + [wa]

    def stats(self) -> dict[str, Hashable]:
        edges = [e for n in self.nodes.values() for e in n.edges.values()]
        return {
            "states": len(self.nodes),
            "edges": len(edges),
            "noops": sum(e.noop for e in edges),
            "deaths": sum(e.game_over for e in edges),
            "frontier": sum(bool(n.untried(self.banned)) for n in self.nodes.values()),
            "hud_masked_cells": int(self.hud.mask.sum()),
        }
