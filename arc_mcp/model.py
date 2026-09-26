"""Rule model: typed game rules induced from observed transitions (no LLM).

Every transition ``(frame, action) -> frame'`` is reduced to an object-level diff (HUD masked)
and folded into counters; rules are read off those counters with their evidence, so the agent
gets a compact, checkable description of the game instead of raw frames:

  entities   avatar (the object a direction action moves consistently), counters (HUD)
  effects    per action: moves avatar by (dx, dy) / blocked by colours / click on <kind> -> effect /
             undo / no effect
  terrain    colours the avatar has entered (walkable) or stopped at (blocking)
  dangers    what the last action touched before each GAME_OVER
  goals      what happened on each level-up (candidate win conditions, checked on later levels)
  open       untried actions, never-clicked object kinds, unentered colours

Nothing here is game specific; the only knobs are in ``config.ExploreConfig``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .perception import COLOR_NAMES, Obj, background_color, diff, objects

Key = tuple  # action key: (a,) or (6, x, y)


def cname(c: int) -> str:
    return f"{COLOR_NAMES[c]}#{c}"


def kind(o: Obj) -> tuple[int, str]:
    """Objects of one kind share colour and shape."""
    return (o.color, o.shape)


def kind_name(o: Obj) -> str:
    return f"{cname(o.color)} {o.width}x{o.height}"


@dataclass
class Transition:
    level: int
    key: Key
    prev: np.ndarray
    cur: np.ndarray
    level_up: bool
    game_over: bool


@dataclass
class ActionStats:
    n: int = 0
    noop: int = 0
    moves: Counter = field(default_factory=Counter)        # (avatar kind, dx, dy) -> count
    blocked_by: Counter = field(default_factory=Counter)   # colour in front when the avatar did not move
    changes: Counter = field(default_factory=Counter)      # coarse effect labels
    undo: int = 0


@dataclass
class ClickStats:
    n: int = 0
    effects: Counter = field(default_factory=Counter)      # effect label -> count
    example: Optional[tuple[int, int]] = None


class RuleModel:
    def __init__(self) -> None:
        self.actions: dict[int, ActionStats] = defaultdict(ActionStats)
        self.clicks: dict[tuple[int, str], ClickStats] = defaultdict(ClickStats)
        self.kind_names: dict[tuple[int, str], str] = {}
        self.entered: Counter = Counter()      # colours under the avatar's new cells
        self.deaths: list[str] = []
        self.level_ups: list[str] = []
        self.transitions = 0
        self.co_moves: Counter = Counter()     # (kind, kind2): moved together by the same vector
        self.history: list[str] = []           # masked hashes of recent frames (undo detection)
        self.level_start: Optional[np.ndarray] = None  # first frame of the current level

    # ------------------------------------------------------------------ induction

    def avatar(self) -> Optional[tuple[tuple[int, str], dict[int, tuple[int, int]]]]:
        """The object kind moved most consistently by direction actions, with each action's vector."""
        votes: Counter = Counter()
        for a, st in self.actions.items():
            if a in (1, 2, 3, 4, 5, 7):
                for (k, dx, dy), n in st.moves.items():
                    votes[k] += n
        if not votes:
            return None
        k, _ = votes.most_common(1)[0]
        vecs = {}
        for a, st in self.actions.items():
            mv = [(n, dx, dy) for (kk, dx, dy), n in st.moves.items() if kk == k]
            if mv:
                n, dx, dy = max(mv)
                vecs[a] = (dx, dy)
        return k, vecs

    def observe(self, t: Transition, mask: Optional[np.ndarray]) -> None:
        self.transitions += 1
        a = t.key[0]
        d = diff(t.prev, t.cur, mask)
        if a == 6:
            self._observe_click(t, d, mask)
        else:
            st = self.actions[a]
            st.n += 1
            if d.is_noop:
                st.noop += 1
                self._blocked(a, t.prev)
            moved = []
            for o, p in d.moved:
                if mask is not None and mask[o.top:o.bottom + 1, o.left:o.right + 1].all():
                    continue
                k = kind(o)
                self.kind_names[k] = kind_name(o)
                v = (p.left - o.left, p.top - o.top)
                st.moves[(k, *v)] += 1
                moved.append((k, v))
            for k, v in moved:
                for k2, v2 in moved:
                    if k != k2 and v == v2:
                        self.co_moves[(k, k2)] += 1
            av = self.avatar()
            if av:
                for o, p in d.moved:
                    if kind(o) == av[0]:
                        self._entered(t.prev, p, self.parts(av[0]))
            if d.recolored:
                st.changes["recolour"] += 1
            if d.appeared or d.disappeared:
                st.changes["objects appear/vanish"] += 1
            if a == 7 and self._is_undo(t, mask):
                st.undo += 1
        if t.game_over:
            self.deaths.append(self._death_note(t, d))
        if t.level_up:
            self.level_ups.append(self._level_note(t, d))

    def _observe_click(self, t: Transition, d, mask) -> None:
        x, y = int(t.key[1]), int(t.key[2])
        bg = background_color(t.prev)
        hit = next((o for o in objects(t.prev, bg) if o.left <= x <= o.right and o.top <= y <= o.bottom
                    and t.prev[y, x] == o.color), None)
        k = kind(hit) if hit else (bg, "background")
        self.kind_names[k] = kind_name(hit) if hit else f"background {cname(bg)}"
        cs = self.clicks[k]
        cs.n += 1
        cs.example = cs.example or (x, y)
        if d.is_noop:
            cs.effects["nothing"] += 1
            return
        if hit is not None and t.cur[y, x] != t.prev[y, x]:
            cs.effects[f"it turns {cname(int(t.cur[y, x]))}"] += 1
        elif hit is not None:
            cs.effects["it stays, something else changes"] += 1
        else:
            cs.effects["something changes"] += 1
        if d.moved:
            cs.effects["objects move"] += 1
        if len(d.appeared) + len(d.disappeared) + len(d.recolored) > 2:
            cs.effects["several objects change"] += 1

    def _blocked(self, a: int, grid: np.ndarray) -> None:
        av = self.avatar()
        if not av or a not in av[1]:
            return
        k, vecs = av
        dx, dy = vecs[a]
        for o in objects(grid, background_color(grid)):
            if kind(o) != k:
                continue
            t0, l0 = o.top + dy, o.left + dx
            t1, l1 = o.bottom + dy, o.right + dx
            h, w = grid.shape
            if t0 < 0 or l0 < 0 or t1 >= h or l1 >= w:
                self.actions[a].blocked_by["the edge"] += 1
                return
            region = grid[t0:t1 + 1, l0:l1 + 1]
            vals = Counter(int(v) for v in region.ravel() if v != o.color)
            if vals:
                self.actions[a].blocked_by[cname(vals.most_common(1)[0][0])] += 1
            return

    def parts(self, k) -> set:
        """Object kinds that move together with kind ``k`` (a multi-colour avatar)."""
        total = sum(n for (kk, *_), n in (m for st in self.actions.values() for m in st.moves.items()) if kk == k)
        return {k2 for (k1, k2), n in self.co_moves.items() if k1 == k and n >= max(2, 0.5 * total)}

    def avatar_colours(self) -> set:
        av = self.avatar()
        if not av:
            return set()
        return {av[0][0]} | {k2[0] for k2 in self.parts(av[0])}

    def _entered(self, prev: np.ndarray, p: Obj, parts: set = frozenset()) -> None:
        own = {p.color} | {k[0] for k in parts}
        region = prev[p.top:p.bottom + 1, p.left:p.right + 1]
        for v, n in Counter(int(v) for v in region.ravel() if v not in own).items():
            self.entered[v] += n

    def _is_undo(self, t: Transition, mask) -> bool:
        from .perception import frame_hash

        h = frame_hash(t.cur, mask)
        return h in self.history[-3:-1]

    def remember(self, grid: np.ndarray, mask) -> None:
        from .perception import frame_hash

        self.history = (self.history + [frame_hash(grid, mask)])[-8:]

    def _death_note(self, t: Transition, d) -> str:
        a = t.key[0]
        if a == 6:
            x, y = t.key[1], t.key[2]
            return f"after clicking {cname(int(t.prev[y, x]))} at ({x},{y})"
        touched = Counter()
        for o, p in d.moved:
            region = t.prev[p.top:p.bottom + 1, p.left:p.right + 1]
            touched.update(int(v) for v in region.ravel() if v != o.color)
        what = ", ".join(cname(c) for c, _ in touched.most_common(2)) or "nothing visible"
        return f"after ACTION{a} (moved onto: {what})"

    def _level_note(self, t: Transition, d) -> str:
        a = t.key[0]
        act = f"CLICK({t.key[1]},{t.key[2]}) on {cname(int(t.prev[t.key[2], t.key[1]]))}" if a == 6 else f"ACTION{a}"
        touched = Counter()
        for o, p in d.moved:
            region = t.prev[p.top:p.bottom + 1, p.left:p.right + 1]
            touched.update(int(v) for v in region.ravel() if v != o.color)
        parts = [f"level {t.level + 1} won by {act}"]
        if touched:
            parts.append("avatar moved onto " + ", ".join(cname(c) for c, _ in touched.most_common(2)))
        if self.level_start is not None:
            a0 = Counter(int(v) for v in self.level_start.ravel())
            a1 = Counter(int(v) for v in t.prev.ravel())
            gone = [cname(c) for c in a0 if a1.get(c, 0) == 0]
            grew = [cname(c) for c in a1 if a1[c] > 1.5 * a0.get(c, 0) + 4]
            shrank = [cname(c) for c in a0 if 0 < a1.get(c, 0) < 0.5 * a0[c] - 4]
            if gone:
                parts.append("colours removed during the level: " + ", ".join(gone))
            if grew:
                parts.append("colours that spread: " + ", ".join(grew))
            if shrank:
                parts.append("colours that shrank: " + ", ".join(shrank))
        return "; ".join(parts)

    # ------------------------------------------------------------------ output

    def describe(self, available: list[int]) -> str:
        lines = []
        av = self.avatar()
        if av:
            k, vecs = av
            moves = ", ".join(f"A{a}=({dx:+d},{dy:+d})" for a, (dx, dy) in sorted(vecs.items()))
            extra = [self.kind_names.get(k2, str(k2)) for k2 in self.parts(k)]
            lines.append(f"avatar: {self.kind_names.get(k, k)}" + (f" + {', '.join(extra)}" if extra else "")
                         + f"; moves (dx,dy): {moves}")
        for a in sorted(self.actions):
            st = self.actions[a]
            bits = [f"tried {st.n}", f"no effect {st.noop}"]
            if av and a in av[1]:
                n = sum(c for (kk, *_), c in st.moves.items() if kk == av[0])
                bits.append(f"moved avatar {n}")
            if st.blocked_by:
                bits.append("blocked by " + ", ".join(f"{c}({n})" for c, n in st.blocked_by.most_common(3)))
            if st.undo:
                bits.append(f"acted as UNDO {st.undo}x")
            for lab, n in st.changes.most_common(2):
                bits.append(f"{lab} {n}")
            lines.append(f"ACTION{a}: " + ", ".join(bits))
        if self.clicks:
            lines.append("clicks:")
            for k, cs in sorted(self.clicks.items(), key=lambda kv: -kv[1].n)[:10]:
                eff = ", ".join(f"{e} {n}/{cs.n}" for e, n in cs.effects.most_common(3))
                lines.append(f"  on {self.kind_names.get(k, k)} (e.g. {cs.example}): {eff}")
        if self.entered:
            lines.append("avatar entered colours: " + ", ".join(f"{cname(c)}" for c, _ in self.entered.most_common(6)))
        if self.deaths:
            lines.append(f"GAME_OVER {len(self.deaths)}x: " + "; ".join(Counter(self.deaths).most_common(1)[0][0:1]) +
                         (f" (+{len(set(self.deaths)) - 1} other causes)" if len(set(self.deaths)) > 1 else ""))
        for note in self.level_ups[-3:]:
            lines.append("won: " + note)
        untried = [a for a in available if a and a not in self.actions and not (a == 6 and self.clicks)]
        if untried:
            lines.append("never tried: " + ", ".join(f"ACTION{a}" for a in untried))
        return "\n".join(lines) if lines else "nothing learned yet"

    def unclicked_kinds(self, grid: np.ndarray, mask: Optional[np.ndarray] = None) -> list[Obj]:
        """One object per kind on screen that has never been clicked."""
        seen: set = set()
        out = []
        for o in objects(grid, background_color(grid)):
            k = kind(o)
            if k in self.clicks or k in seen:
                continue
            if mask is not None and mask[o.top:o.bottom + 1, o.left:o.right + 1].all():
                continue
            seen.add(k)
            out.append(o)
        return out
