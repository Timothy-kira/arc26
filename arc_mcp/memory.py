"""Everything the MCP server remembers about one game: state, step ledger, rules, todo.

``GameMemory`` wraps the ``Session`` (scorecard accounting, RESET rules), the explorer's
per-level state graphs, the ``RuleModel`` and a todo list, and writes ``ledger.json`` after
every action so a restarted agent (new exec round, compaction) resumes with full memory.
Every action is attributed to its source (``agent`` / ``explore`` / ``plan``) so the ledger
shows where a level's steps went.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from arcengine import GameState

from .config import DEFAULT, ExploreConfig
from .explorer import Explorer, RESET
from .graph import ActionKey, HudDetector, key_name, key_to_call
from .model import RuleModel, Transition, cname, kind_name
from .session import Obs, Session


@dataclass
class TodoItem:
    id: int
    text: str
    source: str = "auto"      # auto | agent
    cost: int = 1             # estimated actions
    done: bool = False


@dataclass
class GameMemory:
    session: Session
    cfg: ExploreConfig = field(default_factory=ExploreConfig)
    path: Optional[str] = None
    baseline: Optional[list[int]] = None           # human per-level actions (offline only)
    explorer: Explorer = field(init=False)
    model: RuleModel = field(default_factory=RuleModel)
    by_source: Counter = field(default_factory=Counter)        # this level
    by_source_total: Counter = field(default_factory=Counter)
    todo: list[TodoItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)             # agent hypotheses and verdicts
    events: list[str] = field(default_factory=list)
    blocked_colours: set = field(default_factory=set)          # walls found by the planner (this level)
    clicked: set = field(default_factory=set)                  # object positions clicked by the planner (this level)
    reached: set = field(default_factory=set)                  # colours explore already walked to (this level)
    t0: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.explorer = Explorer(cfg=self.cfg, hud=HudDetector(self.cfg))
        obs = self.session.last or self.session.start()
        self.explorer.start(obs)
        self.model.level_start = obs.grid.copy()
        self.model.remember(obs.grid, None)
        self._todo_id = 0
        self.refresh_todo()
        self.save()

    # ------------------------------------------------------------------ playing

    @property
    def obs(self) -> Obs:
        assert self.session.last is not None
        return self.session.last

    @property
    def mask(self):
        m = self.explorer.hud.mask
        return m if m.any() else None

    def step(self, key: ActionKey, source: str = "agent") -> tuple[Obs, str]:
        """Play one action; returns the new observation and a one-line event note."""
        prev = self.obs
        aid, x, y = key_to_call(key)
        if aid == 0 and not self.session.can_reset():
            return prev, "RESET ignored: the level has just started (RESET only after GAME_OVER)"
        obs = self.session.step(aid, x, y)
        self.by_source[source] += 1
        self.by_source_total[source] += 1
        self.explorer.observe(prev, key if aid else RESET, obs)
        note = ""
        if aid:
            level_up = obs.levels_completed > prev.levels_completed
            game_over = obs.state == GameState.GAME_OVER and prev.state != GameState.GAME_OVER
            self.model.observe(Transition(prev.levels_completed, key, prev.grid, obs.grid, level_up, game_over), self.mask)
            if level_up:
                n = self.session.level_actions[-1]
                base = self.baseline[len(self.session.level_actions) - 1] if self.baseline and \
                    len(self.session.level_actions) <= len(self.baseline) else None
                note = f"LEVEL {prev.levels_completed + 1} COMPLETED in {n} actions" + (
                    f" (human {base}, level score {min(115.0, 100 * (base / n) ** 2):.0f})" if base else "")
                self.events.append(note)
                self.by_source = Counter()
                self.model.level_start = obs.grid.copy()
                self.blocked_colours, self.clicked, self.reached = set(), set(), set()
                self.carry_todo()
            elif game_over:
                note = "GAME_OVER " + self.model.deaths[-1] + " -> RESET to retry the level"
                self.events.append(note)
        self.model.remember(obs.grid, self.mask)
        self.refresh_todo()
        self.save()
        return obs, note

    # ------------------------------------------------------------------ exploring

    def explore(self, budget: int, stop_on_event: bool = True) -> str:
        """Curiosity policy for up to ``budget`` actions: untried actions and never-clicked object
        kinds first; once an avatar is known, walk it to the rarest colour it has not entered yet
        (novelty); otherwise the state graph's nearest unexplored (state, action)."""
        from .planner import plan

        before = self.model.describe(self.obs.available_actions)
        start_actions, notes = self.session.actions, []
        while self.session.actions - start_actions < budget and self.obs.state != GameState.WIN:
            left = budget - (self.session.actions - start_actions)
            target = self._novel_colour()
            if target is not None:
                lvl = self.obs.levels_completed
                self.reached.add(target)
                out = plan(self, {"reach": {"color": target}}, max_actions=min(left, 80))
                for line in out.splitlines():
                    if line.startswith(("LEVEL", "GAME_OVER")):
                        notes.append(line)
                if stop_on_event and self.obs.levels_completed != lvl:
                    break
                continue
            key = self._curious_action()
            obs, note = self.step(key, "explore")
            if note:
                notes.append(note)
                if stop_on_event and ("LEVEL" in note):
                    break
        after = self.model.describe(self.obs.available_actions)
        new = [l for l in after.splitlines() if l not in before.splitlines()]
        out = [f"explored {self.session.actions - start_actions} actions"]
        out += notes[-5:]
        if new:
            out.append("new or changed rules:")
            out += ["  " + l for l in new[:12]]
        return "\n".join(out)

    def _novel_colour(self) -> Optional[int]:
        """Rarest on-screen colour the avatar has never entered (and not yet targeted this level)."""
        import numpy as np

        if self.obs.state != GameState.NOT_FINISHED or not self.model.avatar():
            return None
        if any(self.model.actions.get(a) is None for a in self.obs.available_actions if a not in (0, 6)):
            return None
        g = self.obs.grid
        if self.mask is not None:
            g = np.where(self.mask, -1, g)
        vals, counts = np.unique(g, return_counts=True)
        bg = int(vals[np.argmax(counts)])
        own = self.model.avatar_colours()
        cand = [(int(n), int(c)) for c, n in zip(vals, counts)
                if c >= 0 and c != bg and c not in own and self.model.entered.get(int(c), 0) == 0 and int(c) not in self.reached]
        return min(cand)[1] if cand else None

    def _curious_action(self) -> ActionKey:
        obs = self.obs
        if obs.state == GameState.GAME_OVER:
            return RESET
        avail = [a for a in obs.available_actions if a]
        for a in avail:  # every non-click action a few times first
            st = self.model.actions.get(a)
            if a != 6 and (st is None or st.n < 3):
                return (a,)
        if 6 in avail:
            un = self.model.unclicked_kinds(obs.grid, self.mask)
            if un:
                o = min(un, key=lambda o: o.size)
                x, y = o.center
                if obs.grid[y, x] != o.color:
                    import numpy as np
                    ys, xs = np.where(obs.grid[o.top:o.bottom + 1, o.left:o.right + 1] == o.color)
                    x, y = o.left + int(xs[len(xs) // 2]), o.top + int(ys[len(ys) // 2])
                return (6, x, y)
        key = self.explorer.next_action(obs)
        if key == RESET and not self.session.can_reset():
            node = self.explorer.graph.nodes[self.explorer.graph.key(obs.grid)]
            key = min(node.candidates, key=lambda k: node.edges[k].count if k in node.edges else 0)
        return key

    # ------------------------------------------------------------------ todo

    def refresh_todo(self) -> None:
        keep = [t for t in self.todo if t.source == "agent" or t.done]
        auto: list[tuple[str, int]] = []
        avail = [a for a in self.obs.available_actions if a]
        for a in avail:
            if a != 6 and (a not in self.model.actions or self.model.actions[a].n == 0):
                auto.append((f"try ACTION{a} (never tried)", 1))
        if 6 in avail:
            un = self.model.unclicked_kinds(self.obs.grid, self.mask)
            if un:
                auto.append((f"click the {len(un)} never-clicked object kinds, e.g. "
                             + ", ".join(f"{kind_name(o)} at {o.center}" for o in un[:3]), len(un)))
        g = self.explorer.graph
        if g is not None:
            frontier = sum(1 for n in g.nodes.values() if n.untried(g.banned))
            if frontier:
                auto.append((f"{frontier} known states still have untried actions (arc_explore)", frontier))
        for n in self.model.level_ups[-1:]:
            auto.append((f"check on this level whether the last win condition holds: {n}", 0))
        texts = {t.text for t in keep}
        self.todo = keep + [TodoItem(self._next_id(), t, "auto", c) for t, c in auto if t not in texts]

    def carry_todo(self) -> None:
        for t in self.todo:
            if t.source == "agent" and not t.done:
                t.text = f"(from level {self.obs.levels_completed}) {t.text}"

    def _next_id(self) -> int:
        self._todo_id += 1
        return self._todo_id

    def todo_add(self, text: str, cost: int = 1) -> TodoItem:
        item = TodoItem(self._next_id(), text, "agent", cost)
        self.todo.append(item)
        self.save()
        return item

    def todo_done(self, item_id: int, drop: bool = False) -> bool:
        for t in self.todo:
            if t.id == item_id:
                if drop:
                    self.todo.remove(t)
                else:
                    t.done = True
                self.save()
                return True
        return False

    def todo_text(self, limit: int = 5) -> str:
        open_ = [t for t in self.todo if not t.done]
        open_.sort(key=lambda t: (t.source != "agent", t.cost))
        return "\n".join(f"[{t.id}] ({t.source}, ~{t.cost} actions) {t.text}" for t in open_[:limit]) or "(empty)"

    # ------------------------------------------------------------------ ledger

    def status(self) -> str:
        s, o = self.session, self.obs
        src = "/".join(f"{k} {v}" for k, v in sorted(self.by_source.items())) or "none"
        acts = ",".join("RESET" if a == 0 else str(a) for a in o.available_actions)
        return (f"L{o.levels_completed + 1}/{s.win_levels} {o.state.name} | this level {s.level_so_far} actions ({src}) "
                f"| total {s.actions} | deaths {s.deaths} resets {s.resets} | actions [{acts}]")

    def ledger(self) -> dict[str, Any]:
        s = self.session
        return {
            "status": self.status(),
            "level": self.obs.levels_completed, "win_levels": s.win_levels, "state": self.obs.state.name,
            "actions": s.actions, "level_actions": s.level_actions, "level_so_far": s.level_so_far,
            "by_source_level": dict(self.by_source), "by_source_total": dict(self.by_source_total),
            "deaths": s.deaths, "resets": s.resets, "events": self.events[-20:],
            "rules": self.model.describe(self.obs.available_actions),
            "todo": [t.__dict__ for t in self.todo], "notes": self.notes[-30:],
            "graph_states": len(self.explorer.graph.nodes) if self.explorer.graph else 0,
            "elapsed_s": round(time.time() - self.t0, 1),
        }

    def save(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.ledger(), f, indent=1)
        os.replace(tmp, self.path)
