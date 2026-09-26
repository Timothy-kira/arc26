"""What the game daemon keeps about one game: the session, the step ledger and the notebook.

The ledger counts every action like the scorecard does (via ``Session``), attributes it to its
source, and writes ``ledger.json`` after every action. The notebook (rules / goal / levels /
plan) lives here, outside the conversation, so context compaction can never change it: it is
returned verbatim by the tools and pasted verbatim into every continuation prompt.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from arcengine import GameState

from .session import Obs, Session

NOTEBOOK_SECTIONS = ("rules", "goal", "levels", "plan")
NOTEBOOK_LIMIT = 1500  # characters per section: re-sent verbatim, so it stays short


@dataclass
class GameLedger:
    session: Session
    path: Optional[str] = None
    baseline: Optional[list[int]] = None           # human per-level actions (offline only)
    by_source: Counter = field(default_factory=Counter)        # this level
    by_source_total: Counter = field(default_factory=Counter)
    notebook: dict = field(default_factory=lambda: {k: "" for k in NOTEBOOK_SECTIONS})
    events: list[str] = field(default_factory=list)
    t0: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.session.last is None:
            self.session.start()
        self.save()

    @property
    def obs(self) -> Obs:
        assert self.session.last is not None
        return self.session.last

    def step(self, action: int, x: Optional[int] = None, y: Optional[int] = None, source: str = "agent") -> tuple[Obs, str]:
        """Play one action; returns the new observation and an event note ('' if nothing special)."""
        prev = self.obs
        if int(action) == 0 and not self.session.can_reset():
            return prev, "RESET ignored: the level has just started (RESET only helps after GAME_OVER)"
        obs = self.session.step(action, x, y)
        self.by_source[source] += 1
        self.by_source_total[source] += 1
        note = ""
        if obs.levels_completed > prev.levels_completed:
            n = self.session.level_actions[-1]
            k = len(self.session.level_actions) - 1
            base = self.baseline[k] if self.baseline and k < len(self.baseline) else None
            note = f"LEVEL {prev.levels_completed + 1} COMPLETED in {n} actions" + (
                f" (human {base}, level score {min(115.0, 100 * (base / n) ** 2):.0f})" if base else "")
            self.events.append(note)
            self.by_source = Counter()
        elif obs.state == GameState.GAME_OVER and prev.state != GameState.GAME_OVER:
            note = f"GAME_OVER on level {obs.levels_completed + 1} -> RESET (action 0) retries the level"
            self.events.append(note)
        elif obs.state == GameState.WIN:
            note = "GAME WON"
            self.events.append(note)
        self.save()
        return obs, note

    # ------------------------------------------------------------------ notebook

    def note(self, section: str, text: str, mode: str = "replace") -> str:
        if section not in NOTEBOOK_SECTIONS:
            return f"unknown section {section!r}; use one of {', '.join(NOTEBOOK_SECTIONS)}"
        text = text.strip()
        cur = self.notebook.get(section, "")
        new = (cur + "\n" + text).strip() if mode == "append" and cur else text
        cut = ""
        if len(new) > NOTEBOOK_LIMIT:
            new = new[-NOTEBOOK_LIMIT:]
            cut = f" (kept the last {NOTEBOOK_LIMIT} characters: condense it)"
        self.notebook[section] = new
        self.save()
        return f"notebook.{section} saved ({len(new)} chars){cut}"

    def notebook_text(self) -> str:
        parts = [f"[{k}]\n{v}" for k, v in self.notebook.items() if v]
        return "\n".join(parts) if parts else "(empty)"

    # ------------------------------------------------------------------ ledger

    def status(self) -> str:
        s, o = self.session, self.obs
        src = "/".join(f"{k} {v}" for k, v in sorted(self.by_source.items())) or "none"
        acts = ",".join("RESET" if a == 0 else str(a) for a in o.available_actions)
        return (f"L{o.levels_completed + 1}/{s.win_levels} {o.state.name} | this level {s.level_so_far} actions ({src}) "
                f"| total {s.actions} | per level {s.level_actions} | deaths {s.deaths} resets {s.resets} | actions [{acts}]")

    def ledger(self) -> dict[str, Any]:
        s = self.session
        return {
            "status": self.status(), "level": self.obs.levels_completed, "win_levels": s.win_levels,
            "state": self.obs.state.name, "actions": s.actions, "level_actions": s.level_actions,
            "level_so_far": s.level_so_far, "by_source_level": dict(self.by_source),
            "by_source_total": dict(self.by_source_total), "deaths": s.deaths, "resets": s.resets,
            "events": self.events[-30:], "notebook": self.notebook, "elapsed_s": round(time.time() - self.t0, 1),
        }

    def save(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.ledger(), f, indent=1)
        os.replace(tmp, self.path)
