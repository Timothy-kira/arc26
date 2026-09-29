"""One game being played, with the competition's action accounting and RESET rules.

The Kaggle gateway runs the scorecard in competition mode: a single play per game, and a
RESET at the start of a level (engine ``_action_count == 0``) is ignored instead of
restarting the whole game. Offline, the same RESET would silently start a new play from
level 0, so the session refuses it in both modes and the local numbers match the rerun.

Action accounting follows ``arc_agi/scorecard.py``: every action sent, RESET included,
adds one to the play's total, and a level's count runs from the previous level-up to its
own level-up (deaths and resets in between included).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from arcengine import GameAction, GameState


@dataclass
class Obs:
    frames: list[np.ndarray]
    state: GameState
    levels_completed: int
    win_levels: int
    available_actions: list[int]

    @property
    def grid(self) -> np.ndarray:
        return self.frames[-1] if self.frames else np.zeros((64, 64), dtype=np.int8)


@dataclass
class Session:
    wrapper: Any
    actions: int = 0                                          # scorecard total for this play
    level_actions: list[int] = field(default_factory=list)   # actions per completed level
    level_start: int = 0                                      # total when the current level began
    since_level_reset: int = 0                                # engine _action_count
    deaths: int = 0
    resets: int = 0
    win_levels: int = 0
    last: Optional[Obs] = None

    @staticmethod
    def _convert(raw: Any, prev: Optional[Obs]) -> Obs:
        frames = [np.asarray(f, dtype=np.int8) for f in (raw.frame or [])]
        if not frames and prev is not None:
            frames = [prev.grid]
        return Obs(frames, raw.state, int(raw.levels_completed), int(raw.win_levels),
                   [int(a) for a in (raw.available_actions or [])])

    @property
    def level(self) -> int:
        return self.last.levels_completed if self.last else 0

    @property
    def level_so_far(self) -> int:
        return self.actions - self.level_start

    def can_reset(self) -> bool:
        return self.last is None or self.since_level_reset > 0 or self.last.state == GameState.WIN

    def step(self, action: int, x: Optional[int] = None, y: Optional[int] = None) -> Obs:
        ga = GameAction.from_id(int(action))
        if ga == GameAction.RESET and not self.can_reset():
            assert self.last is not None
            return self.last  # the gateway ignores it; offline it would restart the game
        data = {"x": int(x or 0), "y": int(y or 0)} if ga.is_complex() else None
        raw = self.wrapper.step(ga, data=data)
        if raw is None:
            raise RuntimeError(f"engine returned nothing for {ga.name}")
        prev = self.last
        obs = self._convert(raw, prev)
        self.actions += 1
        self.win_levels = max(self.win_levels, obs.win_levels)  # GAME_OVER frames report 0
        if ga == GameAction.RESET:
            self.resets += 1
            self.since_level_reset = 0
        else:
            self.since_level_reset += 1
        if obs.state == GameState.GAME_OVER and (prev is None or prev.state != GameState.GAME_OVER):
            self.deaths += 1
        if prev is not None and obs.levels_completed > prev.levels_completed:
            self.level_actions.append(self.actions - self.level_start)
            self.level_start = self.actions
            self.since_level_reset = 0
        self.last = obs
        return obs

    def start(self) -> Obs:
        """The opening RESET starts the play (a full reset); the scorecard does not count it."""
        obs = self.step(0)
        self.actions = self.resets = 0
        return obs
