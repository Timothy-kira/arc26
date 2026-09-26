"""Thin adapter over the official ``arc_agi`` engine.

One code path serves both local development (``OFFLINE`` mode over the public
``environment_files``) and the Kaggle competition rerun, where a gateway sidecar
at ``http://gateway:8001`` serves the hidden games through the same REST API the
``ONLINE`` wrapper speaks.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

logger = logging.getLogger(__name__)

KAGGLE_GATEWAY_URL = "http://gateway:8001"
DEFAULT_ENV_DIR = Path(__file__).resolve().parents[2] / "data" / "environment_files"


def is_competition_rerun() -> bool:
    return bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))


@dataclass
class Obs:
    """One observation returned after RESET or an action."""

    frames: list[np.ndarray]
    state: GameState
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    full_reset: bool = False

    @property
    def grid(self) -> np.ndarray:
        return self.frames[-1] if self.frames else np.zeros((64, 64), dtype=np.int8)

    @property
    def done(self) -> bool:
        return self.state == GameState.WIN


@dataclass
class StepRecord:
    action: int
    x: Optional[int]
    y: Optional[int]
    level: int
    state_after: str
    levels_after: int


@dataclass
class GameSession:
    """A single game being played. Counts every environment action (what RHAE scores)."""

    game_id: str
    wrapper: Any
    tags: list[str] = field(default_factory=list)
    baseline_actions: list[int] = field(default_factory=list)
    actions_taken: int = 0
    level_actions: dict[int, int] = field(default_factory=dict)
    history: list[StepRecord] = field(default_factory=list)
    last: Optional[Obs] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @staticmethod
    def _convert(raw: Any) -> Obs:
        frames = [np.asarray(f, dtype=np.int8) for f in (raw.frame or [])]
        return Obs(
            frames=frames,
            state=raw.state,
            levels_completed=int(raw.levels_completed),
            win_levels=int(raw.win_levels),
            available_actions=[int(a) for a in (raw.available_actions or [])],
            full_reset=bool(getattr(raw, "full_reset", False)),
        )

    def reset(self) -> Obs:
        return self.step(GameAction.RESET.value)

    def step(
        self,
        action: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> Obs:
        ga = GameAction.from_id(int(action))
        data = None
        if ga.is_complex():
            data = {"x": int(x if x is not None else 0), "y": int(y if y is not None else 0)}
        with self._lock:
            level = self.last.levels_completed if self.last else 0
            raw = self.wrapper.step(ga, data=data, reasoning=reasoning)
            if raw is None:
                raise RuntimeError(f"{self.game_id}: step {ga.name} returned None")
            obs = self._convert(raw)
            self.actions_taken += 1
            self.level_actions[level] = self.level_actions.get(level, 0) + 1
            self.history.append(
                StepRecord(int(action), x, y, level, obs.state.name, obs.levels_completed)
            )
            self.last = obs
            return obs

    def baseline_for(self, level: int) -> Optional[int]:
        if 0 <= level < len(self.baseline_actions):
            return self.baseline_actions[level]
        return None


class ArcEnv:
    """Factory for game sessions; hides local vs. competition wiring."""

    def __init__(self, mode: str = "auto", environments_dir: Optional[str] = None) -> None:
        if mode == "auto":
            mode = "competition" if is_competition_rerun() else "offline"
        self.mode = mode
        quiet = logging.getLogger("arc_agi.quiet")
        quiet.setLevel(logging.WARNING)
        if mode == "offline":
            env_dir = environments_dir or os.getenv("ENVIRONMENTS_DIR") or str(DEFAULT_ENV_DIR)
            self.arc = Arcade(
                operation_mode=OperationMode.OFFLINE, environments_dir=env_dir, logger=quiet
            )
        else:
            base = os.getenv("ARC_BASE_URL", KAGGLE_GATEWAY_URL).rstrip("/")
            self.arc = Arcade(
                arc_api_key=os.getenv("ARC_API_KEY", "test-key-123"),
                arc_base_url=base,
                operation_mode=OperationMode.ONLINE,
                logger=quiet,
            )
        self._infos = {e.game_id: e for e in self.arc.get_environments()}
        self.card_id: Optional[str] = None

    def list_games(self) -> list[str]:
        return sorted(self._infos)

    def open(self, tags: Optional[list[str]] = None) -> str:
        self.card_id = self.arc.open_scorecard(tags=tags or ["arc26"])
        return self.card_id

    def make(self, game_id: str, seed: int = 0) -> GameSession:
        if self.card_id is None:
            self.open()
        wrapper = self.arc.make(game_id, seed=seed, scorecard_id=self.card_id)
        if wrapper is None:
            raise RuntimeError(f"could not make game {game_id}")
        info = self._infos.get(game_id) or next(
            (v for k, v in self._infos.items() if k.startswith(game_id)), None
        )
        tags = list(getattr(info, "tags", None) or [])
        baseline = list(getattr(info, "baseline_actions", None) or [])
        return GameSession(game_id=game_id, wrapper=wrapper, tags=tags, baseline_actions=baseline)

    def scorecard(self) -> Any:
        return self.arc.get_scorecard(self.card_id) if self.card_id else None

    def close(self) -> Any:
        if self.card_id is None:
            return None
        card = self.arc.close_scorecard(self.card_id)
        self.card_id = None
        return card
