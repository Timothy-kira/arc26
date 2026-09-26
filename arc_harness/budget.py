"""Wall-clock budgeting across ~110 hidden games in ~9 hours.

Each game gets a deadline derived from the time left and the games left, scaled
by how many games run concurrently. Reasoning stops a little before the deadline
so the explorer can spend the tail cheaply; unused time flows to later games.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class GameBudget:
    start: float
    deadline: float
    llm_deadline: float
    max_actions: int

    def now(self) -> float:
        return time.monotonic()

    def expired(self) -> bool:
        return self.now() >= self.deadline

    def llm_ok(self) -> bool:
        return self.now() < self.llm_deadline

    def remaining(self) -> float:
        return max(0.0, self.deadline - self.now())


class GlobalBudget:
    def __init__(
        self,
        total_seconds: float,
        n_games: int,
        concurrency: int,
        reserve_seconds: float = 900.0,
        max_game_seconds: float = 3600.0,
        tail_fraction: float = 0.15,
        max_actions_per_game: int = 2500,
    ) -> None:
        self.t0 = time.monotonic()
        self.total = total_seconds
        self.n_games = n_games
        self.concurrency = max(1, concurrency)
        self.reserve = reserve_seconds
        self.max_game = max_game_seconds
        self.tail_fraction = tail_fraction
        self.max_actions = max_actions_per_game
        self.started = 0
        self.finished = 0
        self._lock = threading.Lock()

    def remaining(self) -> float:
        return max(0.0, self.total - self.reserve - (time.monotonic() - self.t0))

    def for_game(self) -> GameBudget:
        with self._lock:
            self.started += 1
            games_left = max(1, self.n_games - self.started + 1)
            waves_left = max(1.0, games_left / self.concurrency)
            share = min(self.max_game, self.remaining() / waves_left)
            now = time.monotonic()
            return GameBudget(
                start=now,
                deadline=now + share,
                llm_deadline=now + share * (1 - self.tail_fraction),
                max_actions=self.max_actions,
            )

    def done(self) -> None:
        with self._lock:
            self.finished += 1
