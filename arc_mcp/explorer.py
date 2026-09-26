"""LLM-free exploration policy built on :class:`StateGraph`.

It is both a standalone baseline agent and the low-level executor under the LLM:
the LLM supplies action priorities, bans and explicit plans; the explorer turns
them into concrete environment actions and records what happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from arcengine import GameState

from .session import Obs, Session
from .config import DEFAULT, ExploreConfig
from .graph import ActionKey, HudDetector, StateGraph, key_to_call

RESET: ActionKey = (0,)


@dataclass
class LevelLog:
    level: int
    actions: int = 0
    deaths: int = 0
    resets: int = 0
    won: bool = False
    winning_path: Optional[list[ActionKey]] = None


@dataclass
class Explorer:
    cfg: ExploreConfig = field(default_factory=ExploreConfig)
    hud: HudDetector = field(default_factory=HudDetector)
    graph: Optional[StateGraph] = None
    level: int = 0
    plan: list[ActionKey] = field(default_factory=list)
    levels: dict[int, LevelLog] = field(default_factory=dict)
    graphs: dict[int, StateGraph] = field(default_factory=dict)
    steps_since_new_state: int = 0

    def start(self, obs: Obs) -> None:
        self.level = obs.levels_completed
        self._new_graph(obs)

    def _new_graph(self, obs: Obs) -> None:
        self.graph = StateGraph(obs.available_actions, self.cfg, self.hud)
        self.graphs[self.level] = self.graph
        self.graph.visit(obs.grid)
        self.levels.setdefault(self.level, LevelLog(self.level))
        self.plan = []
        self.steps_since_new_state = 0

    @property
    def log(self) -> LevelLog:
        return self.levels[self.level]

    def set_plan(self, plan: list[ActionKey]) -> None:
        self.plan = list(plan)

    def next_action(self, obs: Obs) -> ActionKey:
        if obs.state in (GameState.GAME_OVER, GameState.NOT_PLAYED):
            return RESET
        g = self.graph
        assert g is not None
        cur = g.visit(obs.grid).key
        while self.plan:
            a = self.plan.pop(0)
            if a == RESET or a[0] in obs.available_actions:
                return a
        p = g.plan_to_frontier(cur)
        if p:
            self.plan = p[1:]
            return p[0]
        if self.cfg.reset_when_no_frontier and g.start and cur != g.start:
            return RESET
        # Fully explored and at start: allow repeating the least-tried action.
        node = g.nodes[cur]
        live = [a for a in node.candidates if not node.edges.get(a) or not node.edges[a].game_over]
        if not live:
            return RESET
        return min(live, key=lambda a: node.edges[a].count if a in node.edges else 0)

    def observe(self, prev: Obs, action: ActionKey, obs: Obs) -> None:
        g = self.graph
        assert g is not None
        self.log.actions += 1
        if action == RESET:
            self.log.resets += 1
            self.plan = []
            if obs.levels_completed != self.level:
                self.level = obs.levels_completed
                self._new_graph(obs)
            else:
                g.visit(obs.grid)
            return
        level_up = obs.levels_completed > prev.levels_completed
        game_over = obs.state == GameState.GAME_OVER
        n_before = len(g.nodes)
        edge = g.record(prev.grid, action, obs.grid, game_over=game_over, level_up=level_up)
        self.steps_since_new_state = 0 if len(g.nodes) > n_before else self.steps_since_new_state + 1
        if game_over:
            self.log.deaths += 1
            self.plan = []
        if edge.noop and self.plan:
            self.plan = []
        if level_up or obs.state == GameState.WIN:
            self.log.won = True
            self.log.winning_path = g.winning_path()
            self.level = obs.levels_completed
            if obs.state != GameState.WIN:
                self._new_graph(obs)


def play_explore(session: Session, max_actions: int = 400, cfg: ExploreConfig = DEFAULT) -> Explorer:
    """Baseline: play one game with the pure explorer until WIN or the action cap."""
    ex = Explorer(cfg=cfg, hud=HudDetector(cfg))
    obs = session.last or session.start()
    ex.start(obs)
    while session.actions < max_actions and obs.state != GameState.WIN:
        a = ex.next_action(obs)
        aid, x, y = key_to_call(a)
        if aid == 0 and not session.can_reset():
            # competition rules: no RESET at level start; take the least-tried action instead
            node = ex.graph.nodes[ex.graph.key(obs.grid)]
            a = min(node.candidates, key=lambda k: node.edges[k].count if k in node.edges else 0)
            aid, x, y = key_to_call(a)
        nxt = session.step(aid, x, y)
        ex.observe(obs, a, nxt)
        obs = nxt
    return ex
