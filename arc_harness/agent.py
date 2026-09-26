"""The game agent: a cheap explorer in the loop, an LLM reasoning DAG on events.

Per game:
  explore burst (no LLM) --event--> reasoning round (one DAG) --plan--> act ...
Events: level start, GAME_OVER, no new state for a while, burst exhausted.

A reasoning round is one graph (Raven: "iteration is graphs in series"):
  perceive -> {hyp_objects, hyp_diff, hyp_image} (parallel) -> plan -> act(instance=env)
If ``act`` fails its verdict (all no-ops / immediate death) the host replans once:
a successor graph with a fresh plan that sees what went wrong.

Level end writes a Case; skill usage outcomes are fed back; game end distills
skills from this game's best cases.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from arcengine import GameState

from . import prompts
from .budget import GameBudget
from .charter import Charter
from .dag.graph import DagNode, DagSpec
from .dag.runner import InstanceLocks, NodeContext, run_dag
from .dag.verdict import Decision, Verdict
from .env.arcade import GameSession, Obs
from .env.encode import describe_scene, diff, to_png_b64
from .explore.config import ExploreConfig
from .explore.explorer import RESET, Explorer
from .explore.state_graph import HudDetector, key_name, key_to_call
from .llm.client import ChatModel
from .llm.semantic import call_json
from .memory.cases import Case, objective_quality
from .memory.features import initial_features
from .memory.hub import MemoryHub
from .memory.notebook import GameNotebook, HypothesisEntry
from .memory.retrieve import Hit, render_hits, retrieve
from .roles.schemas import CharterOut, HypothesesOut, LevelSummaryOut, PlanOut

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    max_actions: int = 2500
    llm_rounds_per_level: int = 10
    burst_actions: int = 30
    stuck_steps: int = 12
    plan_max_actions: int = 24
    hypothesizers: tuple[str, ...] = ("objects", "diff", "image")
    plan_thinking: bool = False
    hyp_thinking: bool = False
    distill_per_game: int = 2
    use_skills: bool = True
    explore: ExploreConfig = field(default_factory=ExploreConfig)


@dataclass
class GameReport:
    game_id: str
    actions: int = 0
    levels_completed: int = 0
    win_levels: int = 0
    state: str = ""
    llm_rounds: int = 0
    replans: int = 0
    seconds: float = 0.0
    skills_used: list[str] = field(default_factory=list)
    skills_written: list[str] = field(default_factory=list)
    error: str = ""


class GameAgent:
    def __init__(
        self,
        session: GameSession,
        llm: Optional[ChatModel],
        memory: MemoryHub,
        budget: GameBudget,
        cfg: Optional[AgentConfig] = None,
        workdir: Optional[Path] = None,
        locks: Optional[InstanceLocks] = None,
    ) -> None:
        self.s = session
        self.llm = llm
        self.mem = memory
        self.budget = budget
        self.cfg = cfg or AgentConfig()
        self.workdir = workdir
        self.locks = locks or InstanceLocks()
        self.ex = Explorer(cfg=self.cfg.explore, hud=HudDetector(self.cfg.explore))
        self.nb = GameNotebook(session.game_id)
        self.charter = Charter()
        self.report = GameReport(session.game_id)
        self.obs: Optional[Obs] = None
        self.round = 0
        self.used_ids: set[str] = set()
        self.level_rounds = 0
        self.level_start_actions = 0
        self.skill_hits: list[Hit] = []
        self.level_cases: list[Case] = []
        self._marker = 0
        self._last_spec_hyps: list[str] = []

    # ------------------------------------------------------------------ env

    async def _step(self, key: tuple) -> Obs:
        assert self.obs is not None
        prev = self.obs
        aid, x, y = key_to_call(key)
        obs = await asyncio.to_thread(self.s.step, aid, x, y)
        self.ex.observe(prev, key, obs)
        d = diff(prev.grid, obs.grid, self.ex.hud.mask if self.ex.hud.mask.any() else None)
        res = d.describe(4)
        if obs.levels_completed > prev.levels_completed:
            res = "LEVEL COMPLETE! " + res
        elif obs.state == GameState.GAME_OVER:
            res = "GAME_OVER. " + res
            self.nb.level(prev.levels_completed).deaths.append(
                f"after {key_name(key)}: {res[:160]}"
            )
        self.nb.steps.add(key_name(key) if key != RESET else "RESET", res)
        self.obs = obs
        return obs

    def _level(self) -> int:
        return self.obs.levels_completed if self.obs else 0

    def _level_actions(self) -> int:
        return self.s.actions_taken - self.level_start_actions

    def _can_act(self) -> bool:
        return (
            self.obs is not None
            and self.obs.state != GameState.WIN
            and self.s.actions_taken < min(self.cfg.max_actions, self.budget.max_actions)
            and not self.budget.expired()
        )

    def _can_reason(self) -> bool:
        return (
            self.llm is not None
            and self.budget.llm_ok()
            and self.level_rounds < self.cfg.llm_rounds_per_level
            and self.charter.reasoning_allowed(self._level_actions())
        )

    # ------------------------------------------------------------- top loop

    async def play(self) -> GameReport:
        t0 = time.monotonic()
        try:
            self.obs = await asyncio.to_thread(self.s.reset)
            self.ex.start(self.obs)
            feats = initial_features(self.obs.grid, self.obs.available_actions, len(self.obs.frames))
            self.nb.features = feats.tokens()
            await self._start_level()
            while self._can_act():
                event = await self._explore_burst()
                if not self._can_act():
                    break
                if event == "level_up":
                    await self._end_level(won=True)
                    await self._start_level()
                    continue
                if self._can_reason():
                    await self._reason_round(event)
                    if self.obs.levels_completed > self._marker and self._can_act():
                        await self._end_level(won=True)
                        await self._start_level()
            if self.obs and self.obs.levels_completed > self._marker:
                await self._end_level(won=True)
            elif self._level_actions() > 0:
                await self._end_level(won=False)
            await self._end_game()
        except Exception as exc:  # a crashed game must never take the whole run down
            logger.exception("game %s crashed", self.s.game_id)
            self.report.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.report.seconds = time.monotonic() - t0
            self.report.actions = self.s.actions_taken
            if self.obs:
                self.report.levels_completed = self.obs.levels_completed
                self.report.win_levels = self.obs.win_levels
                self.report.state = self.obs.state.name
            if self.workdir:
                self.workdir.mkdir(parents=True, exist_ok=True)
                (self.workdir / "notebook.md").write_text(self.nb.to_markdown())
                (self.workdir / "report.json").write_text(json.dumps(self.report.__dict__, indent=2))
        return self.report

    async def _explore_burst(self) -> str:
        """Run the explorer until something worth reasoning about happens."""
        assert self.obs is not None
        start_level = self._level()
        n = skips = 0
        limit = self.cfg.burst_actions if self.llm is not None else 10**9
        while self._can_act() and n < limit:
            key = self.ex.next_action(self.obs)
            if key != RESET and not self.charter.allows(key) and skips < 32:
                self.ex.graph.banned.add(key)  # type: ignore[union-attr]
                skips += 1
                continue
            obs = await self._step(key)
            n += 1
            if obs.levels_completed > start_level:
                return "level_up"
            if obs.state == GameState.GAME_OVER:
                return "game_over"
            if self.llm is not None and self.ex.steps_since_new_state >= self.cfg.stuck_steps:
                self.ex.steps_since_new_state = 0
                return "stuck"
        return "burst"

    # ------------------------------------------------------------- levels

    async def _start_level(self) -> None:
        assert self.obs is not None
        self._marker = self._level()
        self.level_rounds = 0
        self.level_start_actions = self.s.actions_taken
        self.skill_hits = []
        self.charter = Charter()
        if self.llm is None:
            return
        level = self._level()
        situation = (
            f"features: {' '.join(self.nb.features)}\n"
            + self.nb.brief(level, 1200)
            + "\n"
            + describe_scene(self.obs.grid, 20)
        )
        if self.cfg.use_skills:
            try:
                self.skill_hits = await retrieve(
                    skills=self.mem.skills.all(),
                    cases=self.mem.cases.cases,
                    query=situation,
                    features=self.nb.features,
                    llm=self.llm,
                    exclude_game=self.s.game_id,
                )
            except Exception as exc:
                logger.warning("skill retrieval failed: %s", exc)
        self.nb.skills_in_use = [h.id for h in self.skill_hits if h.kind == "skill"]
        for sid in self.nb.skills_in_use:
            if sid not in self.report.skills_used:
                self.report.skills_used.append(sid)
        try:
            out = await call_json(
                self.llm,
                CharterOut,
                prompts.load("system"),
                situation + "\n\n" + render_hits(self.skill_hits) + "\n\n" + prompts.load("charter"),
                retries=1,
                max_tokens=600,
            )
            self.charter = Charter.from_out(out)
            self.charter.apply(self.ex)
        except Exception as exc:
            logger.warning("charter failed: %s", exc)

    async def _end_level(self, won: bool) -> None:
        level = self._marker
        notes = self.nb.level(level)
        notes.won = won
        notes.actions = self._level_actions()
        graph = self.ex.graphs.get(level)
        path = graph.winning_path() if (won and graph) else None
        if path:
            notes.winning_path = [key_name(k) for k in path]
        baseline = self.s.baseline_for(level)
        quality = objective_quality(won, notes.actions, baseline, len(path) if path else None)
        outcome = "won" if won else ("died" if notes.deaths else "unsolved")
        summary = LevelSummaryOut(
            task_intent=notes.goal or "unknown goal",
            approach="; ".join(h.statement for h in notes.hypotheses if h.status == "confirmed")[:500]
            or "systematic exploration of all actions",
            key_insight=(notes.deaths[-1] if notes.deaths and not won else ""),
        )
        if self.llm is not None and self.budget.llm_ok():
            try:
                summary = await call_json(
                    self.llm,
                    LevelSummaryOut,
                    prompts.load("system"),
                    f"Outcome: {outcome} after {notes.actions} actions.\n"
                    f"Winning path (shortest known): {' '.join(notes.winning_path) or 'n/a'}\n\n"
                    + self.nb.brief(level, 2500)
                    + "\n\nRecent steps:\n"
                    + self.nb.steps.render(16)
                    + "\n\n"
                    + prompts.load("level_summary"),
                    retries=1,
                    max_tokens=700,
                )
            except Exception as exc:
                logger.warning("level summary failed: %s", exc)
        notes.summary = summary.approach[:300]
        case = Case(
            game_id=self.s.game_id,
            level=level,
            features=self.nb.features,
            task_intent=summary.task_intent,
            approach=summary.approach,
            key_insight=summary.key_insight,
            outcome=outcome,
            actions=notes.actions,
            quality=quality,
            action_summary=" ".join(notes.winning_path[:80]),
            skills_used=list(self.nb.skills_in_use),
        )
        self.mem.sedimenter.add_case(case)
        self.level_cases.append(case)
        eff = min(1.0, (baseline or len(path or []) or notes.actions) / max(notes.actions, 1))
        for sid in self.nb.skills_in_use:
            self.mem.skills.record_use(sid, won, eff)

    async def _end_game(self) -> None:
        if self.llm is None or not self.budget.llm_ok():
            return
        best = sorted((c for c in self.level_cases if c.quality >= 0.2), key=lambda c: -c.quality)
        for case in best[: self.cfg.distill_per_game]:
            try:
                for s in await self.mem.sedimenter.distill(case):
                    self.report.skills_written.append(s.id)
            except Exception as exc:
                logger.warning("distill failed: %s", exc)

    # ------------------------------------------------------------ reasoning

    def _perceive(self, event: str) -> str:
        assert self.obs is not None
        level = self._level()
        mask = self.ex.hud.mask if self.ex.hud.mask.any() else None
        g = self.ex.graph
        stats = g.stats() if g else {}
        acts = ", ".join(f"ACTION{a}" for a in self.obs.available_actions if a != 0)
        parts = [
            self.nb.brief(level),
            self.charter.render(),
            f"Event: {event}. State: {self.obs.state.name}. Levels {self.obs.levels_completed}/{self.obs.win_levels}. "
            f"Actions used this level: {self._level_actions()} (human baseline for this level: "
            f"{self.s.baseline_for(level) or 'unknown'}). Available: {acts}.",
            f"Explored {stats.get('states', 0)} distinct states, {stats.get('noops', 0)} no-op moves, "
            f"{stats.get('deaths', 0)} deaths"
            + (", an edge bar changes every move (likely a move/energy counter)" if mask is not None else "")
            + ".",
            "Current scene:\n" + describe_scene(self.obs.grid, 36, mask),
            "Recent steps (action -> what changed):\n" + self.nb.steps.render(20),
        ]
        if self.skill_hits:
            parts.append("Relevant skills from past games:\n" + render_hits(self.skill_hits))
        return "\n\n".join(p for p in parts if p)

    def _round_spec(self, r: int) -> DagSpec:
        hyp_nodes = [
            DagNode(
                id=f"r{r}_hyp_{v}",
                role="hypothesize",
                summary=f"{v} analyst",
                prompt_template="{{ r%d_perceive.output }}\n\nAnalyst view: {{ input.view }}" % r,
                depends_on=[f"r{r}_perceive"],
                inputs={"view": v},
            )
            for v in self.cfg.hypothesizers
        ]
        hyp_ids = [n.id for n in hyp_nodes]
        plan_tpl = "{{ r%d_perceive.output }}\n\n" % r + "\n\n".join(
            f"Analyst {n.inputs['view']}:\n{{{{ {n.id}.output }}}}" for n in hyp_nodes
        )
        return DagSpec(
            task_summary=f"reasoning round {r}",
            nodes=[
                DagNode(id=f"r{r}_perceive", role="perceive"),
                *hyp_nodes,
                DagNode(id=f"r{r}_plan", role="plan", prompt_template=plan_tpl, depends_on=[f"r{r}_perceive", *hyp_ids]),
                DagNode(
                    id=f"r{r}_act",
                    role="act",
                    prompt_template="{{ r%d_plan.output }}" % r,
                    depends_on=[f"r{r}_plan"],
                    instance=f"env:{self.s.game_id}",
                ),
            ],
        )

    async def _reason_round(self, event: str) -> None:
        self.round += 1
        self.level_rounds += 1
        self.report.llm_rounds += 1
        r = self.round
        plans: dict[str, PlanOut] = {}
        act_outcomes: dict[str, dict[str, Any]] = {}

        async def perceive(ctx: NodeContext) -> str:
            return self._perceive(event)

        async def hypothesize(ctx: NodeContext) -> str:
            view = ctx.node.inputs["view"]
            images = [to_png_b64(self.obs.grid)] if view == "image" else None  # type: ignore[union-attr]
            try:
                out = await call_json(
                    self.llm,  # type: ignore[arg-type]
                    HypothesesOut,
                    prompts.load("system"),
                    ctx.prompt + "\n\n" + prompts.load(f"hypothesize_{view}") + "\n\n" + prompts.load("hypothesize_output"),
                    images=images,
                    thinking=self.cfg.hyp_thinking,
                    max_tokens=900,
                )
            except Exception as exc:  # one analyst failing must not cascade-skip the plan
                out = HypothesesOut(notes=f"analyst unavailable: {type(exc).__name__}")
            return out.model_dump_json()

        async def plan(ctx: NodeContext) -> str:
            extra = ""
            if ctx.continuation:
                extra = f"\n\nYour previous plan failed: {ctx.continuation}\nPrevious plan: {ctx.previous_output[:1500]}"
            out = await call_json(
                self.llm,  # type: ignore[arg-type]
                PlanOut,
                prompts.load("system"),
                ctx.prompt + extra + "\n\n" + prompts.fill("plan", max_actions=self.cfg.plan_max_actions),
                thinking=self.cfg.plan_thinking,
            )
            plans[ctx.node.id] = out
            return out.model_dump_json()

        async def act(ctx: NodeContext) -> str:
            plan_id = ctx.node.depends_on[0]
            p = plans.get(plan_id) or PlanOut.model_validate_json(ctx.upstream[plan_id])
            self._absorb_plan(p)
            return json.dumps(await self._execute(p))

        async def judge(node: DagNode, output: str) -> Verdict:
            if node.role != "act":
                return Verdict()
            o = json.loads(output)
            act_outcomes[node.id] = o
            if o["level_up"] or (o["changed"] > 0 and not o["died"]):
                return Verdict()
            if o["executed"] == 0:
                return Verdict()  # plan handed control to the explorer
            what = "the game ended the attempt (GAME_OVER)" if o["died"] else "every action was a no-op"
            return Verdict(outcome="not_accomplished", category="no_progress", what_is_missing=what, evidence=o["log"][-600:])

        async def adjudicate(node: DagNode, output: str, verdict: Verdict, attempt: int) -> Decision:
            if node.role == "act" and not node.id.endswith("_b"):
                self.report.replans += 1
                base = node.id[: -len("_act")]
                hyp_ids = [d for d in self._last_spec_hyps]
                tpl = "{{ %s_perceive.output }}\n\n" % base + "\n\n".join(
                    f"Analyst:\n{{{{ {h}.output }}}}" for h in hyp_ids
                ) + f"\n\nThe previous plan was executed and failed: {verdict.what_is_missing}.\nObserved: {verdict.evidence}"
                successor = DagSpec(
                    task_summary=f"replan after {node.id}",
                    nodes=[
                        DagNode(id=f"{base}_plan_b", role="plan", prompt_template=tpl, depends_on=[f"{base}_perceive", *hyp_ids]),
                        DagNode(
                            id=f"{base}_act_b",
                            role="act",
                            prompt_template="{{ %s_plan_b.output }}" % base,
                            depends_on=[f"{base}_plan_b"],
                            instance=node.instance,
                        ),
                    ],
                )
                return Decision.replan(successor, verdict.what_is_missing)
            return Decision.abandon(verdict.what_is_missing)

        runners = {"perceive": perceive, "hypothesize": hypothesize, "plan": plan, "act": act}
        spec = self._round_spec(r)
        self._last_spec_hyps = [n.id for n in spec.nodes if n.role == "hypothesize"]
        wd = self.workdir / f"round_{r:03d}" if self.workdir else None
        prior: dict[str, str] = {}
        while spec is not None:
            res = await run_dag(
                spec,
                runners,
                locks=self.locks,
                judge=judge,
                adjudicate=adjudicate,
                prior_outputs=prior,
                used_ids=self.used_ids,
                workdir=wd,
            )
            self.used_ids |= set(res.status)
            prior.update(res.outputs)
            for nid, err in res.errors.items():
                if err and "superseded" not in err:
                    logger.info("%s: node %s %s", self.s.game_id, nid, err)
            spec = res.replanned

    def _absorb_plan(self, p: PlanOut) -> None:
        level = self._level()
        notes = self.nb.level(level)
        if p.goal:
            notes.goal = p.goal[:300]
        self.nb.update_hypotheses(
            level,
            p.confirmed[:6],
            p.refuted[:6],
            [HypothesisEntry(h.statement, h.kind, "open", h.confidence) for h in p.new_hypotheses[:4]],
        )
        self.charter.refine(p)
        self.charter.apply(self.ex)

    async def _execute(self, p: PlanOut) -> dict[str, Any]:
        """Run the plan's explicit actions through the explorer's bookkeeping."""
        assert self.obs is not None
        start_level = self._level()
        executed = changed = 0
        died = False
        log: list[str] = []
        avail = set(self.obs.available_actions)
        for spec in p.actions[: self.cfg.plan_max_actions]:
            for _ in range(spec.repeat):
                if not self._can_act():
                    break
                key = spec.key()
                if key[0] not in avail:
                    log.append(f"skip unavailable {key_name(key)}")
                    continue
                prev = self.obs.grid
                obs = await self._step(key)
                executed += 1
                if (prev != obs.grid).any():
                    changed += 1
                log.append(f"{key_name(key)} -> {self.nb.steps.lines[-1].result}")
                if obs.levels_completed > start_level:
                    return {"executed": executed, "changed": changed, "died": False, "level_up": True, "log": "\n".join(log)}
                if obs.state == GameState.GAME_OVER:
                    died = True
                    break
            if died or not self._can_act():
                break
        if died:
            obs = await self._step(RESET)
        return {"executed": executed, "changed": changed, "died": died, "level_up": False, "log": "\n".join(log)}
