"""Event-driven DAG scheduler (Raven ``dag_runner.py::run_dag``, trimmed).

* The ready set is recomputed after every completion instead of walking a fixed
  topological order, so a node's dependents start as soon as it settles.
* A shared semaphore caps concurrent nodes; nodes that share an ``instance``
  (a stateful resource such as one game environment) run strictly one at a time.
* A node is only ``completed`` once the judge has ruled, so dependents never read
  a rejected output.
* Failures cascade to dependents as ``skipped``; there is no silent retry.
* ``replan`` stops this run and returns the successor spec to the caller, who
  runs it as a new graph ("graphs in series, not a cycle in one graph").
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional

from .graph import DagNode, DagSpec, DagValidationError, validate_and_order
from .render import render_prompt
from .verdict import ACCOMPLISHED, Adjudicator, Decision, Judge, Verdict, abandon_all

logger = logging.getLogger(__name__)

TERMINAL = {"completed", "failed", "skipped", "cancelled"}


@dataclass
class NodeContext:
    node: DagNode
    prompt: str
    upstream: dict[str, str]
    attempt: int = 1
    continuation: str = ""
    previous_output: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


NodeFn = Callable[[NodeContext], Awaitable[str]]


@dataclass
class DagResult:
    spec: DagSpec
    status: dict[str, str]
    outputs: dict[str, str]
    errors: dict[str, str]
    verdicts: dict[str, Verdict]
    attempts: dict[str, int]
    completion_order: list[str]
    replanned: Optional[DagSpec] = None
    replan_message: str = ""

    @property
    def ok(self) -> bool:
        return all(s == "completed" for s in self.status.values())

    def terminal_outputs(self) -> dict[str, str]:
        has_child = {d for n in self.spec.nodes for d in n.depends_on}
        return {n.id: self.outputs[n.id] for n in self.spec.nodes if n.id not in has_child and n.id in self.outputs}


class InstanceLocks:
    """Shared across runs so a resource stays serialized even across successive graphs."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def get(self, name: Optional[str]):
        return self._locks[name] if name else contextlib.nullcontext()


async def _judge_fail_open(judge: Optional[Judge], node: DagNode, output: str, timeout_s: float) -> Verdict:
    if judge is None:
        return ACCOMPLISHED
    try:
        return await asyncio.wait_for(judge(node, output), timeout=timeout_s)
    except Exception as exc:  # fail-open: a broken judge must not block the graph
        logger.warning("judge failed for %s (%s); treating as accomplished", node.id, exc)
        return ACCOMPLISHED


async def run_dag(
    spec: DagSpec,
    runners: Mapping[str, NodeFn],
    *,
    semaphore: Optional[asyncio.Semaphore] = None,
    locks: Optional[InstanceLocks] = None,
    judge: Optional[Judge] = None,
    adjudicate: Optional[Adjudicator] = None,
    prior_outputs: Optional[Mapping[str, str]] = None,
    used_ids: tuple[str, ...] | set[str] | list[str] = (),
    workdir: Optional[Path] = None,
    max_continuations: int = 2,
    judge_timeout_s: float = 180.0,
    cancel: Optional[asyncio.Event] = None,
    extras: Optional[dict[str, Any]] = None,
) -> DagResult:
    prior = dict(prior_outputs or {})
    validate_and_order(spec, roles=runners.keys(), used_ids=used_ids, completed=prior.keys())
    by_id = {n.id: n for n in spec.nodes}
    status = {n.id: "pending" for n in spec.nodes}
    outputs: dict[str, str] = {}
    errors: dict[str, str] = {}
    verdicts: dict[str, Verdict] = {}
    attempts: dict[str, int] = {n.id: 0 for n in spec.nodes}
    continuation: dict[str, str] = {}
    previous: dict[str, str] = {}
    order: list[str] = []
    locks = locks or InstanceLocks()
    adjudicate = adjudicate or abandon_all
    paths: dict[str, str] = {}
    result = DagResult(spec, status, outputs, errors, verdicts, attempts, order)

    def available(dep: str) -> bool:
        return dep in prior or status.get(dep) == "completed"

    def all_outputs() -> dict[str, str]:
        return {**prior, **outputs}

    def cascade() -> None:
        changed = True
        while changed:
            changed = False
            for nid, st in status.items():
                if st != "pending":
                    continue
                bad = [d for d in by_id[nid].depends_on if status.get(d) in ("failed", "skipped", "cancelled")]
                if bad:
                    status[nid] = "skipped"
                    errors[nid] = f"dependency {bad[0]} {status[bad[0]]}"
                    changed = True

    async def run_one(nid: str) -> tuple[str, Verdict]:
        node = by_id[nid]
        async with locks.get(node.instance):
            async with semaphore or contextlib.nullcontext():
                status[nid] = "running"
                attempts[nid] += 1
                ups = {d: all_outputs().get(d, "") for d in node.depends_on}
                prompt = render_prompt(node.prompt_template, all_outputs(), node.inputs, paths)
                ctx = NodeContext(
                    node,
                    prompt,
                    ups,
                    attempts[nid],
                    continuation.pop(nid, ""),
                    previous.get(nid, ""),
                    extras or {},
                )
                out = await runners[node.role](ctx)
        verdict = await _judge_fail_open(judge, node, out, judge_timeout_s)
        return out, verdict

    def write_output(nid: str, text: str) -> None:
        if workdir is None:
            return
        d = Path(workdir) / "nodes"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{nid}.out.md"
        p.write_text(text)
        paths[nid] = str(p)

    tasks: dict[asyncio.Task, str] = {}
    cancel_task: Optional[asyncio.Task] = asyncio.ensure_future(cancel.wait()) if cancel else None
    try:
        while True:
            cascade()
            ready = [
                nid
                for nid, st in status.items()
                if st == "pending" and all(available(d) for d in by_id[nid].depends_on)
            ]
            for nid in ready:
                status[nid] = "queued"
                tasks[asyncio.ensure_future(run_one(nid))] = nid
            if not tasks:
                break
            waitset = set(tasks) | ({cancel_task} if cancel_task else set())
            done, _ = await asyncio.wait(waitset, return_when=asyncio.FIRST_COMPLETED)
            if cancel_task is not None and cancel_task in done:
                for nid in status:
                    if status[nid] in ("running", "queued"):
                        status[nid] = "cancelled"
                    elif status[nid] == "pending":
                        status[nid] = "skipped"
                break
            for t in done:
                nid = tasks.pop(t)
                try:
                    out, verdict = t.result()
                except Exception as exc:
                    status[nid] = "failed"
                    errors[nid] = f"{type(exc).__name__}: {exc}"
                    logger.warning("node %s failed: %s", nid, errors[nid])
                    continue
                verdicts[nid] = verdict
                if verdict.ok:
                    status[nid] = "completed"
                    outputs[nid] = out
                    order.append(nid)
                    write_output(nid, out)
                    continue
                status[nid] = "exception"
                decision: Decision = await adjudicate(by_id[nid], out, verdict, attempts[nid])
                if decision.kind == "continue" and attempts[nid] <= max_continuations:
                    status[nid] = "pending"
                    continuation[nid] = decision.message
                    previous[nid] = out
                elif decision.kind == "replan" and decision.spec is not None:
                    try:
                        validate_and_order(
                            decision.spec,
                            roles=runners.keys(),
                            used_ids=set(used_ids) | set(by_id),
                            completed=set(prior) | {k for k, v in status.items() if v == "completed"},
                        )
                    except DagValidationError as exc:
                        status[nid] = "failed"
                        errors[nid] = f"replan rejected: {exc}"
                        continue
                    result.replanned = decision.spec
                    result.replan_message = decision.message
                    status[nid] = "failed"
                    errors[nid] = "superseded by replan"
                    break
                else:
                    status[nid] = "failed"
                    errors[nid] = f"abandoned: {decision.message or verdict.what_is_missing}"
            if result.replanned is not None:
                for t, nid in tasks.items():
                    t.cancel()
                    status[nid] = "cancelled"
                for nid, st in status.items():
                    if st == "pending":
                        status[nid] = "skipped"
                        errors[nid] = "superseded by replan"
                break
    finally:
        for t in list(tasks):
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if cancel_task is not None:
            cancel_task.cancel()
    return result
