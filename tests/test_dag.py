import asyncio

import pytest

from arc_harness.dag.graph import DagNode, DagSpec, DagValidationError, validate_and_order
from arc_harness.dag.runner import InstanceLocks, run_dag
from arc_harness.dag.verdict import Decision, Verdict


def spec(*nodes):
    return DagSpec(nodes=[DagNode(**n) for n in nodes])


def test_validation_rejects_cycle_and_bad_refs():
    s = spec(
        {"id": "a", "role": "echo", "depends_on": ["b"]},
        {"id": "b", "role": "echo", "depends_on": ["a"]},
    )
    with pytest.raises(DagValidationError, match="cycle"):
        validate_and_order(s, roles={"echo"})
    s = spec({"id": "a", "role": "echo", "prompt_template": "{{ z.output }}"})
    with pytest.raises(DagValidationError, match="without depending"):
        validate_and_order(s, roles={"echo"})
    s = spec({"id": "a", "role": "echo", "inputs": {"k": 1}})
    with pytest.raises(DagValidationError, match="never referenced"):
        validate_and_order(s, roles={"echo"})
    with pytest.raises(DagValidationError, match="already used"):
        validate_and_order(spec({"id": "a", "role": "echo"}), roles={"echo"}, used_ids={"A"})


async def test_parallel_fanout_and_rendering():
    started: list[str] = []

    async def slow(ctx):
        started.append(ctx.node.id)
        await asyncio.sleep(0.05)
        return f"{ctx.node.id}:{ctx.prompt}"

    s = spec(
        {"id": "p", "role": "slow", "prompt_template": "x"},
        {"id": "h1", "role": "slow", "depends_on": ["p"], "prompt_template": "{{ p.output }}"},
        {"id": "h2", "role": "slow", "depends_on": ["p"]},
        {"id": "m", "role": "slow", "depends_on": ["h1", "h2"]},
    )
    t0 = asyncio.get_event_loop().time()
    r = await run_dag(s, {"slow": slow})
    dt = asyncio.get_event_loop().time() - t0
    assert r.ok and r.completion_order[0] == "p" and r.completion_order[-1] == "m"
    assert dt < 0.19  # h1 and h2 ran concurrently
    assert '<upstream-data node="p">' in r.outputs["h1"]
    assert set(r.terminal_outputs()) == {"m"}


async def test_instance_nodes_are_serialized():
    active = 0
    peak = 0

    async def act(ctx):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return "ok"

    s = spec(*({"id": f"a{i}", "role": "act", "instance": "env:g"} for i in range(4)))
    r = await run_dag(s, {"act": act}, locks=InstanceLocks())
    assert r.ok and peak == 1


async def test_failure_cascades():
    async def boom(ctx):
        raise RuntimeError("x")

    async def ok(ctx):
        return "ok"

    s = spec({"id": "a", "role": "boom"}, {"id": "b", "role": "ok", "depends_on": ["a"]}, {"id": "c", "role": "ok"})
    r = await run_dag(s, {"boom": boom, "ok": ok})
    assert r.status == {"a": "failed", "b": "skipped", "c": "completed"}


async def test_judge_continue_then_accept_and_fail_open():
    async def worker(ctx):
        return "good" if ctx.continuation else "bad"

    async def judge(node, out):
        if node.id == "broken":
            raise RuntimeError("judge down")
        return Verdict(outcome="accomplished" if out == "good" else "not_accomplished", what_is_missing="more")

    decisions = []

    async def adjudicate(node, out, verdict, attempt):
        decisions.append((node.id, attempt))
        return Decision.cont("try harder")

    s = spec({"id": "w", "role": "worker"}, {"id": "broken", "role": "worker"})
    r = await run_dag(s, {"worker": worker}, judge=judge, adjudicate=adjudicate)
    assert r.status == {"w": "completed", "broken": "completed"}
    assert r.outputs["w"] == "good" and r.outputs["broken"] == "bad"
    assert decisions == [("w", 1)]


async def test_replan_stops_run_and_returns_successor():
    async def worker(ctx):
        return "nope"

    async def judge(node, out):
        return Verdict(outcome="not_accomplished")

    successor = spec({"id": "b2", "role": "worker", "depends_on": ["a"]})

    async def adjudicate(node, out, verdict, attempt):
        return Decision.replan(successor, "hypothesis falsified")

    async def ok(ctx):
        return "a-done"

    s = spec({"id": "a", "role": "ok"}, {"id": "b", "role": "worker", "depends_on": ["a"]}, {"id": "c", "role": "ok", "depends_on": ["b"]})

    async def judge2(node, out):
        return Verdict() if node.id == "a" else await judge(node, out)

    r = await run_dag(s, {"worker": worker, "ok": ok}, judge=judge2, adjudicate=adjudicate)
    assert r.replanned is successor
    assert r.status["a"] == "completed" and r.status["c"] == "skipped"
    r2 = await run_dag(
        r.replanned, {"worker": ok, "ok": ok}, prior_outputs=r.outputs, used_ids=set(r.status)
    )
    assert r2.ok
