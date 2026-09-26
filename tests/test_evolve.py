"""Evolver loop on a fake benchmark: a candidate that writes BOOST into a prompt scores higher."""

import json
from pathlib import Path

from arc_harness.llm.mock import MockLLM
from evolve.loop import Evolver
from evolve.state import RunSpec, RunState, TreeNode
from evolve.stats import paired, screen_cull
from evolve.workspace import check_candidate, materialize, overlay_diff, path_allowed

REPO = Path(__file__).resolve().parents[1]
GAMES = ["g1", "g2", "g3", "g4", "g5"]


class FakeEval:
    def __init__(self):
        self.calls = []

    def eval(self, node, games, k, tag):
        self.calls.append((node.id, tag, tuple(games), k))
        boost = any("BOOST" in t for t in node.overlay.values())
        return {g: [0.2 + 0.1 * i + (0.3 if boost else 0.0) + 0.01 * j for j in range(k)] for i, g in enumerate(games)}

    def trajectories(self, node, games, tag):
        return [(g, "the agent looped on no-op clicks") for g in games]


def llm_handler(messages, schema):
    title = (schema or {}).get("title")
    if title == "DiagnosisOut":
        return {"labels": [{"why": "W1_action_loop", "where": "prompts/", "dominant": True, "reasoning": "loops", "fix_hint": "ban repeats"}]}
    # designer turns (no schema): read once, then write, then done
    n_assistant = sum(1 for m in messages if m["role"] == "assistant")
    if n_assistant == 0:
        return json.dumps({"action": "read", "path": "prompts/plan.md"})
    if n_assistant == 1:
        return json.dumps({"action": "write", "path": "prompts/plan.md", "content": "BOOST: never repeat a no-op.\n"})
    return json.dumps({"action": "done", "summary": "added a no-repeat rule"})


async def test_evolver_promotes_better_candidate(tmp_path):
    spec = RunSpec(
        work_dir=str(tmp_path / "w"), repo_dir=str(REPO), train_games=GAMES[:4], test_games=GAMES[4:],
        k_confirm=2, anchor_size=2, max_rounds=2, patience=1, max_why_per_round=1, candidates_per_why=1,
    )
    ev = Evolver(spec, FakeEval(), MockLLM(llm_handler), GAMES)
    best = await ev.run()
    assert best.summary == "added a no-repeat rule"
    assert "BOOST" in best.overlay["prompts/plan.md"]
    assert [n.status for n in ev.lineage()] == ["promoted", "promoted"]
    # round 2 designs from the promoted parent; its candidate is no better so patience stops the run
    st = RunState(spec)
    assert st.meta["round"] == 2 and st.meta["no_improve"] == 1
    assert "BOOST" in overlay_diff(REPO, best.overlay)


def test_fingerprint_guard(tmp_path):
    spec = RunSpec(work_dir=str(tmp_path), train_games=["a"], test_games=["b"])
    RunState(spec).save()
    spec.k_confirm = 5
    try:
        RunState(spec)
    except RuntimeError as e:
        assert "fingerprint" in str(e)
    else:
        raise AssertionError("expected fingerprint refusal")


def test_whitelist_and_guards(tmp_path):
    assert path_allowed("prompts/plan.md", ["prompts/"])
    assert not path_allowed("arc_harness/scoring.py", ["arc_harness/"])
    assert not path_allowed("prompts/../evolve/loop.py", ["prompts/"])
    ws = materialize(REPO, {"prompts/plan.md": "solve ls20 by going left\n"}, tmp_path / "ws")
    assert "hard-codes" in check_candidate(ws, ["prompts/plan.md"], ["ls20"])
    ws = materialize(REPO, {"arc_harness/charter.py": "def broken(:\n"}, tmp_path / "ws2")
    assert "compile error" in check_candidate(ws, ["arc_harness/charter.py"], [])


def test_stats():
    base = {"a": [0.5, 0.52, 0.48], "b": [0.2, 0.2, 0.2]}
    culled, _ = screen_cull({"a": 0.1, "b": 0.0}, base, ["a", "b"], 1.5)
    assert culled
    culled, _ = screen_cull({"a": 0.5, "b": 0.2}, base, ["a", "b"], 1.5)
    assert not culled
    r = paired({"a": [0.9], "b": [0.5], "c": [0.4]}, {"a": [0.5], "b": [0.2], "c": [0.35]}, ["a", "b", "c"])
    assert r.mean_diff > 0 and r.z > 0 and r.wilcoxon_p < 0.2
