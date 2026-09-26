import json

from arc_harness.llm.mock import MockLLM
from arc_harness.memory.cases import Case, objective_quality
from arc_harness.memory.hub import MemoryHub
from arc_harness.memory.notebook import GameNotebook
from arc_harness.memory.retrieve import BM25, Hit, retrieve, rrf_fuse
from arc_harness.memory.skill_store import Skill, SkillStore


def test_skill_roundtrip_and_feedback_retires(tmp_path):
    store = SkillStore(tmp_path / "rt")
    s = store.save(Skill("Key and door", "pick up key then open door", "1. find key\n2. walk to door", ["kind:keyboard"]))
    again = SkillStore(tmp_path / "rt").get(s.id)
    assert again is not None and again.body.startswith("1. find key") and again.applies_to == ["kind:keyboard"]
    for _ in range(5):
        store.record_use(s.id, won=False)
    assert store.get(s.id).retired
    assert store.all() == []


def test_library_is_overridden_by_runtime(tmp_path):
    lib = SkillStore(tmp_path / "lib")
    lib.save(Skill("Maze", "d", "library body"))
    store = SkillStore(tmp_path / "rt", [tmp_path / "lib"])
    assert store.get("maze").origin == "library"
    s = store.get("maze")
    s.body = "runtime body"
    store.save(s)
    store.reload()
    assert store.get("maze").body == "runtime body"


def test_objective_quality():
    assert objective_quality(False, 100, 20, None) == 0.1
    assert objective_quality(True, 20, 20, None) == 1.0
    assert 0.4 < objective_quality(True, 200, 20, None) < 0.5


def test_bm25_and_rrf():
    bm = BM25([["key", "door"], ["click", "toggle"], ["key", "maze"]])
    s = bm.scores(["key"])
    assert s[1] == 0 and s[0] > 0 and s[2] > 0
    a = [Hit("skill", "x", "X", "", 1, "library"), Hit("skill", "y", "Y", "", 1, "library")]
    b = [Hit("skill", "y", "Y", "", 1, "runtime")]
    assert [h.id for h in rrf_fuse({"library": a, "runtime": b}, {}, 5)][0] == "y"


async def test_retrieve_and_distill_pipeline(tmp_path):
    def handler(messages, schema):
        title = (schema or {}).get("title")
        if title == "GateOut":
            return {"plan": "", "selected": ["avatar_maze"]}
        if title == "SkillOpsOut":
            return {"ops": [{"op": "add", "name": "Avatar maze", "description": "move avatar to goal", "applies_to": ["kind:keyboard"], "body": "1. probe arrows\n2. BFS to goal"}]}
        return {}

    llm = MockLLM(handler)
    hub = MemoryHub(tmp_path, llm, library_roots=[])
    case = Case("g1", 0, ["kind:keyboard", "hud:bar"], "reach the exit", "probe arrows then walk", "walls block", "won", 30, 0.9)
    hub.sedimenter.add_case(case)
    written = await hub.sedimenter.distill(case)
    assert [s.id for s in written] == ["avatar_maze"]
    assert written[0].source_case_ids == [case.id] and written[0].cluster_id == case.cluster_id
    hits = await retrieve(skills=hub.skills.all(), cases=hub.cases.cases, query="exit avatar", features=["kind:keyboard"], llm=llm, exclude_game="g2")
    assert [h.id for h in hits] == ["avatar_maze"]
    # a second similar case joins the same cluster
    c2 = Case("g2", 1, ["kind:keyboard", "hud:bar"], "reach the exit door", "walk", "", "won", 40, 0.8)
    hub.sedimenter.add_case(c2)
    assert c2.cluster_id == case.cluster_id
    assert json.loads((tmp_path / "clusters.json").read_text())[case.cluster_id]["case_ids"] == [case.id, c2.id]


def test_notebook_brief_tracks_hypotheses():
    nb = GameNotebook("g")
    nb.update_hypotheses(0, ["arrows move the blue block"], ["clicking does nothing"], [])
    b = nb.brief(0)
    assert "arrows move the blue block" in b and "clicking does nothing" in b
    for i in range(40):
        nb.steps.add("ACTION1", f"moved {i}")
    assert "earlier steps elided" in nb.steps.render(5)
