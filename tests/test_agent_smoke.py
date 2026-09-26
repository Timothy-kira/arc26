"""End-to-end smoke test on a real local game with a scripted LLM (no GPU needed)."""

import json

import pytest

from arc_harness.agent import AgentConfig, GameAgent
from arc_harness.budget import GlobalBudget
from arc_harness.env.arcade import DEFAULT_ENV_DIR, ArcEnv
from arc_harness.llm.mock import MockLLM, schema_default_handler
from arc_harness.memory.hub import MemoryHub

pytestmark = pytest.mark.skipif(not DEFAULT_ENV_DIR.exists(), reason="competition data not downloaded")


async def test_agent_plays_with_mock_llm(tmp_path):
    calls = {"plan": 0}

    def handler(messages, schema):
        title = (schema or {}).get("title")
        if title == "PlanOut":
            calls["plan"] += 1
            return {"goal": "test", "actions": [{"action": 1, "repeat": 2}], "mode": "execute"}
        return json.loads(schema_default_handler(messages, schema))

    env = ArcEnv("offline")
    session = env.make("ls20")
    hub = MemoryHub(tmp_path / "mem", MockLLM(handler), library_roots=[])
    budget = GlobalBudget(600, 1, 1, reserve_seconds=0).for_game()
    agent = GameAgent(session, hub.sedimenter.llm, hub, budget, AgentConfig(max_actions=120, llm_rounds_per_level=3), tmp_path / "g")
    rep = await agent.play()
    assert rep.error == ""
    assert rep.actions == 120 and rep.llm_rounds == 3
    assert calls["plan"] >= 3
    assert (tmp_path / "g" / "notebook.md").exists()
    assert (tmp_path / "g" / "round_001" / "nodes" / "r1_plan.out.md").exists()
    assert hub.cases.cases, "level end should write a case"
