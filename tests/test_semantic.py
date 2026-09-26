from pydantic import BaseModel

from arc_harness.llm.mock import MockLLM
from arc_harness.llm.semantic import call_json, extract_json


class Out(BaseModel):
    action: int
    why: str


def test_extract_json_variants():
    assert extract_json('<think>hmm {"a":1}</think>```json\n{"action": 2, "why": "x"}\n```') == {"action": 2, "why": "x"}
    assert extract_json('sure: {"action": 3, "why": "y"} done') == {"action": 3, "why": "y"}


async def test_call_json_retries_with_error_feedback():
    replies = iter(["not json", '{"action": "left"}', '{"action": 4, "why": "ok"}'])
    llm = MockLLM(lambda m, s: next(replies))
    out = await call_json(llm, Out, "sys", "user", retries=2)
    assert out.action == 4
    assert len(llm.calls) == 3
    assert "could not be parsed" in llm.calls[-1][-1]["content"]
