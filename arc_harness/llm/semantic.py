"""Schema-validated LLM calls with error-feedback retries.

Ported idea from Raven's ``evolver/orchestrator/nodes/semantic.py::SemanticNode``:
a weaker local model becomes dependable when every reply is parsed against a
schema and parse/validation errors are fed back for a bounded number of retries.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .client import ChatModel, user_message

T = TypeVar("T", bound=BaseModel)

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S | re.I)


class SemanticCallError(RuntimeError):
    pass


def extract_json(text: str) -> Any:
    text = _THINK.sub("", text).strip()
    m = _FENCE.search(text)
    if m:
        return json.loads(m.group(1))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON object in reply")
    s = min(starts)
    closer = "}" if text[s] == "{" else "]"
    e = text.rfind(closer)
    if e <= s:
        raise ValueError("unterminated JSON in reply")
    return json.loads(text[s : e + 1])


async def call_json(
    llm: ChatModel,
    schema: Type[T],
    system: str,
    user: str,
    *,
    images: Optional[list[str]] = None,
    thinking: bool = False,
    retries: int = 2,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> T:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        user_message(user, images),
    ]
    json_schema = schema.model_json_schema()
    last_err: Optional[str] = None
    for _ in range(retries + 1):
        reply = await llm.chat(
            messages,
            thinking=thinking,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            return schema.model_validate(extract_json(reply))
        except (ValueError, ValidationError) as exc:
            last_err = str(exc)[:800]
            messages.append({"role": "assistant", "content": reply[-4000:]})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your reply could not be parsed against the required JSON schema:\n"
                        f"{last_err}\nReply again with ONLY one JSON object matching:\n"
                        f"{json.dumps(json_schema)[:3000]}"
                    ),
                }
            )
    raise SemanticCallError(f"{schema.__name__}: {last_err}")
