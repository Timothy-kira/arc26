"""OpenAI-compatible chat client (standard library only) for the agent loop.

Thinking is on and there is no output-token budget; instead each call has a wall-clock limit:
the reply is streamed and a call that runs past ``call_seconds`` (a runaway chain of thought) is cut; the
reasoning it produced is kept and the next call continues from it with the thinking closed, so
the model answers from what it has already worked out.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Reply:
    content: str
    reasoning: str
    usage: dict
    seconds: float
    attempts: int
    failures: tuple = ()


class CallTimeout(Exception):
    def __init__(self, msg: str, reasoning: str = "", content: str = "") -> None:
        super().__init__(msg)
        self.reasoning, self.content = reasoning, content


class LLM:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", thinking: bool = True,
                 call_seconds: float = 180.0, retries: int = 2, temperature: Optional[float] = None,
                 hurry: str = "") -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key, self.thinking = model, api_key, thinking
        self.call_seconds, self.retries, self.temperature = call_seconds, retries, temperature
        self.hurry = hurry  # appended to the messages of a retry, after an attempt was cut off
        self.continue_chars = 12000  # how much of a cut-off reasoning a continuation carries

    def _once(self, messages: list[dict], limit: float, thinking: bool, prefill: Optional[str] = None) -> Reply:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True,
                                "stream_options": {"include_usage": True},
                                "chat_template_kwargs": {"enable_thinking": thinking}}
        if prefill is not None:  # continue a cut-off reasoning: the reply starts with it, thinking closed
            body["messages"] = messages + [{"role": "assistant", "content": prefill}]
            body.update(continue_final_message=True, add_generation_prompt=False)
        if self.temperature is not None:
            body["temperature"] = self.temperature
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        t0 = time.time()
        content, reasoning, usage = [], [], {}
        with urllib.request.urlopen(req, timeout=120) as r:
            for raw in r:
                if time.time() - t0 > limit:
                    raise CallTimeout(f"call ran past {limit:.0f}s", "".join(reasoning), "".join(content))
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    r_ = delta.get("reasoning_content") or delta.get("reasoning")
                    if r_:
                        reasoning.append(r_)
        return Reply("".join(content), "".join(reasoning), usage, time.time() - t0, 1)

    def chat(self, messages: list[dict], deadline: Optional[float] = None, thinking: Optional[bool] = None) -> Reply:
        """One reply; each attempt is cut at ``call_seconds`` and nothing runs past ``deadline``. When an
        attempt is cut off mid-reasoning, the next one continues from that reasoning with the thinking
        closed, so the work already done turns into an answer instead of being thrown away; if that is
        not possible the retry starts over with ``hurry`` appended."""
        failures: list[str] = []
        partial = ""
        think = self.thinking if thinking is None else thinking
        for attempt in range(1, self.retries + 2):
            t0 = time.time()
            limit = self.call_seconds if deadline is None else min(self.call_seconds, deadline - t0)
            if limit < 5:
                failures.append("no time left before the deadline")
                break
            try:
                if partial:
                    # continue from the end of the cut-off reasoning (its conclusions are at the end; a very
                    # long prefill made the model return nothing)
                    tail = partial.strip()[-self.continue_chars:]
                    rep = self._once(messages, limit, False, prefill="<think>\n" + tail + "\n</think>\n\n")
                    rep.reasoning = partial + rep.reasoning
                else:
                    rep = self._once(messages if attempt == 1 or not self.hurry else messages + [
                        {"role": "user", "content": self.hurry}], limit, think)
                rep.attempts, rep.failures = attempt, tuple(failures)
                if rep.content.strip():
                    return rep
                failures.append(f"empty reply after {rep.seconds:.0f}s ({len(rep.reasoning)} reasoning chars)")
                partial = ""  # the next attempt starts over, with the hurry note
            except CallTimeout as exc:
                failures.append(f"CallTimeout after {time.time() - t0:.0f}s" + (" (continuing it)" if exc.reasoning and not partial else ""))
                partial = exc.reasoning if exc.reasoning and not partial and not exc.content else ""
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                failures.append(f"{type(exc).__name__} after {time.time() - t0:.0f}s: {exc}")
                partial = ""
            if not partial:
                time.sleep(min(30, 3 * attempt))
        raise RuntimeError(f"LLM call failed after {self.retries + 1} attempts: {'; '.join(failures)}")
