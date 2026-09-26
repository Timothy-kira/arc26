"""Candidate designer: a bounded file-editing agent over the whitelisted paths.

Raven's AppWorld line uses a bash editor in a git worktree (<=22 turns, forced edit
after 4 read-only turns). Here the agent works on a materialized copy with four JSON
actions: list, read, write, done. Writes outside the whitelist are refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from arc_harness.llm.client import ChatModel
from arc_harness.llm.semantic import extract_json

from .workspace import path_allowed

SYSTEM = """You improve the harness of an ARC-AGI-3 game-playing agent (Qwen-driven). The agent explores games with a state-graph explorer and calls the LLM in reasoning rounds (analysts -> plan -> act). You may edit ONLY these paths: {whitelist}.

Work in turns. Each turn reply with ONLY one JSON object, one of:
{{"action": "list", "path": "<dir>"}}
{{"action": "read", "path": "<file>"}}
{{"action": "write", "path": "<file>", "content": "<complete new file text>"}}
{{"action": "done", "summary": "<what you changed and why>"}}

Rules:
- Make a small, targeted change aimed at the failure class below; one lever per candidate.
- The change must generalize to unseen games: never mention game ids, specific coordinates or colours of a particular game.
- Python edits must keep the module importable and its public names intact.
- Prefer prompt and config edits; code edits must be minimal.
"""


class EditAction(BaseModel):
    action: Literal["list", "read", "write", "done"]
    path: str = ""
    content: str = ""
    summary: str = ""


async def design_candidate(
    llm: ChatModel,
    ws: Path,
    whitelist: list[str],
    brief: str,
    history: str,
    distinct_from: list[str],
    max_turns: int = 22,
) -> tuple[dict[str, str], str]:
    listing = "\n".join(
        str(p.relative_to(ws)) for w in whitelist for p in sorted((ws / w).rglob("*") if (ws / w).is_dir() else [ws / w]) if p.is_file()
    )
    user = (
        f"{brief}\n\nPrevious attempts on this failure class:\n{history or '(none)'}\n\n"
        + (f"Other candidates this round already tried these levers (choose a different one): {distinct_from}\n\n" if distinct_from else "")
        + f"Editable files:\n{listing}"
    )
    messages = [{"role": "system", "content": SYSTEM.format(whitelist=", ".join(whitelist))}, {"role": "user", "content": user}]
    changes: dict[str, str] = {}
    read_only_streak = 0
    summary = ""
    for _ in range(max_turns):
        reply = await llm.chat(messages, thinking=False, max_tokens=6000)
        messages.append({"role": "assistant", "content": reply[-8000:]})
        try:
            act = EditAction.model_validate(extract_json(reply))
        except (ValueError, ValidationError) as exc:
            messages.append({"role": "user", "content": f"Invalid action ({str(exc)[:300]}). Reply with ONE JSON action."})
            continue
        if act.action == "done":
            summary = act.summary
            if changes:
                break
            messages.append({"role": "user", "content": "You have not written any change yet. Write one now."})
            continue
        if act.action == "write":
            if not path_allowed(act.path, whitelist):
                messages.append({"role": "user", "content": f"Refused: {act.path} is not editable."})
                continue
            p = ws / act.path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(act.content)
            changes[act.path.lstrip("./")] = act.content
            read_only_streak = 0
            messages.append({"role": "user", "content": f"Wrote {act.path} ({len(act.content)} chars). Continue or finish with done."})
            continue
        read_only_streak += 1
        p = ws / act.path
        if act.action == "list":
            items = sorted(str(x.relative_to(ws)) for x in p.rglob("*") if x.is_file())[:200] if p.is_dir() else []
            obs = "\n".join(items) or "(empty or not a directory)"
        else:
            obs = p.read_text()[:8000] if p.is_file() else "(no such file)"
        nudge = "\nYou have read enough; make your edit now." if read_only_streak >= 4 else ""
        messages.append({"role": "user", "content": f"{obs}{nudge}"})
    return changes, summary or json.dumps(sorted(changes))
