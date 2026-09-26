"""MiniMax Code plugin hooks for the ARC-AGI-3 harness (standard library only).

Hook I/O is the Claude-compatible command-hook contract: the event arrives as JSON on stdin
(hook_event_name, cwd, tool_name, tool_input, ...), the verdict leaves as JSON on stdout.

  context  SessionStart / PostCompact: status, notebook, journal and the last DAG nodes of this
           game, read verbatim from the workspace files, as additionalContext. Whatever the
           conversation lost to a restart or a compaction comes back unchanged.
  guard    PreToolUse: the game is played through the arc MCP tools only. Shell, file reading,
           search and web tools are denied; write/edit only inside .minimax/skills.
  record   PostToolUse: every non-REPL tool call (todowrite plans, notes, skill reads, goal calls)
           becomes an event in dag_events.jsonl, so the DAG shows the whole trajectory.
  stop     (not registered) blocking Stop leaves the turn in a settling phase where MiniMax Code
           rejects every further tool call ("Tool-result delivery seam is closed"); goal mode's own
           auto-continuation keeps the session going instead.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Tool names arrive in either MiniMax (bash, web_fetch) or Claude (Bash, WebFetch) spelling.
DENY = {"bash", "read", "grep", "glob", "webfetch", "websearch", "websitedeploy", "ls", "notebookedit"}
SKILL_EDIT = {"write", "edit", "multiedit"}


def norm(name: str) -> str:
    return name.split("__")[-1].replace("_", "").lower()


def load(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def out(obj) -> None:
    sys.stdout.write(json.dumps(obj))
    sys.stdout.flush()


def workspace(ev: dict) -> str:
    return ev.get("cwd") or os.getcwd()


def context_text(ws: str) -> str:
    led = load(os.path.join(ws, "ledger.json"), {})
    book = led.get("notebook") or {}
    notes = "\n".join(f"[{k}]\n{v}" for k, v in book.items() if v) or "(empty)"
    journal = load(os.path.join(ws, "dag_journal.json"), [])
    wrong = [j for j in journal if j.get("ok") is False][-6:]
    right = [j for j in journal if j.get("ok") is True][-4:]
    jr = "\n".join([f"WRONG n{j['node']} L{j['level']}: {j.get('expect') or j.get('purpose')} -> {j.get('why')}" for j in wrong]
                   + [f"RIGHT n{j['node']} L{j['level']}: {j.get('expect') or j.get('purpose')}" for j in right]) or "(empty)"
    nodes = load(os.path.join(ws, "dag.json"), [])[-8:]
    dag = "\n".join(f"[{n['id']}]{'!' if n.get('flag') else ''} {n.get('actions', 0)}a {n.get('purpose', '')[:90]}"
                    for n in nodes) or "(no cells yet)"
    return ("ARC GAME STATE (restored by the arc26 plugin; the REPL still holds your variables and helpers)\n"
            f"Status: {led.get('status', 'not started')}\n"
            f"Notebook (verbatim):\n{notes}\n"
            f"Journal (what was right and wrong so far):\n{jr}\n"
            f"Last REPL cells (arc_dag / dag() for more):\n{dag}")


def game_unfinished(ws: str) -> tuple[bool, str]:
    led = load(os.path.join(ws, "ledger.json"), {})
    if not led:
        return False, ""
    status = led.get("status", "")
    if led.get("state") == "WIN" or "Stop playing now" in status:
        return False, status
    deadline = float(os.environ.get("ARC_DEADLINE", "0") or 0)
    if deadline and deadline - time.time() < 60:
        return False, status
    return True, status


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        ev = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        ev = {}
    ws = workspace(ev)
    name = str(ev.get("tool_name") or "")
    if mode == "context":
        out({"hookSpecificOutput": {"hookEventName": ev.get("hook_event_name", "SessionStart"),
                                    "additionalContext": context_text(ws)}})
    elif mode == "guard":
        short = norm(name)
        if short in DENY:
            out({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                        "permissionDecisionReason": f"{short} is not available in this game: play and "
                                        "compute through arc_python (the REPL), keep notes with arc_note."}})
        elif short in SKILL_EDIT:
            path = str((ev.get("tool_input") or {}).get("file_path") or (ev.get("tool_input") or {}).get("path") or "")
            if ".minimax/skills/" not in path.replace("\\", "/"):
                out({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                            "permissionDecisionReason": "files may only be written under .minimax/skills/ "
                                            "(skills for later games); use arc_note for game notes."}})
    elif mode == "record":
        if not name.endswith("arc_python"):
            rec = {"t": round(time.time(), 1), "tool": name.replace("mcp__arc__", ""),
                   "input": json.dumps(ev.get("tool_input") or {}, ensure_ascii=False)[:600]}
            try:
                with open(os.path.join(ws, "dag_events.jsonl"), "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError:
                pass
    elif mode == "stop":
        unfinished, status = game_unfinished(ws)
        if unfinished and not ev.get("stop_hook_active"):
            out({"decision": "block", "reason": "The game is not finished (" + status.splitlines()[0][:160] + "). "
                 "Continue: call arc_python with the next experiment or your planned actions."})
    return 0


if __name__ == "__main__":
    sys.exit(main())
