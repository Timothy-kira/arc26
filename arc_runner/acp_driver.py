"""Drive MiniMax Code through ACP (``mcode acp``, JSON-RPC over stdio) with its Goal mode.

Headless ``mcode exec`` runs one turn and stops whenever the model ends its reply. Goal mode keeps
a session working toward an objective by itself (auto-continuation turns, a repeated-reply breaker,
budgets), but goals can only be created through the TUI ``/goal`` command or the ACP extension
``mcode/session/goal/create``. This driver therefore:

  initialize -> session/new (cwd = the game workspace, MCP servers) -> goal/create(objective)
  -> wait while the goal is active (MiniMax Code continues on its own) -> stop at WIN, at the
  deadline, or when the goal ends (complete / blocked / budget / breaker)

and answers the agent's permission requests with "allow". Standard library only.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional


class AcpClient:
    def __init__(self, cmd: list[str], env: dict[str, str], cwd: str, log_path: Optional[Path] = None) -> None:
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=open(log_path, "a") if log_path else subprocess.DEVNULL,
                                     env=env, cwd=cwd, text=True, bufsize=1)
        self.next_id = 0
        self.pending: dict[int, queue.Queue] = {}
        self.events: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.log = open(log_path.with_suffix(".jsonl"), "a") if log_path else None
        threading.Thread(target=self._reader, daemon=True).start()

    def _write(self, msg: dict[str, Any]) -> None:
        with self.lock:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()

    def _reader(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if self.log:
                self.log.write(line if line.endswith("\n") else line + "\n")
                self.log.flush()
            if "id" in msg and ("result" in msg or "error" in msg) and "method" not in msg:
                q = self.pending.pop(msg["id"], None)
                if q:
                    q.put(msg)
            elif "method" in msg and "id" in msg:  # a request from the agent to us
                self._answer(msg)
            elif "method" in msg:
                self.events.put(msg)
        self.events.put({"method": "__exit__"})

    def _answer(self, msg: dict[str, Any]) -> None:
        method, params = msg["method"], msg.get("params") or {}
        if method == "session/request_permission":
            opts = params.get("options") or []
            pick = next((o for o in opts if o.get("kind") in ("allow_always", "allow_once")), opts[0] if opts else None)
            result: Any = {"outcome": {"outcome": "selected", "optionId": pick["optionId"]}} if pick else \
                {"outcome": {"outcome": "cancelled"}}
            self._write({"jsonrpc": "2.0", "id": msg["id"], "result": result})
        else:
            self._write({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"unsupported: {method}"}})

    def request(self, method: str, params: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
        with self.lock:
            self.next_id += 1
            mid = self.next_id
        q: queue.Queue = queue.Queue()
        self.pending[mid] = q
        self._write({"jsonrpc": "2.0", "id": mid, "method": method, "params": params})
        msg = q.get(timeout=timeout)
        if "error" in msg:
            raise RuntimeError(f"{method}: {msg['error'].get('message')}")
        return msg["result"]

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(10)
        except Exception:
            self.proc.kill()


def run_goal(node: str, mcode: str, ws: Path, env: dict[str, str], mcp_servers: list[dict[str, Any]],
             objective: str, deadline: float, is_done: Callable[[], bool], log_path: Path,
             poll: float = 5.0, prefix: Optional[list[str]] = None) -> dict[str, Any]:
    """Run one goal-mode session until the game is done, the deadline passes or the goal ends.
    ``prefix`` (e.g. setpriv ...) runs MiniMax Code as an unprivileged user."""
    c = AcpClient([*(prefix or []), node, mcode, "acp"], env, str(ws), log_path)
    out: dict[str, Any] = {"driver": "acp"}
    try:
        c.request("initialize", {"protocolVersion": 1, "clientCapabilities": {
            "fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False}})
        sess = c.request("session/new", {"cwd": str(ws), "mcpServers": mcp_servers})
        sid = sess["sessionId"]
        out["session"] = sid
        goal = c.request("mcode/session/goal/create", {"sessionId": sid, "objective": objective})["goal"]
        out["goal_created"] = goal.get("status")
        status = goal.get("status")
        while time.monotonic() < deadline:
            if is_done():
                out["stop"] = "game finished"
                break
            try:
                ev = c.events.get(timeout=poll)
                if ev.get("method") == "__exit__":
                    out["stop"] = "mcode acp exited"
                    break
            except queue.Empty:
                pass
            try:
                g = c.request("mcode/session/goal/get", {"sessionId": sid}, timeout=60).get("goal") or {}
            except Exception as exc:
                out["stop"] = f"goal/get failed: {exc}"
                break
            status = g.get("status")
            out["tokens_used"], out["time_used_s"] = g.get("tokensUsed"), g.get("timeUsedSeconds")
            if status not in ("active", None):
                out["stop"] = f"goal {status}"
                break
        else:
            out["stop"] = "deadline"
        out["goal_status"] = status
        if out.get("stop") in ("deadline", "game finished"):
            try:
                c.request("mcode/session/goal/patch", {"sessionId": sid, "status": "paused"}, timeout=30)
                c._write({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": sid}})
            except Exception:
                pass
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        c.close()
    return out
