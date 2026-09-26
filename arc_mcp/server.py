"""ARC-AGI-3 game for MCP-capable agent CLIs: a live Python REPL over the game, a notebook, a DAG.

Processes per game attempt (started by ``arc_runner/batch.py``):
  game daemon   ``server.py --daemon <game.sock>``: owns the engine and the ledger (scorecard step
                counts, notebook, events -> ledger.json); serves obs / step / note / status.
  kernel        ``python -m arc_mcp.kernel <kernel.sock> <game.sock>``: the REPL, no engine inside.
  MCP server    ``server.py`` (stdio, spawned by the agent CLI, may restart any time): the tools
                arc_python -> kernel, arc_note -> daemon, arc_dag -> kernel.

Environment:
  ARC_GAME / ARC_ENV_DIR     game id (prefix ok) / local environment_files dir (offline)
  ARC_GATEWAY / ARC_CARD_ID  competition gateway and the scorecard id from the parent
  ARC_RESULT / ARC_LEDGER    progress JSON / ledger JSON paths
  ARC_SOCKET / ARC_KERNEL    daemon and kernel sockets (MCP server side)
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arc_mcp.session import Session  # noqa: E402

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


# ---------------------------------------------------------------------- game daemon

class GameHost:
    def __init__(self) -> None:
        from arc_agi import Arcade, OperationMode

        from arc_mcp.ledger import GameLedger

        quiet = logging.getLogger("arc.engine")
        gateway = os.getenv("ARC_GATEWAY")
        if gateway:
            self.arcade = Arcade(arc_api_key=os.getenv("ARC_API_KEY", "test-key-123"), arc_base_url=gateway.rstrip("/"),
                                 operation_mode=OperationMode.ONLINE, logger=quiet)
            self.card_id = os.environ["ARC_CARD_ID"]
        else:
            self.arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                                 environments_dir=os.getenv("ARC_ENV_DIR", "environment_files"), logger=quiet)
            self.card_id = self.arcade.open_scorecard(tags=["arc26"])
        want = os.environ["ARC_GAME"]
        envs = {e.game_id: e for e in self.arcade.get_environments()}
        self.game_id = next((g for g in envs if g == want or g.startswith(want)), want)
        baseline = None
        if not gateway:  # offline only: human per-level baselines, for the level-score note
            baseline = list(getattr(envs.get(self.game_id), "baseline_actions", None) or []) or None
        self.max_actions = int(os.getenv("ARC_MAX_ACTIONS", "100000"))
        wrapper = self.arcade.make(self.game_id, scorecard_id=self.card_id)
        ledger = os.getenv("ARC_LEDGER") or (os.path.join(os.path.dirname(os.environ["ARC_RESULT"]), "ledger.json")
                                             if os.getenv("ARC_RESULT") else None)
        self.led = GameLedger(Session(wrapper), path=ledger, baseline=baseline)
        self.t0 = time.time()
        self.record()

    def status(self) -> str:
        text = self.led.status()
        if self.led.obs.state.name == "WIN":
            text += "\nThe game is won. Stop playing now."
        elif self.led.session.actions >= self.max_actions:
            text += "\nThe action budget is used up. Stop playing now."
        return text

    def obs(self, note: str = "", counted: int = 0) -> dict[str, Any]:
        o, s = self.led.obs, self.led.session
        return {"grid": o.grid.tolist(), "frames": [f.tolist() for f in o.frames[-8:]], "state": o.state.name,
                "level": o.levels_completed, "win_levels": s.win_levels, "available": o.available_actions,
                "actions": s.actions, "level_so_far": s.level_so_far, "note": note, "counted": counted,
                "status": self.status()}

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        op = req.get("op")
        if op == "obs":
            return self.obs()
        if op == "step":
            if self.led.obs.state.name == "WIN":
                return {"error": "the game is won; stop playing"}
            if self.led.session.actions >= self.max_actions:
                return {"error": "the action budget is used up"}
            before = self.led.session.actions
            _, note = self.led.step(int(req.get("action", 0)), req.get("x"), req.get("y"), str(req.get("source", "agent")))
            self.record()
            return self.obs(note, self.led.session.actions - before)
        if op == "note":
            return {"text": self.status() + "\n" + self.led.note(str(req.get("section", "")), str(req.get("text", "")),
                                                                  str(req.get("mode", "replace")))}
        if op == "notebook":
            return {"text": self.status() + "\nNOTEBOOK (kept verbatim by the server):\n" + self.led.notebook_text()}
        if op == "status":
            return {"text": self.status()}
        return {"error": f"unknown op {op}"}

    def record(self) -> None:
        path = os.getenv("ARC_RESULT")
        if not path:
            return
        s, o = self.led.session, self.led.obs
        out: dict[str, Any] = {"game_id": self.game_id, "levels_completed": o.levels_completed, "win_levels": s.win_levels,
                               "actions": s.actions, "level_actions": s.level_actions, "state": o.state.name,
                               "elapsed_s": round(time.time() - self.t0, 1)}
        if not os.getenv("ARC_GATEWAY"):  # the gateway hides scores
            try:
                card = self.arcade.get_scorecard(self.card_id).model_dump()
                env = next(e for e in card.get("environments", []) if e.get("id") == self.game_id)
                out["score"] = float(env.get("score") or 0.0)
            except Exception:
                pass
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, path)


def serve(path: str, handler) -> None:
    """One thread per connection (the kernel keeps a connection open while the MCP server makes
    short ones); the handler runs under a lock, so game operations stay strictly sequential."""
    import threading

    lock = threading.Lock()
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(16)

    def client(conn) -> None:
        with conn, conn.makefile("rw") as f:
            for line in f:
                try:
                    with lock:
                        reply = handler(json.loads(line))
                except Exception as exc:
                    reply = {"error": f"{type(exc).__name__}: {exc}"}
                f.write(json.dumps(reply) + "\n")
                f.flush()

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=client, args=(conn,), daemon=True).start()


def daemon(path: str) -> None:
    host = GameHost()
    serve(path, host.handle)


# ---------------------------------------------------------------------- MCP front end

def request(path: str, req: dict[str, Any], timeout: float = 900) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(path)
        f = s.makefile("rw")
        f.write(json.dumps(req) + "\n")
        f.flush()
        line = f.readline()
    if not line:
        raise RuntimeError("no reply")
    return json.loads(line)


PYTHON_DOC = (
    "Run Python in this game's live REPL (variables persist across calls and across restarts of the agent). "
    "The game state is in variables: grid (np.ndarray 64x64, grid[y, x], colours 0-15), prev_grid (frame before the last "
    "action), frames (animation frames of the last action), state ('NOT_FINISHED'|'GAME_OVER'|'WIN'), level "
    "(levels completed), win_levels, available (action ids usable now), actions_used, level_actions_used, history "
    "(one dict per action). Functions: act(a, x=None, y=None) plays one action and returns {changed, level_up, "
    "game_over, note, ...} (0 RESET only after GAME_OVER, 1-4 directions (their meaning differs per game), "
    "5 interact, 6 click at (x, y), 7 undo); show(g=None, y0, y1, x0, x1) -> hex text of a region; changes(a, b) -> "
    "[(y, x, old, new)]; objects(g=None) -> connected components [{color, size, bbox, center}]; anim() -> diff "
    "between the animation frames of the last action; action_stats(level=None) -> what each action id did so far (moves "
    "with vectors, no-ops): check it before re-testing a direction; regions(a, b) -> changed regions between two frames "
    "(several regions = side effects: a counter, a key, a door elsewhere); cells(step=None, ox=None, oy=None) -> the frame as a map of lattice cells (dominant colour per cell, lattice inferred from how far actions moved objects; array in lattice_cells), use it for maps and path search instead of decoding pixels; deaths(level=None) -> every GAME_OVER so far with what the dying action moved and the budget meter state (read it before trying again: never repeat a death; act() refuses, without spending an action, an action that already killed you from exactly the same frame); a 'meter on ...' line in the perception = a bar that changes with every action, usually the move budget of the level; look() attaches an image; grid, prev_grid and frames are copies (grid is reset to the current frame after every cell); journal / dag() = this game's live "
    "record of RIGHT and WRONG expectations and the DAG; np is numpy. After every cell that plays actions the "
    "reply carries the perception block: a 4x image of the frame, the objects and the animation diff. Write your own helpers (simulators, BFS, solvers) and reuse them; do not inspect or "
    "rewire the REPL itself (its built-ins are restored automatically if you overwrite them). "
    "Every action counts against the score: probe with few actions, then act with a plan. The last expression "
    "is printed. Each call becomes a node of your exploration DAG: purpose = the question it answers; parents = "
    "the node ids it builds on (default the previous node); expect = what you expect to happen, recorded next to "
    "what actually happened; revises = the node whose surprise this cell corrects. At most 300 actions and 120 s "
    "per call."
)

TOOLS = [
    {"name": "arc_python", "description": PYTHON_DOC, "inputSchema": {"type": "object", "properties": {
        "code": {"type": "string"}, "purpose": {"type": "string", "description": "one line: the question this cell answers"},
        "parents": {"type": "array", "items": {"type": "integer"}},
        "expect": {"type": "string", "description": "what you expect to happen (e.g. 'the avatar reaches the door and the level completes')"},
        "revises": {"type": "integer", "description": "id of the node whose surprise this cell corrects"},
        "check": {"type": "string", "description": "optional Python expression evaluated after the cell that decides RIGHT/WRONG, e.g. \"outcome['levels'] > 0\" or \"grid[30, 20] == 12\" (outcome has actions, levels, deaths, state, cells_changed, error)"}},
        "required": ["code", "purpose", "expect"]}},
    {"name": "arc_note", "description": (
        "Your notebook for this game, kept by the server and handed back verbatim after every restart or context "
        "compaction. Sections: rules (confirmed mechanics), goal (what wins a level), levels (one line per level: what "
        "worked, actions used), plan (next steps). mode=replace (default) or append; max 1500 characters per section. "
        "Call without text to read it."), "inputSchema": {"type": "object", "properties": {
        "section": {"type": "string", "enum": ["rules", "goal", "levels", "plan"]}, "text": {"type": "string"},
        "mode": {"type": "string", "enum": ["replace", "append"]}}}},
    {"name": "arc_dag", "description": (
        "Your exploration DAG so far: one line per REPL cell (id, parents, level, actions used, purpose; * = an event "
        "such as a level-up, ! = a surprise: an error, a death or an expected level-up that did not happen), with "
        "expected vs actual outcome and the list of surprises no cell has revised yet. node(i)['code'] in the REPL gives a cell's code; rerun(i) runs it "
        "again. Free."), "inputSchema": {"type": "object", "properties": {
        "last": {"type": "integer", "minimum": 1, "maximum": 60}}}},
]


def call_tool(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    """MCP content for a tool call: text, plus the 4x frame image after acting cells."""
    game, kernel = os.environ["ARC_SOCKET"], os.environ["ARC_KERNEL"]
    if name == "arc_python":
        r = request(kernel, {"code": args.get("code", ""), "purpose": args.get("purpose", ""),
                             "parents": args.get("parents"), "expect": args.get("expect", ""),
                             "revises": args.get("revises"), "check": args.get("check", "")})
        return [{"type": "text", "text": r["text"]}] + [{"type": "image", "data": b, "mimeType": "image/png"}
                                                        for b in r.get("images") or []]
    return [{"type": "text", "text": _call_text(name, args, game, kernel)}]


def _call_text(name: str, args: dict[str, Any], game: str, kernel: str) -> str:
    if name == "arc_dag":
        return request(kernel, {"op": "dag", "last": int(args.get("last", 12))})["text"]
    if name == "arc_note":
        if not args.get("text"):
            return request(game, {"op": "notebook"})["text"]
        return request(game, {"op": "note", "section": args.get("section", ""), "text": args.get("text", ""),
                              "mode": args.get("mode", "replace")})["text"]
    raise ValueError(f"unknown tool {name}")


def main() -> None:
    # fd 1 carries JSON-RPC only; anything else a library prints goes to stderr
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def reply(msg_id: Any, result: Any = None, error: Optional[dict] = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}
        msg["error" if error is not None else "result"] = error if error is not None else result
        out.write(json.dumps(msg) + "\n")
        out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method, msg_id, params = req.get("method"), req.get("id"), req.get("params") or {}
        if msg_id is None:
            continue
        try:
            if method == "initialize":
                reply(msg_id, {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                               "capabilities": {"tools": {"listChanged": False}},
                               "serverInfo": {"name": "arc-agi-3", "version": "0.3.0"}})
            elif method == "ping":
                reply(msg_id, {})
            elif method == "tools/list":
                reply(msg_id, {"tools": TOOLS})
            elif method == "tools/call":
                content = call_tool(params.get("name"), params.get("arguments") or {})
                reply(msg_id, {"content": content, "isError": False})
            elif method in ("resources/list", "prompts/list"):
                reply(msg_id, {method.split("/")[0]: []})
            else:
                reply(msg_id, error={"code": -32601, "message": f"method not found: {method}"})
        except Exception as exc:
            if method == "tools/call":
                reply(msg_id, {"content": [{"type": "text", "text": f"error: {type(exc).__name__}: {exc}"}], "isError": True})
            else:
                reply(msg_id, error={"code": -32603, "message": str(exc)})


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--daemon":
        daemon(sys.argv[2])
    else:
        main()
