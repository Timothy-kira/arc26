"""ARC-AGI-3 game as a stdio MCP server (no SDK: plain JSON-RPC 2.0, one message per line).

Any MCP-capable agent CLI (here: MiniMax Code ``mcode exec``) plays one game through two tools:
  arc_observe - the current frame as hex rows plus a PNG (free)
  arc_act     - send up to 20 actions; returns what changed after each

The engine runs in this process. Locally it plays the public games offline; in the Kaggle
rerun it joins the scorecard the batch parent opened on the competition gateway.
Progress is written to ``ARC_RESULT`` after every call, so a killed agent still leaves a score.

Environment:
  ARC_GAME        game id (prefix ok)                 ARC_RESULT   path of the progress JSON
  ARC_ENV_DIR     local environment_files dir         ARC_MAX_ACTIONS  action budget (default 2000)
  ARC_GATEWAY     gateway base URL (competition)      ARC_CARD_ID  scorecard id from the parent
  ARC_SOCKET      unix socket of a game daemon (``server.py --daemon``); the MCP server then only
                  forwards tool calls, so the game outlives any one agent process
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import socket
import sys
import time
from typing import Any, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arc_mcp.session import Obs, Session  # noqa: E402

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

HEX = "0123456789abcdef"
PALETTE = np.array(
    [
        (0xFF, 0xFF, 0xFF), (0xCC, 0xCC, 0xCC), (0x99, 0x99, 0x99), (0x66, 0x66, 0x66),
        (0x33, 0x33, 0x33), (0x00, 0x00, 0x00), (0xE5, 0x3A, 0xA3), (0xFF, 0x7B, 0xCC),
        (0xF9, 0x3C, 0x31), (0x1E, 0x93, 0xFF), (0x88, 0xD8, 0xF1), (0xFF, 0xDC, 0x00),
        (0xFF, 0x85, 0x1B), (0x92, 0x12, 0x31), (0x4F, 0xCC, 0x30), (0xA3, 0x56, 0xD6),
    ],
    dtype=np.uint8,
)
MAX_BATCH = 20


def grid_hex(grid: np.ndarray) -> str:
    """One line per row ("NN hex"); runs of identical rows collapse to "NN-MM hex"."""
    rows = ["".join(HEX[int(v)] for v in row) for row in grid]
    out, r = [], 0
    while r < len(rows):
        e = r
        while e + 1 < len(rows) and rows[e + 1] == rows[r]:
            e += 1
        out.append((f"{r:02d}-{e:02d} " if e > r else f"{r:02d} ") + rows[r])
        r = e + 1
    return "\n".join(out)


def patch_hex(prev: np.ndarray, cur: np.ndarray, max_cells: int = 600) -> str:
    """The changed bounding box of ``cur`` (rows with a column offset), or "" if none / too big."""
    changed = prev != cur
    if not changed.any():
        return ""
    rows = np.where(changed.any(axis=1))[0]
    cols = np.where(changed.any(axis=0))[0]
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
    if (r1 - r0 + 1) * (c1 - c0 + 1) > max_cells:
        return ""
    body = "\n".join(f"{r:02d} " + "".join(HEX[int(v)] for v in cur[r, c0:c1 + 1]) for r in range(r0, r1 + 1))
    return f"new cells in rows {r0}-{r1}, starting at column {c0}:\n{body}"


def grid_png_b64(grid: np.ndarray, scale: int = 4) -> str:
    from PIL import Image

    rgb = PALETTE[np.clip(grid, 0, 15).astype(np.int64)]
    img = Image.fromarray(rgb, "RGB").resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def change_summary(prev: np.ndarray, cur: np.ndarray) -> str:
    changed = prev != cur
    n = int(changed.sum())
    if n == 0:
        return "no cells changed"
    rows = np.where(changed.any(axis=1))[0]
    cols = np.where(changed.any(axis=0))[0]
    return f"{n} cells changed within rows {rows[0]}-{rows[-1]}, columns {cols[0]}-{cols[-1]}"


class Game:
    """One game, played through ``GameMemory`` (ledger, rules, todo), exposed as MCP tools."""

    def __init__(self) -> None:
        from arc_agi import Arcade, OperationMode

        from arc_mcp.memory import GameMemory

        quiet = logging.getLogger("arc.engine")
        gateway = os.getenv("ARC_GATEWAY")
        if gateway:
            self.arcade = Arcade(arc_api_key=os.getenv("ARC_API_KEY", "test-key-123"), arc_base_url=gateway.rstrip("/"),
                                 operation_mode=OperationMode.ONLINE, logger=quiet)
            self.card_id = os.environ["ARC_CARD_ID"]
        else:
            self.arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=os.getenv("ARC_ENV_DIR", "environment_files"),
                                 logger=quiet)
            self.card_id = self.arcade.open_scorecard(tags=["arc26"])
        want = os.environ["ARC_GAME"]
        envs = {e.game_id: e for e in self.arcade.get_environments()}
        self.game_id = next((g for g in envs if g == want or g.startswith(want)), want)
        baseline = None
        if not gateway:  # offline only: the human per-level baseline from metadata
            try:
                baseline = list(getattr(envs[self.game_id], "baseline_actions", None) or []) or None
            except KeyError:
                pass
        self.max_actions = int(os.getenv("ARC_MAX_ACTIONS", "100000"))
        wrapper = self.arcade.make(self.game_id, scorecard_id=self.card_id)
        ledger = os.getenv("ARC_LEDGER") or (os.path.join(os.path.dirname(os.getenv("ARC_RESULT", "")), "ledger.json")
                                             if os.getenv("ARC_RESULT") else None)
        self.mem = GameMemory(Session(wrapper), path=ledger, baseline=baseline)
        self.t0 = time.time()

    @property
    def finished(self) -> bool:
        return self.mem.obs.state.name == "WIN" or self.mem.session.actions >= self.max_actions

    def head(self) -> str:
        text = self.mem.status()
        if self.finished:
            text += "\nThe game is won. Stop playing now." if self.mem.obs.state.name == "WIN" else \
                "\nThe action budget is used up. Stop playing now."
        return text

    def record(self) -> None:
        path = os.getenv("ARC_RESULT")
        if not path:
            return
        s = self.mem.session
        out: dict[str, Any] = {"game_id": self.game_id, "levels_completed": self.mem.obs.levels_completed,
                               "win_levels": s.win_levels, "actions": s.actions, "level_actions": s.level_actions,
                               "state": self.mem.obs.state.name, "elapsed_s": round(time.time() - self.t0, 1)}
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

    # ----------------------------------------------------------------- tools

    def observe(self, args: dict[str, Any]) -> str:
        from arc_mcp.perception import describe_scene

        g = self.mem.obs.grid
        parts = [self.head(), describe_scene(g, max_objects=int(args.get("max_objects", 30)), mask=self.mem.mask)]
        if self.mem.mask is not None:
            rows = np.where(self.mem.mask.any(axis=1))[0]
            parts.append(f"HUD / counter area (ignored by the explorer): rows {rows[0]}-{rows[-1]}")
        if args.get("grid"):
            parts.append("grid (row: hex colours; NN-MM = identical rows):\n" + grid_hex(g))
        return "\n".join(parts)

    def act(self, args: dict[str, Any]) -> str:
        from arc_mcp.perception import diff

        lines = []
        start = self.mem.obs.grid.copy()
        for i, a in enumerate(list(args.get("actions") or [])[:MAX_BATCH]):
            if self.finished:
                break
            aid = int(a.get("action", 0))
            key = (6, int(a.get("x", 0)), int(a.get("y", 0))) if aid == 6 else (aid,)
            label = "RESET" if aid == 0 else (f"CLICK({key[1]},{key[2]})" if aid == 6 else f"ACTION{aid}")
            prev = self.mem.obs.grid.copy()
            obs, note = self.mem.step(key, "agent")
            what = note or diff(prev, obs.grid, self.mem.mask).describe(limit=4)
            lines.append(f"{i + 1}. {label}: {what}")
            if "LEVEL" in note or obs.state.name == "GAME_OVER":
                break
        patch = patch_hex(start, self.mem.obs.grid)
        return "\n".join([self.head()] + lines + ([patch] if patch else []))

    def explore(self, args: dict[str, Any]) -> str:
        budget = max(1, min(200, int(args.get("budget", 30))))
        budget = min(budget, self.max_actions - self.mem.session.actions)
        out = self.mem.explore(budget) if budget > 0 else "no action budget left"
        return self.head() + "\n" + out

    def model(self, args: dict[str, Any]) -> str:
        m = self.mem
        parts = [self.head(), "YOUR NOTEBOOK (kept verbatim by the server; edit with arc_note):", m.notebook_text(),
                 "RULES (learned from play; counts are evidence):", m.model.describe(m.obs.available_actions)]
        if m.notes:
            parts += ["YOUR HYPOTHESES:"] + m.notes[-6:]
        parts += ["TODO (top 5; arc_todo to manage):", m.todo_text(5)]
        if m.events:
            parts += ["RECENT EVENTS:"] + m.events[-4:]
        return "\n".join(parts)

    def hypothesize(self, args: dict[str, Any]) -> str:
        claim = str(args.get("claim", "")).strip()
        if not claim:
            return "give a claim, e.g. {\"claim\": \"yellow#11 blocks the avatar\", \"kind\": \"rule\"}"
        kind_ = str(args.get("kind", "rule"))
        rules = self.mem.model.describe(self.mem.obs.available_actions).splitlines()
        words = [w.strip(".,:;()").lower() for w in claim.split() if len(w) > 3]
        evidence = [r for r in rules if any(w in r.lower() for w in words)]
        self.mem.notes.append(f"[{kind_}] {claim}  (level {self.mem.obs.levels_completed + 1})")
        if kind_ == "goal" and args.get("plan"):
            self.mem.todo_add(f"test goal: {claim} -> arc_plan {json.dumps(args['plan'])}")
        self.mem.save()
        return "\n".join([self.head(), f"noted [{kind_}]: {claim}", "related evidence:"] + (evidence[:6] or ["(none yet: test it)"]))

    def plan(self, args: dict[str, Any]) -> str:
        from arc_mcp.planner import plan

        goal = args.get("goal") or {}
        if isinstance(goal, str):
            goal = json.loads(goal)
        n = max(1, min(200, int(args.get("max_actions", 60))))
        return self.head() + "\n" + plan(self.mem, goal, n)

    def note(self, args: dict[str, Any]) -> str:
        msg = self.mem.note(str(args.get("section", "")), str(args.get("text", "")), str(args.get("mode", "replace")))
        return self.head() + "\n" + msg

    def todo(self, args: dict[str, Any]) -> str:
        op = str(args.get("op", "list"))
        if op == "add":
            t = self.mem.todo_add(str(args.get("text", "")), int(args.get("cost", 1)))
            msg = f"added [{t.id}]"
        elif op in ("done", "drop"):
            msg = "ok" if self.mem.todo_done(int(args.get("id", -1)), drop=op == "drop") else "no such id"
        else:
            msg = ""
        return "\n".join(x for x in [self.head(), msg, self.mem.todo_text(12)] if x)


TOOL_DEFS = [
    ("arc_observe", "Objects on screen (colour, size, position, click centre), the HUD area and the status line. "
                    "grid=true adds the full 64x64 hex grid (costly: use rarely). Free: costs no game actions.",
     {"grid": {"type": "boolean"}, "max_objects": {"type": "integer", "minimum": 5, "maximum": 80}}, []),
    ("arc_act", "Send up to 20 actions in order. Each is {\"action\": 0-7, \"x\": col, \"y\": row}: 0=RESET (only "
                "after GAME_OVER), 1-4 directions (their effect differs per game), 5=interact, 6=click at (x, y), "
                "7=undo. Every action counts against the score. Stops early on level-up or GAME_OVER. Returns what "
                "changed after each action.",
     {"actions": {"type": "array", "minItems": 1, "maxItems": MAX_BATCH, "items": {"type": "object", "properties": {
         "action": {"type": "integer", "minimum": 0, "maximum": 7}, "x": {"type": "integer", "minimum": 0, "maximum": 63},
         "y": {"type": "integer", "minimum": 0, "maximum": 63}}, "required": ["action"]}}}, ["actions"]),
    ("arc_explore", "Let the built-in explorer play up to `budget` actions (default 30): it tries untried actions and "
                    "never-clicked object kinds, then walks to unexplored states, learning rules as it goes. Stops at a "
                    "level-up. Returns the new rules it found. Cheap in thinking, but its actions count too.",
     {"budget": {"type": "integer", "minimum": 1, "maximum": 200}}, []),
    ("arc_model", "What is known so far: learned rules with evidence (avatar and how each action moves it, what blocks "
                  "it, what clicks do, what caused GAME_OVER, what won earlier levels), your hypotheses, the todo list "
                  "and recent events. Free.", {}, []),
    ("arc_hypothesize", "Record a hypothesis (kind rule | goal | label), e.g. 'green#14 is the exit' or 'clicking "
                        "turns blue#9 cells red'. Returns the related evidence. Optional `plan` = a goal for arc_plan "
                        "to test it later (added to the todo).",
     {"claim": {"type": "string"}, "kind": {"type": "string", "enum": ["rule", "goal", "label"]},
      "plan": {"type": "object"}}, ["claim"]),
    ("arc_plan", "Reach a goal with the learned rules, executing and re-planning when a prediction fails. Goals: "
                 "{\"reach\": {\"color\": c}} (walk the avatar onto that colour), {\"reach\": {\"x\": X, \"y\": Y}}, "
                 "{\"click_all\": {\"color\": c}}; add \"avoid\": [colours] to never step on them. Uses the fewest "
                 "actions it can find.",
     {"goal": {"type": "object"}, "max_actions": {"type": "integer", "minimum": 1, "maximum": 200}}, ["goal"]),
    ("arc_note", "Your notebook for this game, stored by the server and shown verbatim by arc_model and at every "
                 "restart, so it survives context compaction. Sections: rules (confirmed mechanics), goal (what wins a "
                 "level), levels (one line per level: what worked, actions used), plan (next steps). mode=replace "
                 "(default) or append. Keep each section short (max 1500 characters).",
     {"section": {"type": "string", "enum": ["rules", "goal", "levels", "plan"]}, "text": {"type": "string"},
      "mode": {"type": "string", "enum": ["replace", "append"]}}, ["section", "text"]),
    ("arc_todo", "The game's todo list (auto items from the explorer plus yours): op=list | add (text, cost) | done "
                 "(id) | drop (id). Survives restarts.",
     {"op": {"type": "string", "enum": ["list", "add", "done", "drop"]}, "text": {"type": "string"},
      "id": {"type": "integer"}, "cost": {"type": "integer"}}, []),
]
TOOLS = [{"name": n, "description": d, "inputSchema": {"type": "object", "properties": p, **({"required": r} if r else {})}}
         for n, d, p, r in TOOL_DEFS]


def call_tool(game: "Game", name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    fn = {"arc_observe": game.observe, "arc_act": game.act, "arc_explore": game.explore, "arc_model": game.model,
          "arc_hypothesize": game.hypothesize, "arc_plan": game.plan, "arc_todo": game.todo,
          "arc_note": game.note}.get(name)
    if fn is None:
        raise ValueError(f"unknown tool {name}")
    text = fn(args)
    game.record()
    return [{"type": "text", "text": text}]


def daemon(path: str) -> None:
    """Own one game and serve tool calls on a unix socket, one JSON line per request and reply."""
    game = Game()
    game.record()
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(8)
    while True:
        conn, _ = srv.accept()
        with conn, conn.makefile("rw") as f:
            for line in f:
                try:
                    req = json.loads(line)
                    reply = {"content": call_tool(game, req.get("name"), req.get("arguments") or {}), "isError": False}
                except Exception as exc:
                    reply = {"content": [{"type": "text", "text": f"error: {type(exc).__name__}: {exc}"}], "isError": True}
                f.write(json.dumps(reply) + "\n")
                f.flush()


class Remote:
    """Forwards tool calls to the game daemon at ``path``."""

    def __init__(self, path: str) -> None:
        self.path, self.f = path, None

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(2):
            try:
                if self.f is None:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(self.path)
                    self.f = s.makefile("rw")
                self.f.write(json.dumps({"name": name, "arguments": args}) + "\n")
                self.f.flush()
                line = self.f.readline()
                if line:
                    return json.loads(line)
            except OSError:
                pass
            self.f = None
        raise RuntimeError("game daemon unreachable")


def main() -> None:
    # The engine logs to stdout; keep fd 1 for JSON-RPC only and send everything else to stderr.
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    game: Optional[Game] = None
    remote = Remote(os.environ["ARC_SOCKET"]) if os.getenv("ARC_SOCKET") else None

    def reply(msg_id: Any, result: Any = None, error: Optional[dict] = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
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
        if msg_id is None:  # notification (e.g. notifications/initialized)
            continue
        try:
            if method == "initialize":
                reply(msg_id, {
                    "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "arc-agi-3", "version": "0.1.0"},
                })
            elif method == "ping":
                reply(msg_id, {})
            elif method == "tools/list":
                reply(msg_id, {"tools": TOOLS})
            elif method == "tools/call":
                name, args = params.get("name"), params.get("arguments") or {}
                if remote is not None:
                    reply(msg_id, remote.call(name, args))
                    continue
                if game is None:
                    game = Game()
                reply(msg_id, {"content": call_tool(game, name, args), "isError": False})
            elif method in ("resources/list", "prompts/list"):
                reply(msg_id, {method.split("/")[0]: []})
            else:
                reply(msg_id, error={"code": -32601, "message": f"method not found: {method}"})
        except Exception as exc:  # report tool failures to the model instead of dying
            if method == "tools/call":
                reply(msg_id, {"content": [{"type": "text", "text": f"error: {type(exc).__name__}: {exc}"}], "isError": True})
            else:
                reply(msg_id, error={"code": -32603, "message": str(exc)})


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--daemon":
        daemon(sys.argv[2])
    else:
        main()
