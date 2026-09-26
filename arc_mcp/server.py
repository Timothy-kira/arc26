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
    def __init__(self) -> None:
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction, GameState

        self.GameAction, self.GameState = GameAction, GameState
        quiet = logging.getLogger("arc.engine")
        gateway = os.getenv("ARC_GATEWAY")
        if gateway:
            self.arcade = Arcade(
                arc_api_key=os.getenv("ARC_API_KEY", "test-key-123"),
                arc_base_url=gateway.rstrip("/"),
                operation_mode=OperationMode.ONLINE,
                logger=quiet,
            )
            self.card_id = os.environ["ARC_CARD_ID"]
        else:
            self.arcade = Arcade(
                operation_mode=OperationMode.OFFLINE,
                environments_dir=os.getenv("ARC_ENV_DIR", "environment_files"),
                logger=quiet,
            )
            self.card_id = self.arcade.open_scorecard(tags=["arc26"])
        want = os.environ["ARC_GAME"]
        ids = [e.game_id for e in self.arcade.get_environments()]
        self.game_id = next((g for g in ids if g == want or g.startswith(want)), want)
        self.wrapper = self.arcade.make(self.game_id, scorecard_id=self.card_id)
        self.max_actions = int(os.getenv("ARC_MAX_ACTIONS", "2000"))
        self.session = Session(self.wrapper)
        self.t0 = time.time()
        self._sync(self.session.start())

    def _sync(self, obs: Obs) -> None:
        self.grid, self.frames_last = obs.grid, len(obs.frames)
        self.state, self.levels, self.available = obs.state, obs.levels_completed, obs.available_actions
        self.win_levels, self.actions = self.session.win_levels, self.session.actions

    def step(self, action: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Play one action; False if it was a RESET the competition rules ignore (level start)."""
        if int(action) == 0 and not self.session.can_reset():
            return False
        self._sync(self.session.step(action, x, y))
        return True

    @property
    def finished(self) -> bool:
        return self.state == self.GameState.WIN or self.actions >= self.max_actions

    def status(self) -> str:
        acts = ", ".join("RESET" if a == 0 else f"ACTION{a}" for a in self.available)
        return (
            f"state={self.state.name} levels_completed={self.levels}/{self.win_levels} "
            f"actions_used={self.actions}/{self.max_actions} available=[{acts}]"
        )

    def record(self) -> None:
        path = os.getenv("ARC_RESULT")
        if not path:
            return
        out: dict[str, Any] = {
            "game_id": self.game_id,
            "levels_completed": self.levels,
            "win_levels": self.win_levels,
            "actions": self.actions,
            "state": self.state.name,
            "elapsed_s": round(time.time() - self.t0, 1),
        }
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

    def observe(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"type": "text", "text": f"{self.status()}\n{grid_hex(self.grid)}"}]
        # Off by default: many OpenAI-compatible servers reject images inside tool-result messages.
        if os.getenv("ARC_IMAGE") == "1":
            out.append({"type": "image", "data": grid_png_b64(self.grid), "mimeType": "image/png"})
        return out

    def act(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        lines = []
        start_grid = self.grid.copy()
        for i, a in enumerate(actions[:MAX_BATCH]):
            if self.finished:
                break
            prev_grid, prev_levels = self.grid.copy(), self.levels
            aid = int(a.get("action", 0))
            label = "RESET" if aid == 0 else (f"ACTION6({a.get('x')},{a.get('y')})" if aid == 6 else f"ACTION{aid}")
            if not self.step(aid, a.get("x"), a.get("y")):
                lines.append(f"{i + 1}. RESET: ignored (the level has just started; RESET only helps after GAME_OVER)")
                continue
            note = change_summary(prev_grid, self.grid)
            if self.frames_last > 1:
                note += f" ({self.frames_last} animation frames)"
            if self.levels > prev_levels:
                note = "LEVEL COMPLETED; " + note
            lines.append(f"{i + 1}. {label}: {note}; state={self.state.name}")
            if self.levels > prev_levels or self.state == self.GameState.GAME_OVER:
                break
        text = "\n".join(lines) + f"\n{self.status()}"
        patch = patch_hex(start_grid, self.grid)
        if patch:
            text += "\n" + patch
        if self.finished:
            text += "\nThe game is won." if self.state == self.GameState.WIN else "\nThe action budget is used up."
            text += " Stop playing now."
        return [{"type": "text", "text": text}]


TOOLS = [
    {
        "name": "arc_observe",
        "description": (
            "Show the current game frame: status line and the 64x64 grid as hex digits (one row per line, row number "
            "first, colour 0-f per cell; x = column index, y = row number; \"NN-MM\" marks identical rows NN..MM). "
            "Costs no game actions. arc_act already shows small changes, so observe only when you need the full frame."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "arc_act",
        "description": (
            "Send actions to the game, in order. Each is {\"action\": 0-7, \"x\": col, \"y\": row}: 0=RESET (restart the "
            "level; needed after GAME_OVER), 1=up, 2=down, 3=left, 4=right, 5=interact, 6=click at (x, y), 7=undo. "
            "Only available actions have an effect. Every action counts against the score (fewer is better). Up to 20 "
            "per call; stops early when a level is completed or the game is over. Returns what changed after each and, "
            "when the changed area is small, its new cells."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_BATCH,
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "integer", "minimum": 0, "maximum": 7},
                            "x": {"type": "integer", "minimum": 0, "maximum": 63},
                            "y": {"type": "integer", "minimum": 0, "maximum": 63},
                        },
                        "required": ["action"],
                    },
                }
            },
            "required": ["actions"],
        },
    },
]


def call_tool(game: "Game", name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    if name == "arc_observe":
        content = game.observe()
    elif name == "arc_act":
        content = game.act(list(args.get("actions") or []))
    else:
        raise ValueError(f"unknown tool {name}")
    game.record()
    return content


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
