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
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
from typing import Any, Optional

import numpy as np

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
    return "\n".join(f"{r:02d} " + "".join(HEX[int(v)] for v in row) for r, row in enumerate(grid))


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
        self.actions = 0
        self.grid = np.zeros((64, 64), dtype=np.int8)
        self.state = GameState.NOT_PLAYED
        self.levels = self.win_levels = 0
        self.available: list[int] = []
        self.frames_last = 0
        self.t0 = time.time()
        self.step(0)

    def step(self, action: int, x: Optional[int] = None, y: Optional[int] = None) -> None:
        ga = self.GameAction.from_id(int(action))
        data = {"x": int(x or 0), "y": int(y or 0)} if ga.is_complex() else None
        raw = self.wrapper.step(ga, data=data)
        if raw is None:
            raise RuntimeError(f"engine returned nothing for {ga.name}")
        self.actions += 1
        frames = [np.asarray(f, dtype=np.int8) for f in (raw.frame or [])]
        if frames:
            self.grid = frames[-1]
        self.frames_last = len(frames)
        self.state = raw.state
        self.levels, self.win_levels = int(raw.levels_completed), int(raw.win_levels)
        self.available = [int(a) for a in (raw.available_actions or [])]

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
        return [
            {"type": "text", "text": f"{self.status()}\n{grid_hex(self.grid)}"},
            {"type": "image", "data": grid_png_b64(self.grid), "mimeType": "image/png"},
        ]

    def act(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        lines = []
        for i, a in enumerate(actions[:MAX_BATCH]):
            if self.finished:
                break
            prev_grid, prev_levels = self.grid.copy(), self.levels
            aid = int(a.get("action", 0))
            self.step(aid, a.get("x"), a.get("y"))
            label = "RESET" if aid == 0 else (f"ACTION6({a.get('x')},{a.get('y')})" if aid == 6 else f"ACTION{aid}")
            note = change_summary(prev_grid, self.grid)
            if self.frames_last > 1:
                note += f" ({self.frames_last} animation frames)"
            if self.levels > prev_levels:
                note = "LEVEL COMPLETED; " + note
            lines.append(f"{i + 1}. {label}: {note}; state={self.state.name}")
            if self.levels > prev_levels or self.state == self.GameState.GAME_OVER:
                break
        text = "\n".join(lines) + f"\n{self.status()}"
        if self.finished:
            text += "\nThe game is won." if self.state == self.GameState.WIN else "\nThe action budget is used up."
            text += " Stop playing now."
        return [{"type": "text", "text": text}]


TOOLS = [
    {
        "name": "arc_observe",
        "description": (
            "Show the current game frame: status line, the 64x64 grid as hex digits (one row per line, row number "
            "first, colour 0-f per cell; x = column index, y = row number) and a rendered image. Costs no game actions."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "arc_act",
        "description": (
            "Send actions to the game, in order. Each is {\"action\": 0-7, \"x\": col, \"y\": row}: 0=RESET (restart the "
            "level; needed after GAME_OVER), 1=up, 2=down, 3=left, 4=right, 5=interact, 6=click at (x, y), 7=undo. "
            "Only available actions have an effect. Every action counts against the score (fewer is better). Up to 20 "
            "per call; stops early when a level is completed or the game is over. Returns what changed after each."
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


def main() -> None:
    game: Optional[Game] = None

    def reply(msg_id: Any, result: Any = None, error: Optional[dict] = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

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
                if game is None:
                    game = Game()
                name, args = params.get("name"), params.get("arguments") or {}
                if name == "arc_observe":
                    content = game.observe()
                elif name == "arc_act":
                    content = game.act(list(args.get("actions") or []))
                else:
                    raise ValueError(f"unknown tool {name}")
                game.record()
                reply(msg_id, {"content": content, "isError": False})
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
    main()
