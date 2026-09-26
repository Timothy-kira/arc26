"""A live Python REPL per game: the game state is a set of Python variables, the model writes and
runs code against it, and every cell is recorded as a node of an exploration DAG.

The kernel is its own process and holds no game engine: ``act()`` asks the game daemon over its
unix socket to play an action, so code in the REPL can see exactly what a player sees (frames,
state, level counters) and nothing else. That keeps local evaluation honest and matches the
competition, where the game runs behind the gateway.

Namespace (refreshed after every action):
  grid        np.ndarray (64, 64) int: the current frame; grid[y, x], colours 0-15
  prev        the frame before the last action;  frames: all frames of the last action
  state       'NOT_FINISHED' | 'GAME_OVER' | 'WIN';  level: levels completed so far
  win_levels  number of levels;  available: action ids that do something now
  actions_used / level_actions_used: scorecard counts (every action counts, RESET too)
  history     list of dicts, one per action: n, action, x, y, level, state, changed, note
  act(a, x=None, y=None) -> dict   play one action (0 RESET, 1-4 directions, 5 interact,
                                   6 click at (x, y), 7 undo); returns changed/level_up/...
  reset()     RESET (only useful after GAME_OVER)
  show(g=None, y0=0, y1=64, x0=0, x1=64) -> str   hex rows with row/column labels
  changes(a=None, b=None) -> list of (y, x, old, new) between two frames (default prev->grid)
  nodes, node(i), rerun(i)          the exploration DAG: earlier cells and their code

Serve: ``python -m arc_mcp.kernel <kernel.sock> <game-daemon.sock>`` (env ARC_DAG: dag.json path).
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import signal
import socket
import sys
import time
import traceback
from typing import Any, Optional

import numpy as np

try:
    from arc_mcp import percept
except ImportError:  # run as a script
    import percept  # type: ignore

HEX = "0123456789abcdef"
CELL_ACTIONS = int(os.getenv("ARC_CELL_ACTIONS", "300"))
CELL_SECONDS = int(os.getenv("ARC_CELL_SECONDS", "120"))
OUT_LIMIT = int(os.getenv("ARC_OUT_LIMIT", "6000"))


PROTECTED = ("act", "reset", "show", "changes", "objects", "anim", "look", "regions", "node", "rerun", "dag",
             "journal", "np", "action_stats")


class CellBudget(Exception):
    pass


class CellTimeout(Exception):
    pass


class Game:
    """Client side of the game daemon."""

    def __init__(self, sock_path: str) -> None:
        self.path = sock_path
        self.f = None

    def call(self, req: dict[str, Any]) -> dict[str, Any]:
        for _ in range(2):
            try:
                if self.f is None:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(self.path)
                    self.f = s.makefile("rw")
                self.f.write(json.dumps(req) + "\n")
                self.f.flush()
                line = self.f.readline()
                if line:
                    return json.loads(line)
            except OSError:
                pass
            self.f = None
        raise RuntimeError("game daemon unreachable")


def show(g: Optional[np.ndarray] = None, y0: int = 0, y1: int = 64, x0: int = 0, x1: int = 64) -> str:
    g = NS["grid"] if g is None else np.asarray(g)
    sub = g[y0:y1, x0:x1]
    head = "   " + "".join(str((x // 10) % 10) for x in range(x0, x0 + sub.shape[1])) + "\n" + \
           "   " + "".join(str(x % 10) for x in range(x0, x0 + sub.shape[1]))
    rows = [f"{y0 + i:02d} " + "".join(HEX[int(v)] for v in row) for i, row in enumerate(sub)]
    return head + "\n" + "\n".join(rows)


def changes(a: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None, limit: int = 200) -> list:
    a = NS["prev"] if a is None else np.asarray(a)
    b = NS["grid"] if b is None else np.asarray(b)
    ys, xs = np.where(a != b)
    return [(int(y), int(x), int(a[y, x]), int(b[y, x])) for y, x in list(zip(ys, xs))[:limit]]


class Kernel:
    def __init__(self, game_sock: str, dag_path: Optional[str]) -> None:
        self.game = Game(game_sock)
        self.dag_path = dag_path
        self.nodes: list[dict[str, Any]] = []
        if dag_path and os.path.exists(dag_path):  # a restarted kernel keeps the DAG (not the variables)
            try:
                self.nodes = json.load(open(dag_path))
            except (OSError, ValueError):
                self.nodes = []
        self.cell_actions = 0
        self.cell_events: list[str] = []
        self.want_image = False
        self.journal_path = (os.path.splitext(dag_path)[0] + "_journal.json") if dag_path else None
        self.journal: list[dict[str, Any]] = []
        if self.journal_path and os.path.exists(self.journal_path):
            try:
                self.journal = json.load(open(self.journal_path))
            except (OSError, ValueError):
                self.journal = []
        NS.update(regions=lambda a=None, b=None: percept.change_regions(NS["prev"] if a is None else a,
                                                                          NS["grid"] if b is None else b),
                  objects=lambda g=None: percept.objects(NS["grid"] if g is None else g),
                  anim=lambda: percept.anim(NS["frames"], NS["prev"]), look=self._look, journal=self.journal,
                  dag=lambda last=20: self.dag(last), action_stats=self.action_stats)
        NS.update(np=np, show=show, changes=changes, act=self.act, reset=lambda: self.act(0), history=[],
                  node=self.node, rerun=self.rerun)
        NS["nodes"] = self.nodes
        self._sync(self.game.call({"op": "obs"}), first=True)

    # ------------------------------------------------------------------ game API

    def _sync(self, r: dict[str, Any], first: bool = False) -> None:
        g = np.array(r["grid"], dtype=np.int64)
        NS["prev"] = NS.get("grid", g) if not first else g
        NS["grid"] = g
        NS["frames"] = [np.array(f, dtype=np.int64) for f in r.get("frames") or [r["grid"]]]
        NS.update(state=r["state"], level=r["level"], win_levels=r["win_levels"], available=r["available"],
                  actions_used=r["actions"], level_actions_used=r["level_so_far"])
        self.status = r["status"]

    def act(self, a: int, x: Optional[int] = None, y: Optional[int] = None) -> dict[str, Any]:
        if self.cell_actions >= CELL_ACTIONS:
            raise CellBudget(f"this cell already played {CELL_ACTIONS} actions; look at the results first")
        before = NS["level"]
        r = self.game.call({"op": "step", "action": int(a), "x": None if x is None else int(x),
                            "y": None if y is None else int(y), "source": "code"})
        if "error" in r:
            raise RuntimeError(r["error"])
        self._sync(r)
        self.cell_actions += int(r.get("counted", 1))
        changed = int((NS["prev"] != NS["grid"]).sum())
        note = r.get("note", "")
        if note:
            self.cell_events.append(note)
        mv = percept.moves(NS["prev"], NS["grid"], limit=3) if changed and NS["level"] == before else []
        rec = {"n": NS["actions_used"], "action": int(a), "x": x, "y": y, "level": NS["level"], "state": NS["state"],
               "changed": changed, "note": note, "moves": [(m["color"], m["size"], m["d"]) for m in mv]}
        NS["history"].append(rec)
        return {"changed": changed, "state": NS["state"], "level": NS["level"], "level_up": NS["level"] > before,
                "game_over": NS["state"] == "GAME_OVER", "note": note, "frames": len(NS["frames"])}

    def action_stats(self, level: Optional[int] = None) -> str:
        """Facts from the history: for each action id, how often it was played and what it moved."""
        from collections import Counter as C
        rows: dict = {}
        for h in NS["history"]:
            if level is not None and h["level"] != level:
                continue
            r = rows.setdefault(h["action"], {"n": 0, "noop": 0, "moves": C()})
            r["n"] += 1
            r["noop"] += h["changed"] == 0
            for c, size, d in h.get("moves") or []:
                r["moves"][(c, tuple(size), tuple(d))] += 1
        out = []
        for a, r in sorted(rows.items()):
            mv = ", ".join(f"{percept.NAMES[c]}#{c} {w}x{hh} d=({d[0]:+d},{d[1]:+d}) x{n}"
                           for (c, (w, hh), d), n in r["moves"].most_common(3))
            out.append(f"ACTION{a}: played {r['n']}, no change {r['noop']}" + (f"; moved {mv}" if mv else ""))
        return "\n".join(out) or "no actions yet"

    def _look(self) -> str:
        """Attach the current frame (4x image) to this cell's reply."""
        self.want_image = True
        return "image of the current frame attached"

    # ------------------------------------------------------------------ DAG

    def node(self, i: int) -> dict[str, Any]:
        return self.nodes[i]

    def rerun(self, i: int) -> None:
        exec(compile(self.nodes[i]["code"], f"<node {i}>", "exec"), NS)

    def _save(self) -> None:
        for path, data in ((self.dag_path, self.nodes), (self.journal_path, self.journal)):
            if not path:
                continue
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=1)
            os.replace(tmp, path)

    # ------------------------------------------------------------------ cells

    def run(self, code: str, purpose: str = "", parents: Optional[list[int]] = None, expect: str = "",
            revises: Optional[int] = None, check: str = "") -> tuple[str, list[str]]:
        self.cell_actions, self.cell_events, self.want_image = 0, [], False
        builtins_before = {k: NS.get(k) for k in PROTECTED}
        level0, t0 = NS["level"], time.time()
        grid0, state0, deaths0 = NS["grid"].copy(), NS["state"], sum(1 for h in NS["history"] if h["state"] == "GAME_OVER")
        buf = io.StringIO()
        err = None

        def alarm(*_):
            raise CellTimeout(f"cell exceeded {CELL_SECONDS}s")

        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(CELL_SECONDS)
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                tree = ast.parse(code, "<cell>", "exec")
                last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                exec(compile(tree, "<cell>", "exec"), NS)
                if last is not None:
                    val = eval(compile(ast.Expression(last.value), "<cell>", "eval"), NS)
                    if val is not None:
                        print(val if isinstance(val, str) else repr(val))
        except (CellBudget, CellTimeout) as e:
            err = str(e)
        except Exception:
            tb = traceback.format_exc().strip().splitlines()
            err = "\n".join(l for l in tb if "arc_mcp/kernel.py" not in l)[-1500:]
        finally:
            signal.alarm(0)
        out = buf.getvalue()
        clobbered = [k for k, v in builtins_before.items() if v is not None and NS.get(k) is not v]
        for k in clobbered:  # the game API must survive the model's own variable names
            NS[k] = builtins_before[k]
        if clobbered:
            out += (f"\nNOTE: this cell overwrote {', '.join(clobbered)}; the REPL restored the built-in version. "
                    "Use other names for your variables.")
        if len(out) > OUT_LIMIT:
            out = out[:2000] + f"\n... [{len(out) - OUT_LIMIT} chars cut] ...\n" + out[-(OUT_LIMIT - 2000):]
        nid = len(self.nodes)
        parents = parents if parents is not None else ([nid - 1] if nid else [])
        if revises is not None and revises not in parents:
            parents = [revises] + parents
        deaths = sum(1 for h in NS["history"] if h["state"] == "GAME_OVER") - deaths0
        outcome = {  # what actually happened, recorded next to what the cell expected
            "actions": self.cell_actions, "levels": NS["level"] - level0, "deaths": deaths,
            "state": f"{state0}->{NS['state']}" if NS["state"] != state0 else NS["state"],
            "cells_changed": int((grid0 != NS["grid"]).sum()), "error": err.splitlines()[-1] if err else None,
        }
        # verdict: the cell's own check expression if given, else the obvious failure signals
        verdict, why = None, ""
        if check:
            NS["outcome"] = outcome
            try:
                verdict = bool(eval(compile(check, "<check>", "eval"), NS))
                why = f"check `{check}` is {verdict}"
            except Exception as exc:
                verdict, why = False, f"check `{check}` raised {type(exc).__name__}: {exc}"
        if err or deaths > 0:
            verdict, why = False, ("error: " + outcome["error"]) if err else f"{deaths} GAME_OVER"
        elif verdict is None and expect and "level" in expect.lower() and self.cell_actions > 0:
            verdict = outcome["levels"] > 0
            why = "expected a level-up: " + ("it came" if verdict else "it did not come")
        flag = verdict is False
        if verdict is not None or expect:
            self.journal.append({"node": nid, "level": level0 + 1, "ok": verdict, "purpose": purpose[:100],
                                 "expect": expect[:120], "why": why[:160], "actions": self.cell_actions,
                                 "revises": revises})
        self.nodes.append({"id": nid, "parents": parents, "purpose": purpose, "expect": expect, "check": check,
                           "verdict": verdict, "why": why, "outcome": outcome, "flag": flag, "revises": revises, "code": code, "actions": self.cell_actions,
                           "level_before": level0, "level_after": NS["level"], "events": self.cell_events,
                           "error": outcome["error"], "out": out[:400], "seconds": round(time.time() - t0, 1)})
        self._save()
        got = f"{self.cell_actions}a, levels {outcome['levels']:+d}, deaths {deaths}, {outcome['state']}, " \
              f"{outcome['cells_changed']} cells changed" + (", error" if err else "")
        head = f"[node {nid}] {got}" + (f"; events: {'; '.join(self.cell_events[-4:])}" if self.cell_events else "")
        if expect:
            mark = {True: "RIGHT", False: "WRONG", None: "unchecked"}[verdict]
            head += f"\nexpected: {expect[:200]} -> {mark}" + (f" ({why})" if why else "") + \
                    (f"; fix it in a cell with revises={nid}" if flag else "")
        parts = [self.status, head, self.journal_text()]
        images: list[str] = []
        if self.cell_actions > 0 or self.want_image:
            images.append(percept.png4x(NS["grid"]))
            per = ["PERCEPTION (4x image of the current frame attached):", percept.objects_text(NS["grid"])]
            if self.cell_actions > 0:
                per.append("last action " + percept.regions_text(NS["prev"], NS["grid"]))
                mt = percept.moves_text(NS["prev"], NS["grid"])
                if mt:
                    per.append("last action " + mt)
                if self.cell_actions > 1:
                    per.append("whole cell " + percept.regions_text(grid0, NS["grid"]))
            at = percept.anim_text(NS["frames"], NS["prev"]) if self.cell_actions > 0 else ""
            if at:
                per.append(at)
            parts.append("\n".join(per))
        if out.strip():
            parts.append(out.rstrip())
        if err:
            parts.append("ERROR:\n" + err)
        return "\n".join(p for p in parts if p), images

    def journal_text(self, wrong: int = 4, right: int = 3) -> str:
        """This game's live record of what was right and wrong, newest last."""
        revised = {j["revises"] for j in self.journal if j.get("revises") is not None}
        bad = [j for j in self.journal if j["ok"] is False][-wrong:]
        good = [j for j in self.journal if j["ok"] is True][-right:]
        if not bad and not good:
            return ""
        lines = ["JOURNAL (this game; `journal` in the REPL has all of it):"]
        for j in bad:
            lines.append(f"  WRONG n{j['node']} L{j['level']}: {j['expect'] or j['purpose']} -> {j['why']}"
                         + ("" if j["node"] in revised else "  [not revised yet]"))
        for j in good:
            lines.append(f"  RIGHT n{j['node']} L{j['level']}: {j['expect'] or j['purpose']}")
        return "\n".join(lines)

    def events(self) -> list[dict[str, Any]]:
        """Non-REPL tool calls (plans, notes, skill reads) logged by the arc26 plugin's PostToolUse hook."""
        if not self.dag_path:
            return []
        path = os.path.join(os.path.dirname(self.dag_path), "dag_events.jsonl")
        try:
            return [json.loads(l) for l in open(path) if l.strip()]
        except (OSError, ValueError):
            return []

    def dag(self, last: int = 12) -> str:
        if not self.nodes:
            return "(no cells yet)"
        lines = []
        ev = self.events()[-6:]
        if ev:
            lines.append("other tool calls (plans, notes, skills): " + "; ".join(
                f"{e['tool']} {e['input'][:70]}" for e in ev))
        for n in self.nodes[-last:]:
            lv = f"L{n['level_before'] + 1}" + (f"->L{n['level_after'] + 1}" if n["level_after"] != n["level_before"] else "")
            tag = "!" if n.get("flag") else ("*" if n["events"] else "")
            rev = f" revises {n['revises']}" if n.get("revises") is not None else ""
            line = f"[{n['id']}]{tag} <- {n['parents']}{rev} {lv} {n['actions']}a {n['purpose'][:80]}"
            if n.get("expect"):
                o = n.get("outcome") or {}
                line += f"\n     expected: {n['expect'][:90]} | got: levels {o.get('levels', 0):+d}, deaths {o.get('deaths', 0)}" \
                        + (f", {o['error'][:60]}" if o.get("error") else "")
            lines.append(line)
        open_ = [n["id"] for n in self.nodes if n.get("flag") and not any(m.get("revises") == n["id"] for m in self.nodes)]
        if open_:
            lines.append(f"unresolved surprises (no cell revises them yet): {open_[-8:]}")
        return "\n".join(lines)


NS: dict[str, Any] = {"__name__": "__arc__"}


def guard(blocked: list[str]) -> None:
    """Audit hook: no reading of game files, no subprocesses (the REPL sees only what a player sees)."""

    def hook(event: str, args: tuple) -> None:
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            if any(b and b in p for b in blocked):
                raise PermissionError(f"reading {p} is not allowed")
        elif event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork"):
            raise PermissionError(f"{event} is not allowed in the game REPL")

    sys.addaudithook(hook)


def main() -> None:
    ksock, gsock = sys.argv[1], sys.argv[2]
    k = Kernel(gsock, os.getenv("ARC_DAG"))
    blocked = [p for p in os.getenv("ARC_BLOCK_PATHS", "").split(os.pathsep) if p]
    if os.path.exists(ksock):
        os.unlink(ksock)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(ksock)
    srv.listen(4)
    guard(blocked)  # after binding: from here on the cells run
    while True:
        conn, _ = srv.accept()
        with conn, conn.makefile("rw") as f:
            for line in f:
                try:
                    req = json.loads(line)
                    if req.get("op") == "dag":
                        reply = {"text": "\n".join(x for x in [k.status, k.dag(int(req.get("last", 12))), k.journal_text(8, 5)] if x)}
                    else:
                        text, images = k.run(str(req.get("code", "")), str(req.get("purpose", "")), req.get("parents"),
                                             str(req.get("expect", "") or ""), req.get("revises"), str(req.get("check", "") or ""))
                        reply = {"text": text, "images": images}
                except Exception as exc:  # the kernel itself must survive
                    reply = {"text": f"kernel error: {type(exc).__name__}: {exc}"}
                f.write(json.dumps(reply) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
