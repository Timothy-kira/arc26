"""ARC benchmark backend for the evolver: evaluate a candidate overlay on a set of games.

Each evaluation runs the agent in a materialized copy of the repo (so prompt and
code edits take effect) as a subprocess, K times, and reads per-game RHAE from the
engine's own scorecard. Runs that crash are retried up to twice (Gate-f); a game
that still has no result scores 0 and stays in the denominator.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Protocol

from .state import RunSpec, TreeNode
from .workspace import materialize

logger = logging.getLogger(__name__)


class Evaluator(Protocol):
    def eval(self, node: TreeNode, games: list[str], k: int, tag: str) -> dict[str, list[float]]: ...

    def trajectories(self, node: TreeNode, games: list[str], tag: str) -> list[tuple[str, str]]: ...


def _match(summary_games: dict, gid: str) -> Optional[dict]:
    for k, v in summary_games.items():
        if k == gid or k.startswith(gid):
            return v
    return None


class SubprocessEvaluator:
    def __init__(self, spec: RunSpec, extra_env: Optional[dict[str, str]] = None, infra_retries: int = 2) -> None:
        self.spec = spec
        self.repo = Path(spec.repo_dir).resolve()
        self.root = Path(spec.work_dir).resolve()
        self.extra_env = extra_env or {}
        self.infra_retries = infra_retries
        self.infra_failures: dict[str, int] = {}

    def _ws(self, node: TreeNode) -> Path:
        ws = self.root / "ws" / node.id
        if not (ws / "arc_harness").exists():
            materialize(self.repo, node.overlay, ws)
        return ws

    def _out(self, node: TreeNode, tag: str, i: int) -> Path:
        return self.root / "evals" / node.id / f"{tag}_k{i}"

    def _run_once(self, ws: Path, games: list[str], out: Path, tag: str, seed: int) -> Optional[dict]:
        cmd = [
            sys.executable, "-m", "arc_harness.run", "--mode", "offline",
            "--games", ",".join(games), "--out", str(out), "--tag", tag, "--seed", str(seed),
            *self.spec.run_args,
        ]
        env = {**os.environ, **self.extra_env, "PYTHONPATH": str(ws), "ARC26_PROMPTS_DIR": str(ws / "prompts")}
        env.setdefault("ENVIRONMENTS_DIR", str(self.repo / "data" / "environment_files"))
        for attempt in range(self.infra_retries + 1):
            r = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True)
            summ = out / "summary.json"
            if r.returncode == 0 and summ.exists():
                return json.loads(summ.read_text())
            self.infra_failures[str(out)] = attempt + 1
            logger.warning("eval run failed (attempt %d): %s", attempt + 1, r.stderr[-800:])
        return None

    def eval(self, node: TreeNode, games: list[str], k: int, tag: str) -> dict[str, list[float]]:
        ws = self._ws(node)
        scores: dict[str, list[float]] = {g: [] for g in games}
        for i in range(k):
            out = self._out(node, tag, i)
            summ_path = out / "summary.json"
            summary = json.loads(summ_path.read_text()) if summ_path.exists() else self._run_once(ws, games, out, tag, i)
            for g in games:
                v = _match((summary or {}).get("games", {}), g)
                scores[g].append(float(v["score"]) if v else 0.0)
        return scores

    def trajectories(self, node: TreeNode, games: list[str], tag: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        base = self.root / "evals" / node.id
        for run in sorted(base.glob(f"{tag}_k*")):
            gdir = run / "games"
            for g in games:
                for d in gdir.glob(f"{g}*"):
                    parts = []
                    for name in ("report.json", "notebook.md"):
                        p = d / name
                        if p.exists():
                            parts.append(f"### {name}\n{p.read_text()[:4000]}")
                    rounds = sorted(d.glob("round_*/nodes/*_plan*.out.md"))[-2:]
                    for p in rounds:
                        parts.append(f"### {p.parent.parent.name}/{p.name}\n{p.read_text()[:1500]}")
                    if parts:
                        out.append((g, "\n\n".join(parts)))
            if out:
                break
        return out
