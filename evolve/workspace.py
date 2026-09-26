"""Materialize an overlay into a scratch repo copy; guard edits; export diffs."""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from .state import IMMUTABLE

_SKIP = {".git", ".venv", "runs", "data", "__pycache__", ".pytest_cache", ".ruff_cache", ".kaggle"}


def path_allowed(path: str, whitelist: Iterable[str]) -> bool:
    p = path.lstrip("./")
    if ".." in Path(p).parts:
        return False
    if any(p.startswith(i) for i in IMMUTABLE):
        return False
    return any(p == w or p.startswith(w) for w in whitelist)


def materialize(repo: Path, overlay: dict[str, str], dest: Path | None = None) -> Path:
    dest = Path(dest or tempfile.mkdtemp(prefix="arc26_ws_"))
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(repo, dest, ignore=lambda d, names: [n for n in names if n in _SKIP])
    data = Path(repo) / "data"
    if data.exists():
        (dest / "data").symlink_to(data.resolve())
    for rel, text in overlay.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return dest


def overlay_diff(repo: Path, overlay: dict[str, str]) -> str:
    out: list[str] = []
    for rel, new in sorted(overlay.items()):
        p = Path(repo) / rel
        old = p.read_text() if p.exists() else ""
        out += difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True), f"a/{rel}", f"b/{rel}"
        )
    return "".join(out)


def check_candidate(ws: Path, changed: Iterable[str], forbidden_tokens: Iterable[str]) -> str:
    """Free pruning: compile, import smoke, and no hard-coded game ids. Returns '' if ok."""
    changed = list(changed)
    tokens = [t for t in forbidden_tokens if len(t) >= 4]
    for rel in changed:
        text = (ws / rel).read_text()
        for t in tokens:
            if re.search(rf"\b{re.escape(t)}\b", text):
                return f"{rel} hard-codes game id {t!r}"
        if rel.endswith(".py"):
            r = subprocess.run([sys.executable, "-m", "py_compile", str(ws / rel)], capture_output=True, text=True)
            if r.returncode:
                return f"compile error in {rel}: {r.stderr[-400:]}"
    r = subprocess.run(
        [sys.executable, "-c", "import arc_harness.agent, arc_harness.run"],
        cwd=ws,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ws), "PATH": "/usr/bin:/bin"},
    )
    if r.returncode:
        return f"import smoke failed: {r.stderr[-400:]}"
    return ""
