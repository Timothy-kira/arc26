"""Game sets behind one environment interface (``arc_agi.Arcade`` over an environments dir).

- ``official``: the 25 public ARC-AGI-3 games. Same designers as the hidden set, so they are
  the validation set: evaluate on them, never tune on them.
- ``community``: arc-interactive (github.com/theredbluepill/arc-interactive, MIT), 252 games
  in the same format; the development set. Its ``baseline_actions`` are not per-level human
  counts, so only levels and actions are reported there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SETS = {
    "official": ROOT / "data" / "environment_files",
    "community": ROOT / "data" / "community" / "environment_files",
}
HAS_HUMAN_BASELINE = {"official": True, "community": False}


@dataclass(frozen=True)
class GameInfo:
    game_id: str
    env_dir: Path
    tags: tuple[str, ...]
    baseline: Optional[tuple[int, ...]]


def games(name: str) -> list[GameInfo]:
    env_dir = SETS[name]
    out = []
    for meta in sorted(env_dir.glob("*/*/metadata.json")):
        m = json.loads(meta.read_text())
        base = tuple(m.get("baseline_actions") or ()) if HAS_HUMAN_BASELINE[name] else None
        out.append(GameInfo(m["game_id"], env_dir, tuple(m.get("tags") or ()), base or None))
    return out
