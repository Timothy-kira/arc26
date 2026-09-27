"""Game sets behind one environment interface (``arc_agi.Arcade`` over an environments dir).

- ``official``: the 25 public ARC-AGI-3 games (same designers as the hidden set), split into
  ``official_dev`` (20, used for development) and ``official_val`` (5, validation only: never
  inspected or tuned on). The 5 are drawn once with a fixed seed, stratified by tag family
  (1 keyboard, 2 click, 2 keyboard_click), not by how well anything plays them.
- ``community``: arc-interactive (github.com/theredbluepill/arc-interactive, MIT), 249 games
  in the same format (its copies of official games are left out); the development set. Its
  ``baseline_actions`` are not per-level human counts, so only levels and actions are reported there.
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


VAL_SEED = 2026
VAL_PER_FAMILY = {"keyboard": 1, "click": 2, "keyboard_click": 2}


def family(g: GameInfo) -> str:
    return next((t for t in g.tags if t in VAL_PER_FAMILY), "click")


def official_val_ids() -> set[str]:
    import random

    allg = games("official")
    rng = random.Random(VAL_SEED)
    out: set[str] = set()
    for fam, n in VAL_PER_FAMILY.items():
        pool = sorted(g.game_id for g in allg if family(g) == fam)
        out |= set(rng.sample(pool, n))
    return out


def games(name: str) -> list[GameInfo]:
    if name in ("official_dev", "official_val"):
        val = official_val_ids()
        return [g for g in games("official") if (g.game_id in val) == (name == "official_val")]
    if name == "dev":  # everything we may tune on
        return games("official_dev") + games("community")
    env_dir = SETS[name]
    official = {m.parent.parent.name for m in SETS["official"].glob("*/*/metadata.json")} if name == "community" else set()
    out = []
    for meta in sorted(env_dir.glob("*/*/metadata.json")):
        if meta.parent.parent.name in official:
            continue  # arc-interactive ships copies of some official games (ft09, ls20, vc33)
        m = json.loads(meta.read_text())
        base = tuple(m.get("baseline_actions") or ()) if HAS_HUMAN_BASELINE[name] else None
        out.append(GameInfo(m["game_id"], env_dir, tuple(m.get("tags") or ()), base or None))
    return out
