"""Tunable exploration knobs. On the evolver's edit whitelist."""

from dataclasses import dataclass


@dataclass
class ExploreConfig:
    hud_band: int = 3
    hud_min_transitions: int = 4
    hud_change_rate: float = 0.4
    max_click_targets: int = 32
    click_large_object_px: int = 400
    stuck_steps_before_llm: int = 12
    undo_priority: float = 0.2
    reset_when_no_frontier: bool = True
    max_path_len: int = 64


DEFAULT = ExploreConfig()
