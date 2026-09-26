"""Observable game features: the retrieval key for cases and skills.

Hidden games carry no reliable metadata, so everything here is computed from what
the agent itself sees: the action set, HUD bars, animation length, colours and
how the scene reacts to directional input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..env.encode import COLOR_NAMES, background_color, objects


@dataclass
class GameFeatures:
    available_actions: list[int] = field(default_factory=list)
    has_hud: bool = False
    multi_frame: bool = False
    background: int = 0
    colors: list[int] = field(default_factory=list)
    n_objects: int = 0
    player_like: bool = False
    click_only: bool = False
    extra: set[str] = field(default_factory=set)

    def tokens(self) -> list[str]:
        acts = set(self.available_actions)
        kind = (
            "click" if acts == {6} else
            "keyboard" if 6 not in acts else
            "keyboard_click"
        )
        toks = [f"kind:{kind}", "actions:" + "".join(str(a) for a in sorted(acts))]
        toks += [f"act:{a}" for a in sorted(acts)]
        if 5 in acts:
            toks.append("has:interact")
        if 7 in acts:
            toks.append("has:undo")
        if self.has_hud:
            toks.append("hud:bar")
        if self.multi_frame:
            toks.append("anim:multi")
        if self.player_like:
            toks.append("player:avatar")
        toks.append(f"bg:{COLOR_NAMES[self.background]}")
        toks += [f"color:{COLOR_NAMES[c]}" for c in self.colors]
        bucket = "few" if self.n_objects < 8 else "some" if self.n_objects < 30 else "many"
        toks.append(f"objects:{bucket}")
        toks += sorted(self.extra)
        return toks


def initial_features(grid: np.ndarray, available: list[int], n_frames: int) -> GameFeatures:
    bg = background_color(grid)
    objs = objects(grid, bg)
    colors = sorted({o.color for o in objs})
    return GameFeatures(
        available_actions=sorted(a for a in available if a != 0),
        multi_frame=n_frames > 1,
        background=bg,
        colors=colors,
        n_objects=len(objs),
        click_only=set(available) - {0} == {6},
    )


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
