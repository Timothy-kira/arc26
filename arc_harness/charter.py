"""Per-level Charter: Raven's runtime "hot evolution", declarative only.

Raven regenerates a Charter (system prompt addendum / stop condition / tool subset
/ declarative checks) per dispatch and applies it on the next turn. Here a Charter
is regenerated at each level start (and refined by every plan) and applied to the
explorer: action priorities, bans, and a stop-reasoning threshold. No generated
code is executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .explore.explorer import Explorer
from .roles.schemas import CharterOut, PlanOut


@dataclass
class Charter:
    system_addendum: str = ""
    priorities: dict[int, float] = field(default_factory=dict)
    avoid: set[tuple] = field(default_factory=set)
    max_level_actions: int = 0
    max_noop_repeats: int = 1

    @classmethod
    def from_out(cls, out: CharterOut) -> "Charter":
        return cls(
            system_addendum=out.system_addendum.strip()[:600],
            priorities=dict(out.priorities),
            avoid={a.key() for a in out.avoid},
            max_level_actions=max(0, int(out.max_level_actions)),
        )

    def refine(self, plan: PlanOut) -> None:
        if plan.priorities:
            self.priorities.update(plan.priorities)
        self.avoid |= {a.key() for a in plan.avoid}

    def apply(self, explorer: Explorer) -> None:
        g = explorer.graph
        if g is None:
            return
        g.priority = dict(self.priorities)
        simple_bans = {k for k in self.avoid if k[0] != 6}
        g.banned = set(simple_bans) | {k for k in self.avoid if k[0] == 6}
        for node in g.nodes.values():
            node.candidates.sort(key=lambda a: -g.priority.get(a[0], 1.0))

    def allows(self, key: tuple) -> bool:
        return key not in self.avoid

    def reasoning_allowed(self, level_actions: int) -> bool:
        return self.max_level_actions <= 0 or level_actions < self.max_level_actions

    def render(self) -> str:
        if not (self.system_addendum or self.priorities or self.avoid):
            return ""
        pr = ", ".join(f"ACTION{a}:{p:.1f}" for a, p in sorted(self.priorities.items()))
        return f"Charter for this level: {self.system_addendum} Priorities: {pr or 'default'}."
