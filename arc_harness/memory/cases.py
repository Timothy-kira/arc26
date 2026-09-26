"""Cases: one reusable experience per natural boundary (level won / death / game end).

EverOS's ``AgentCase{task_intent, approach, key_insight, quality_score}`` with two
changes: boundaries come from the game itself (no LLM boundary detector), and
``quality`` is computed from the outcome rather than self-rated by the LLM.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


def objective_quality(won: bool, actions: int, baseline: Optional[int], shortest: Optional[int]) -> float:
    """0..1. Unwon experiences score low (they still teach hazards), won ones by efficiency."""
    if not won:
        return 0.1
    ref = baseline or shortest
    if not ref or actions <= 0:
        return 0.6
    eff = min(1.0, ref / actions)
    return round(0.4 + 0.6 * eff, 4)


@dataclass
class Case:
    game_id: str
    level: int
    features: list[str]
    task_intent: str
    approach: str
    key_insight: str
    outcome: str  # won | died | unsolved
    actions: int
    quality: float
    action_summary: str = ""
    skills_used: list[str] = field(default_factory=list)
    cluster_id: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)

    def text(self) -> str:
        return f"{self.task_intent}\n{self.approach}\n{self.key_insight}"

    def to_markdown(self) -> str:
        return (
            f"---\ntype: agent_case\nid: {self.id}\ngame: {self.game_id}\nlevel: {self.level + 1}\n"
            f"outcome: {self.outcome}\nquality: {self.quality}\nfeatures: {json.dumps(self.features)}\n---\n\n"
            f"## TaskIntent\n{self.task_intent}\n\n## Approach\n{self.approach}\n\n"
            f"## KeyInsight\n{self.key_insight}\n\n## Actions\n{self.action_summary}\n"
        )


class CaseStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index = self.root / "cases.jsonl"
        self._lock = threading.Lock()
        self.cases: list[Case] = []
        if self._index.exists():
            for line in self._index.read_text().splitlines():
                if line.strip():
                    self.cases.append(Case(**json.loads(line)))

    def add(self, case: Case) -> Case:
        with self._lock:
            self.cases.append(case)
            with self._index.open("a") as f:
                f.write(json.dumps(asdict(case)) + "\n")
            (self.root / f"case_{case.id}.md").write_text(case.to_markdown())
        return case

    def by_ids(self, ids: list[str]) -> list[Case]:
        want = set(ids)
        return [c for c in self.cases if c.id in want]

    def in_cluster(self, cluster_id: str) -> list[Case]:
        return [c for c in self.cases if c.cluster_id == cluster_id]
