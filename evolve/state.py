"""Run spec, config fingerprint, candidate tree and resumable state.

Raven's evolver stores candidates as git commits; Kaggle datasets do not carry
``.git`` reliably, so a candidate here is an *overlay*: the full text of every
whitelisted file it changes, applied on top of its parent's overlay. Overlays
materialize into a scratch copy of the repo and export as unified diffs.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

WHY_SEED = {
    "W1_action_loop": "keeps repeating actions/states without new information",
    "W2_mechanic_undiscovered": "never discovered what an available action or object does",
    "W3_wrong_hypothesis_persisted": "kept acting on a goal/rule the evidence had refuted",
    "W4_inefficient_path": "won the level but with far more actions than needed",
    "W5_repeated_death": "died the same way more than once",
    "W6_click_target_miss": "clicked the wrong cells/objects or missed the relevant target",
    "W7_budget_exhausted": "ran out of time/actions/reasoning rounds before finishing",
    "W8_perception_error": "misread the scene (objects, HUD, colours, positions)",
}

WHITELIST = (
    "prompts/",
    "arc_harness/roles/",
    "arc_harness/charter.py",
    "arc_harness/explore/config.py",
    "skills/",
)

IMMUTABLE = (
    "evolve/",
    "arc_harness/scoring.py",
    "arc_harness/env/",
    "tests/",
    "pyproject.toml",
)


@dataclass
class RunSpec:
    work_dir: str
    repo_dir: str = "."
    train_games: list[str] = field(default_factory=list)
    test_games: list[str] = field(default_factory=list)
    k_confirm: int = 3
    anchor_size: int = 6
    max_rounds: int = 12
    patience: int = 5
    max_why_per_round: int = 2
    candidates_per_why: int = 2
    confirm_top: int = 2
    screen_sigma: float = 1.5
    editor_turns: int = 22
    run_args: list[str] = field(default_factory=lambda: ["--max-actions", "1500", "--hours", "0.5", "--concurrency", "12"])
    whitelist: list[str] = field(default_factory=lambda: list(WHITELIST))

    @classmethod
    def load(cls, path: Path) -> "RunSpec":
        return cls(**json.loads(Path(path).read_text()))

    def fingerprint(self) -> str:
        data = asdict(self)
        data.pop("work_dir")
        blob = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class TreeNode:
    id: str
    parent_id: Optional[str]
    overlay: dict[str, str] = field(default_factory=dict)  # path -> full new text (cumulative)
    why: str = ""
    summary: str = ""
    screen: dict[str, float] = field(default_factory=dict)
    scores: dict[str, list[float]] = field(default_factory=dict)  # game -> K scores (train)
    status: str = "new"  # new | pruned | culled | confirmed | promoted | rejected
    reason: str = ""
    created: float = field(default_factory=time.time)

    @staticmethod
    def new_id() -> str:
        return "n_" + uuid.uuid4().hex[:8]

    def mean(self, games: Optional[list[str]] = None) -> float:
        gs = games or list(self.scores)
        vals = [sum(self.scores[g]) / len(self.scores[g]) for g in gs if self.scores.get(g)]
        return sum(vals) / len(gs) if gs else 0.0


class RunState:
    """Everything the loop needs to resume; one JSON file per node plus run_meta.json."""

    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self.root = Path(spec.work_dir)
        self.nodes_dir = self.root / "nodes"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "run_meta.json"
        meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}
        fp = spec.fingerprint()
        if meta and meta.get("fingerprint") != fp:
            raise RuntimeError(
                f"config fingerprint changed ({meta.get('fingerprint')} -> {fp}); refusing to resume {self.root}"
            )
        if meta.get("unsealed_at"):
            raise RuntimeError("sealed test already opened for this run; start a new work_dir")
        self.meta: dict[str, Any] = meta or {
            "fingerprint": fp,
            "created": time.time(),
            "round": 0,
            "parent": None,
            "vanilla": None,
            "no_improve": 0,
            "attempts": {},
            "history": [],
        }
        self.nodes: dict[str, TreeNode] = {}
        for p in self.nodes_dir.glob("*.json"):
            d = json.loads(p.read_text())
            self.nodes[d["id"]] = TreeNode(**d)
        assert not set(spec.train_games) & set(spec.test_games), "test games leaked into train"

    def save_node(self, node: TreeNode) -> None:
        self.nodes[node.id] = node
        (self.nodes_dir / f"{node.id}.json").write_text(json.dumps(asdict(node), indent=1))

    def save(self) -> None:
        self.meta_path.write_text(json.dumps(self.meta, indent=1))

    @property
    def parent(self) -> Optional[TreeNode]:
        pid = self.meta.get("parent")
        return self.nodes.get(pid) if pid else None
