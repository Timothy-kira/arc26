"""SKILL.md storage with usage feedback and retirement.

Frontmatter follows EverOS's agent_skill layout (``name, description, confidence,
maturity_score, source_case_ids, cluster_id``) plus ``applies_to`` feature tokens
and ``uses/wins`` counters. Values are JSON-encoded so the parser stays trivial.

Unlike Raven (whose skill-usage feedback is a no-op), outcomes flow back: every
level a skill was injected into updates its confidence, and skills that keep
failing are retired.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

_SLUG = re.compile(r"[^a-z0-9_]+")


def slug(name: str) -> str:
    s = _SLUG.sub("_", name.lower()).strip("_")
    return s[:48] or "skill"


@dataclass
class Skill:
    name: str
    description: str
    body: str
    applies_to: list[str] = field(default_factory=list)
    confidence: float = 0.5
    maturity_score: float = 0.0
    source_case_ids: list[str] = field(default_factory=list)
    cluster_id: str = ""
    uses: int = 0
    wins: int = 0
    retired: bool = False
    origin: str = "runtime"  # library | runtime
    path: Optional[str] = None

    @property
    def id(self) -> str:
        return slug(self.name)

    def to_markdown(self) -> str:
        meta = asdict(self)
        body = meta.pop("body")
        meta.pop("path")
        meta = {"type": "agent_skill", **meta}
        fm = "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items())
        return f"---\n{fm}\n---\n\n{body.strip()}\n"

    @classmethod
    def parse(cls, text: str, path: Optional[str] = None) -> "Skill":
        m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
        if not m:
            raise ValueError("missing frontmatter")
        meta: dict = {}
        for line in m.group(1).splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            try:
                meta[k.strip()] = json.loads(v)
            except json.JSONDecodeError:
                meta[k.strip()] = v.strip('"')
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in meta.items() if k in known and k not in ("body", "path")}
        return cls(body=m.group(2).strip(), path=path, **kwargs)


class SkillStore:
    """Layered store: a read-only frozen library plus a writable runtime root."""

    def __init__(self, runtime_root: Path, library_roots: Iterable[Path] = (), retire_below: float = 0.15, retire_min_uses: int = 4) -> None:
        self.runtime_root = Path(runtime_root)
        self.library_roots = [Path(p) for p in library_roots]
        self.retire_below = retire_below
        self.retire_min_uses = retire_min_uses
        self._lock = threading.Lock()
        self._skills: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        skills: dict[str, Skill] = {}
        for root, origin in [*((r, "library") for r in self.library_roots), (self.runtime_root, "runtime")]:
            if not root.exists():
                continue
            for p in sorted(root.glob("skill_*/SKILL.md")):
                try:
                    s = Skill.parse(p.read_text(), str(p))
                except ValueError:
                    continue
                s.origin = origin
                skills[s.id] = s  # runtime copy overrides library copy of the same skill
        self._skills = skills

    def all(self, include_retired: bool = False) -> list[Skill]:
        return [s for s in self._skills.values() if include_retired or not s.retired]

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(slug(skill_id))

    def save(self, skill: Skill) -> Skill:
        with self._lock:
            d = self.runtime_root / f"skill_{skill.id}"
            d.mkdir(parents=True, exist_ok=True)
            p = d / "SKILL.md"
            skill.path = str(p)
            p.write_text(skill.to_markdown())
            self._skills[skill.id] = skill
            return skill

    def record_use(self, skill_id: str, won: bool, efficiency: float = 0.0) -> Optional[Skill]:
        s = self.get(skill_id)
        if s is None:
            return None
        s.uses += 1
        s.wins += int(won)
        prior = 2.0
        s.confidence = round((s.wins + prior * 0.5 + 0.5 * efficiency * int(won)) / (s.uses + prior), 4)
        s.maturity_score = round(min(1.0, s.uses / 10), 3)
        if s.uses >= self.retire_min_uses and s.confidence < self.retire_below:
            s.retired = True
        return self.save(s)
