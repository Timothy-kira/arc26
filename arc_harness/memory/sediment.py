"""Skill sedimentation: cases -> clusters -> SKILL.md (EverOS agent-memory pipeline, local).

1. ``assign_cluster``: a new case joins the most similar cluster (feature Jaccard +
   intent token overlap) or starts a new one.
2. ``distill``: for a target case, the extractor sees up to 10 existing skills from
   its cluster (ranked by similarity) and up to 9 supporting cases drawn from those
   skills' ``source_case_ids`` (ranked by quality, then recency), and decides to add
   a new skill, update an existing one, or do nothing. This is the dedup/merge step.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .. import prompts
from ..llm.client import ChatModel
from ..llm.semantic import call_json
from .cases import Case, CaseStore
from .features import jaccard
from .retrieve import tokenize
from .skill_store import Skill, SkillStore, slug


class SkillOp(BaseModel):
    op: Literal["add", "update", "none"]
    target: str = ""
    name: str = ""
    description: str = ""
    applies_to: list[str] = Field(default_factory=list)
    body: str = ""


class SkillOpsOut(BaseModel):
    reasoning: str = ""
    ops: list[SkillOp] = Field(default_factory=list, max_length=2)


class Clusters:
    def __init__(self, path: Path, threshold: float = 0.45) -> None:
        self.path = Path(path)
        self.threshold = threshold
        self._lock = threading.Lock()
        self.data: dict[str, dict] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _sim(self, c: dict, case: Case) -> float:
        return 0.7 * jaccard(c["features"], case.features) + 0.3 * jaccard(c["intent_tokens"], tokenize(case.task_intent))

    def assign(self, case: Case) -> str:
        with self._lock:
            best, best_s = None, 0.0
            for cid, c in self.data.items():
                s = self._sim(c, case)
                if s > best_s:
                    best, best_s = cid, s
            if best is None or best_s < self.threshold:
                best = "c_" + uuid.uuid4().hex[:8]
                self.data[best] = {"features": list(case.features), "intent_tokens": tokenize(case.task_intent), "case_ids": []}
            c = self.data[best]
            c["case_ids"].append(case.id)
            c["features"] = sorted(set(c["features"]) | set(case.features))[:60]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1))
            return best


class Sedimenter:
    def __init__(self, store: SkillStore, cases: CaseStore, clusters: Clusters, llm: Optional[ChatModel]) -> None:
        self.store = store
        self.cases = cases
        self.clusters = clusters
        self.llm = llm

    def add_case(self, case: Case) -> Case:
        case.cluster_id = self.clusters.assign(case)
        return self.cases.add(case)

    def _context(self, target: Case) -> tuple[list[Skill], list[Case]]:
        skills = [s for s in self.store.all() if s.cluster_id == target.cluster_id]
        if len(skills) < 3:
            others = sorted(
                (s for s in self.store.all() if s not in skills),
                key=lambda s: -jaccard(s.applies_to, target.features),
            )
            skills += others[: 10 - len(skills)]
        skills.sort(key=lambda s: -jaccard(s.applies_to, target.features))
        skills = skills[:10]
        support_ids = [cid for s in skills for cid in s.source_case_ids]
        support = [c for c in self.cases.by_ids(support_ids) if c.id != target.id]
        support += [c for c in self.cases.in_cluster(target.cluster_id) if c.id != target.id and c not in support]
        support.sort(key=lambda c: (-c.quality, -c.created))
        return skills, support[:9]

    async def distill(self, target: Case) -> list[Skill]:
        if self.llm is None or target.quality < 0.2:
            return []
        skills, support = self._context(target)
        existing = "\n\n".join(
            f"[{s.id}] {s.name} (confidence {s.confidence:.2f}, uses {s.uses})\n{s.description}\n{s.body[:1500]}" for s in skills
        ) or "(none)"
        supporting = "\n\n".join(f"- {c.game_id} L{c.level + 1} ({c.outcome}, q={c.quality}): {c.text()}" for c in support) or "(none)"
        user = prompts.fill(
            "skill_extract_user",
            target=target.to_markdown(),
            existing=existing,
            supporting=supporting,
        )
        out = await call_json(self.llm, SkillOpsOut, prompts.load("skill_extract_system"), user, retries=2, max_tokens=2500)
        saved: list[Skill] = []
        for op in out.ops:
            if op.op == "none" or not op.body.strip():
                continue
            if op.op == "update" and self.store.get(op.target):
                s = self.store.get(op.target)
                assert s is not None
                s.description = op.description or s.description
                s.body = op.body
                s.applies_to = sorted(set(s.applies_to) | set(op.applies_to or target.features))
                if target.id not in s.source_case_ids:
                    s.source_case_ids.append(target.id)
                s.origin = "runtime"
            else:
                name = op.name or target.task_intent[:40]
                if self.store.get(slug(name)):
                    name = f"{name} {uuid.uuid4().hex[:4]}"
                s = Skill(
                    name=name,
                    description=op.description,
                    body=op.body,
                    applies_to=sorted(set(op.applies_to) | set(target.features)),
                    confidence=0.5,
                    source_case_ids=[target.id],
                    cluster_id=target.cluster_id,
                )
            saved.append(self.store.save(s))
        return saved
