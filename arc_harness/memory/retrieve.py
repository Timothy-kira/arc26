"""Skill retrieval: per-source BM25, RRF fusion, and an LLM gate.

Mirrors Raven's ``skill_forge`` router: each source over-fetches ``k*2`` hits,
scores are fused with ``sum(w_i / (rrf_k + rank))``, duplicates collapse by name,
and a gate picks at most two skills ("selecting an irrelevant or unexecutable
skill is strictly worse than selecting none").
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from pydantic import BaseModel, Field

from ..llm.client import ChatModel
from ..llm.semantic import call_json
from .cases import Case
from .features import jaccard
from .skill_store import Skill

_TOK = re.compile(r"[a-z0-9_:]+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


class BM25:
    def __init__(self, docs: Sequence[list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.n, 1)
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i]) or 1
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


@dataclass
class Hit:
    kind: str  # skill | case
    id: str
    title: str
    text: str
    score: float
    source: str


def _skill_doc(s: Skill) -> list[str]:
    return s.applies_to * 2 + tokenize(f"{s.name} {s.name} {s.description} {s.body[:4000]}")


def rank_skills(skills: list[Skill], query_tokens: list[str], features: list[str], k: int) -> list[Hit]:
    if not skills:
        return []
    bm = BM25([_skill_doc(s) for s in skills])
    base = bm.scores(query_tokens + features)
    scored = []
    for s, b in zip(skills, base):
        score = b * (0.5 + s.confidence) + 3.0 * jaccard(s.applies_to, features)
        scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return [
        Hit("skill", s.id, s.name, s.body, sc, s.origin)
        for sc, s in scored[:k]
        if sc > 0
    ]


def rank_cases(cases: list[Case], query_tokens: list[str], features: list[str], k: int, exclude_game: str = "") -> list[Hit]:
    pool = [c for c in cases if c.game_id != exclude_game and c.quality >= 0.2]
    if not pool:
        return []
    bm = BM25([c.features * 2 + tokenize(c.text()) for c in pool])
    base = bm.scores(query_tokens + features)
    scored = sorted(((b * (0.5 + c.quality) + 2.0 * jaccard(c.features, features), c) for b, c in zip(base, pool)), key=lambda x: -x[0])
    return [
        Hit("case", c.id, c.task_intent[:80], f"Approach: {c.approach}\nKey insight: {c.key_insight}", sc, "cases")
        for sc, c in scored[:k]
        if sc > 0
    ]


def rrf_fuse(lists: dict[str, list[Hit]], weights: dict[str, float], k: int, rrf_k: int = 10) -> list[Hit]:
    fused: dict[str, tuple[float, Hit]] = {}
    for src, hits in lists.items():
        w = weights.get(src, 1.0)
        for rank, h in enumerate(hits):
            key = f"{h.kind}:{h.title.lower()}"
            s = w / (rrf_k + rank + 1)
            if key in fused:
                prev_s, prev_h = fused[key]
                fused[key] = (prev_s + s, prev_h if prev_h.score >= h.score else h)
            else:
                fused[key] = (s, h)
    return [h for _, h in sorted(fused.values(), key=lambda x: -x[0])[:k]]


class GateOut(BaseModel):
    plan: str = ""
    selected: list[str] = Field(default_factory=list, max_length=2)


GATE_SYSTEM = (
    "You pick which stored game-playing skills apply to the current ARC-AGI-3 game. "
    "1. Plan: say in one sentence what the current game seems to require. "
    "2. Filter: keep a skill only if its procedure can be executed with the game's available actions. "
    "3. Match: vague topical overlap is not enough. "
    "4. Decide: select AT MOST 2 ids. Selecting an irrelevant or unexecutable skill is strictly worse than selecting none. "
    'Reply with JSON {"plan": str, "selected": [ids]}.'
)


async def gate(llm: Optional[ChatModel], hits: list[Hit], situation: str, max_select: int = 2) -> list[Hit]:
    if not hits:
        return []
    if llm is None:
        return hits[:max_select]
    menu = "\n".join(f"- {h.id}: {h.title} :: {h.text[:300].replace(chr(10), ' ')}" for h in hits)
    try:
        out = await call_json(llm, GateOut, GATE_SYSTEM, f"Situation:\n{situation}\n\nCandidates:\n{menu}", retries=1, max_tokens=400)
    except Exception:
        return hits[:max_select]
    by_id = {h.id: h for h in hits}
    return [by_id[i] for i in out.selected if i in by_id][:max_select]


async def retrieve(
    *,
    skills: list[Skill],
    cases: list[Case],
    query: str,
    features: list[str],
    llm: Optional[ChatModel] = None,
    k: int = 5,
    exclude_game: str = "",
    use_gate: bool = True,
) -> list[Hit]:
    q = tokenize(query)
    lists = {
        "library": rank_skills([s for s in skills if s.origin == "library"], q, features, k * 2),
        "runtime": rank_skills([s for s in skills if s.origin == "runtime"], q, features, k * 2),
        "cases": rank_cases(cases, q, features, k * 2, exclude_game),
    }
    fused = rrf_fuse(lists, {"library": 0.96, "runtime": 0.9, "cases": 0.85}, k)
    return await gate(llm, fused, query) if use_gate else fused[:2]


def render_hits(hits: Iterable[Hit]) -> str:
    blocks = []
    for h in hits:
        label = "Skill" if h.kind == "skill" else "Past case"
        blocks.append(f"### {label}: {h.title} [{h.id}]\n{h.text.strip()}")
    return "\n\n".join(blocks)
