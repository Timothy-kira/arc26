"""Per-game working memory, doubling as the context-compaction "handoff brief".

Raven compacts a long transcript into a brief of goal / confirmed facts / what was
done / results / remaining plan. Here the brief *is* the memory: every LLM call is
built from ``brief()`` plus a short window of recent steps, so no transcript grows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HypothesisEntry:
    statement: str
    kind: str = "mechanic"
    status: str = "open"  # open | confirmed | refuted
    confidence: float = 0.5
    evidence: str = ""


@dataclass
class LevelNotes:
    level: int
    goal: str = ""
    hypotheses: list[HypothesisEntry] = field(default_factory=list)
    deaths: list[str] = field(default_factory=list)
    actions: int = 0
    won: bool = False
    winning_path: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class StepLine:
    n: int
    action: str
    result: str


class StepLog:
    """Rolling window of recent steps; older ones are elided (Raven's tool-output pruning)."""

    def __init__(self, keep: int = 24) -> None:
        self.keep = keep
        self.lines: deque[StepLine] = deque(maxlen=keep)
        self.total = 0

    def add(self, action: str, result: str) -> None:
        self.total += 1
        self.lines.append(StepLine(self.total, action, result[:240]))

    def render(self, n: Optional[int] = None) -> str:
        items = list(self.lines)[-(n or self.keep):]
        head = f"[{self.total - len(items)} earlier steps elided]\n" if self.total > len(items) else ""
        return head + "\n".join(f"{s.n:4d}. {s.action:14s} -> {s.result}" for s in items)


@dataclass
class GameNotebook:
    game_id: str
    features: list[str] = field(default_factory=list)
    game_rules: list[str] = field(default_factory=list)
    levels: dict[int, LevelNotes] = field(default_factory=dict)
    skills_in_use: list[str] = field(default_factory=list)
    steps: StepLog = field(default_factory=StepLog)

    def level(self, n: int) -> LevelNotes:
        if n not in self.levels:
            self.levels[n] = LevelNotes(n)
        return self.levels[n]

    def add_rule(self, rule: str) -> None:
        rule = rule.strip()
        if rule and rule not in self.game_rules:
            self.game_rules.append(rule)
            del self.game_rules[:-20]

    def update_hypotheses(self, level: int, confirmed: list[str], refuted: list[str], new: list[HypothesisEntry]) -> None:
        notes = self.level(level)
        index = {h.statement: h for h in notes.hypotheses}
        for s in confirmed:
            h = index.get(s) or HypothesisEntry(s)
            h.status, h.confidence = "confirmed", 0.9
            index[s] = h
            self.add_rule(s)
        for s in refuted:
            h = index.get(s) or HypothesisEntry(s)
            h.status, h.confidence = "refuted", 0.05
            index[s] = h
        for h in new:
            if h.statement not in index:
                index[h.statement] = h
        notes.hypotheses = list(index.values())[-16:]

    def brief(self, level: int, max_chars: int = 3000) -> str:
        notes = self.level(level)
        parts = [f"GAME {self.game_id} — level {level + 1} (levels already won: {sum(l.won for l in self.levels.values())})"]
        if notes.goal:
            parts.append(f"Current goal: {notes.goal}")
        if self.game_rules:
            parts.append("Confirmed rules (this game):\n" + "\n".join(f"- {r}" for r in self.game_rules[-12:]))
        open_h = [h for h in notes.hypotheses if h.status == "open"]
        if open_h:
            parts.append("Open hypotheses:\n" + "\n".join(f"- ({h.confidence:.1f}) {h.statement}" for h in open_h[-8:]))
        ref = [h for h in notes.hypotheses if h.status == "refuted"]
        if ref:
            parts.append("Refuted (do not retry):\n" + "\n".join(f"- {h.statement}" for h in ref[-6:]))
        if notes.deaths:
            parts.append("Deaths so far this level:\n" + "\n".join(f"- {d}" for d in notes.deaths[-5:]))
        prev = [l for l in self.levels.values() if l.won and l.summary]
        if prev:
            parts.append("How earlier levels were solved:\n" + "\n".join(f"- L{l.level + 1}: {l.summary}" for l in prev[-3:]))
        text = "\n\n".join(parts)
        return text[-max_chars:]

    def to_markdown(self) -> str:
        lines = [f"# {self.game_id}", "", "features: " + " ".join(self.features), "", "## Rules"]
        lines += [f"- {r}" for r in self.game_rules]
        for n, l in sorted(self.levels.items()):
            lines += ["", f"## Level {n + 1} ({'won' if l.won else 'unsolved'}, {l.actions} actions)"]
            if l.goal:
                lines.append(f"goal: {l.goal}")
            if l.summary:
                lines.append(f"summary: {l.summary}")
            for h in l.hypotheses:
                lines.append(f"- [{h.status}] {h.statement}")
            if l.winning_path:
                lines.append("winning path: " + " ".join(l.winning_path))
        return "\n".join(lines) + "\n"
