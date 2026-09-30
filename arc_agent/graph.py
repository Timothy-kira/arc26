"""The game's reasoning graph: the only memory the agent has between steps.

Every step starts from a fresh context, so what the agent knows lives here as typed nodes joined
by typed edges (a small symbolic network). The model must add at least one node per step and reads
the whole graph at the start of every step. The loop adds its own nodes too: each executed action
and whether the prediction made for it came true.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

NODE_TYPES = ("observation", "rule", "hypothesis", "goal", "plan", "question", "action", "outcome", "simulator")
EDGE_TYPES = ("supports", "refutes", "causes", "part_of", "leads_to", "about", "tests", "revises")
STATUSES = ("open", "confirmed", "refuted", "done")


@dataclass
class Node:
    id: int
    type: str
    text: str
    level: int
    step: int
    status: str = "open"


@dataclass
class Edge:
    src: int
    dst: int
    rel: str


@dataclass
class Graph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, type_: str, text: str, level: int, step: int, status: str = "open") -> int:
        nid = len(self.nodes)
        self.nodes.append(Node(nid, type_, text.strip()[:300], level, step, status))
        return nid

    def add_edge(self, src: int, dst: int, rel: str) -> None:
        if 0 <= src < len(self.nodes) and 0 <= dst < len(self.nodes) and src != dst:
            e = Edge(src, dst, rel)
            if e not in self.edges:
                self.edges.append(e)

    def apply(self, update: dict[str, Any], level: int, step: int) -> tuple[list[int], list[str]]:
        """Apply the model's graph_update. New nodes may be referenced by the edges of the same update
        as "n0", "n1", ... (their position in ``nodes``) or by an existing integer id. Returns the new
        ids and a list of problems (bad types, unknown ids) the model is told about next step."""
        problems: list[str] = []
        new_ids: list[int] = []
        for i, n in enumerate(update.get("nodes") or []):
            t = str(n.get("type", "")).lower()
            text = str(n.get("text", "")).strip()
            if t not in NODE_TYPES or not text:
                problems.append(f"node {i} skipped: type must be one of {', '.join(NODE_TYPES[:6])} and text non-empty")
                continue
            new_ids.append(self.add_node(t, text, level, step))

        def ref(v: Any) -> Optional[int]:
            if isinstance(v, str) and v.startswith("n") and v[1:].isdigit():
                k = int(v[1:])
                return new_ids[k] if k < len(new_ids) else None
            try:
                k = int(v)
            except (TypeError, ValueError):
                return None
            return k if 0 <= k < len(self.nodes) else None

        for e in update.get("edges") or []:
            s, d, rel = ref(e.get("src")), ref(e.get("dst")), str(e.get("rel", "")).lower()
            if s is None or d is None or rel not in EDGE_TYPES:
                problems.append(f"edge {e} skipped: unknown node id or rel not in {', '.join(EDGE_TYPES)}")
                continue
            self.add_edge(s, d, rel)
        for u in update.get("status") or []:
            k, st = ref(u.get("id")), str(u.get("status", "")).lower()
            if k is None or st not in STATUSES:
                problems.append(f"status {u} skipped")
                continue
            self.nodes[k].status = st
        return new_ids, problems

    def render(self, max_chars: int = 60000, recent: int = 10, obs_window: int = 15) -> str:
        """The graph as text: nodes grouped by level, each with its outgoing edges. Everything the model
        wrote that states knowledge (rules, goals, hypotheses, plans, questions, and observations that
        were confirmed, refuted or linked) is always shown. What only records history is folded:
        action/outcome pairs of earlier levels, and in the current level the RIGHT pairs older than the
        last ``recent`` actions (WRONG ones stay: they are what was learned) and unlinked open
        observations and plans older than ``obs_window`` steps (earlier levels: all unlinked open ones)."""
        out_edges: dict[int, list[Edge]] = {}
        linked: set[int] = set()
        for e in self.edges:
            out_edges.setdefault(e.src, []).append(e)
            if self.nodes[e.src].type not in ("action", "outcome"):  # the loop's own tests-edges do not count
                linked.add(e.dst)
        lines: list[str] = []
        folded_note: list[str] = []
        levels = sorted({n.level for n in self.nodes})
        current = levels[-1] if levels else 0
        last_step = max((n.step for n in self.nodes), default=0)
        for lv in levels:
            lines.append(f"## level {lv + 1}")
            nodes = [n for n in self.nodes if n.level == lv]
            acts = [n for n in nodes if n.type == "action"]
            keep: set[int] = set()
            if lv == current:
                outcome_of = {e.src: e.dst for e in self.edges if e.rel == "leads_to"}
                for a in acts[-recent:]:
                    keep |= {a.id, outcome_of.get(a.id, -1)}
                for a in acts:
                    o = outcome_of.get(a.id)
                    if o is not None and self.nodes[o].status == "refuted":
                        keep |= {a.id, o}
            folded_acts = sum(1 for a in acts if a.id not in keep)
            shown = []
            folded_obs = 0
            for n in nodes:
                if n.type in ("action", "outcome"):
                    if n.id in keep:
                        shown.append(n)
                elif n.type in ("observation", "plan") and n.status == "open" and n.id not in linked and (
                        lv < current or last_step - n.step > obs_window):
                    folded_obs += 1  # superseded snapshots and plans
                else:
                    shown.append(n)
            if folded_acts or folded_obs:  # noted at the end so the text above stays stable (prefix cache)
                folded_note.append(f"level {lv + 1}: {folded_acts} earlier actions whose predictions came true and "
                                   f"{folded_obs} old observations/plans are folded")
            # knowledge first (it only grows, so the text stays stable for the prefix cache), then the
            # level's action history (which shifts as old RIGHT pairs fold)
            for n in sorted(shown, key=lambda n: (n.type in ("action", "outcome"), n.id)):
                edges = "".join(f" -{e.rel}->{e.dst}" for e in out_edges.get(n.id, []))
                mark = "" if n.status == "open" else f" [{n.status}]"
                text = "RIGHT" if n.type == "outcome" and n.status == "confirmed" else n.text
                lines.append(f"  {n.id} {n.type}{mark} s{n.step}: {text}{edges}")
        if folded_note:
            lines.append("(" + "; ".join(folded_note) + ")")
        text = "\n".join(lines) or "(empty graph: this is the first step)"
        if len(text) > max_chars:  # keep the head (rules, goals) and the most recent part
            text = text[: max_chars // 3] + "\n  ...\n" + text[-(2 * max_chars) // 3:]
        return text

    def lines(self, ids: list[int]) -> str:
        """The given nodes as graph lines (same format as ``render``), for a step that continues a
        conversation and only needs what was added since the last one."""
        out_edges: dict[int, list[Edge]] = {}
        for e in self.edges:
            out_edges.setdefault(e.src, []).append(e)
        rows = []
        for i in ids:
            if 0 <= i < len(self.nodes):
                n = self.nodes[i]
                edges = "".join(f" -{e.rel}->{e.dst}" for e in out_edges.get(n.id, []))
                mark = "" if n.status == "open" else f" [{n.status}]"
                rows.append(f"  {n.id} {n.type}{mark} s{n.step}: {n.text}{edges}")
        return "\n".join(rows)

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"nodes": [asdict(n) for n in self.nodes],
                                   "edges": [asdict(e) for e in self.edges]}, ensure_ascii=False))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Graph":
        d = json.loads(path.read_text())
        return cls([Node(**n) for n in d["nodes"]], [Edge(**e) for e in d["edges"]])
