"""DAG data model, static validation and Kahn topological order.

Mirrors Raven's ``raven/agent/subagent/dag_graph.py``: the whole graph is
validated before anything is dispatched, ids are unique across the session,
``{{ x.output }}`` may only name a declared dependency, every declared input
must be referenced, and cycles are rejected.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_\-]+)\.(output|output_path)\s*\}\}")
INPUT_RE = re.compile(r"\{\{\s*input\.([A-Za-z0-9_\-]+)\s*\}\}")


class DagNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    role: str
    summary: str = ""
    prompt_template: str = ""
    depends_on: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    instance: Optional[str] = None


class DagSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_summary: str = ""
    nodes: list[DagNode]


class DagValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def collect_static_errors(
    spec: DagSpec,
    *,
    roles: Iterable[str],
    used_ids: Iterable[str] = (),
    completed: Iterable[str] = (),
) -> list[str]:
    errors: list[str] = []
    roles = set(roles)
    used = {u.casefold() for u in used_ids}
    completed = set(completed)
    local: dict[str, DagNode] = {}
    for n in spec.nodes:
        if not ID_RE.match(n.id):
            errors.append(f"node id {n.id!r} must match {ID_RE.pattern}")
        k = n.id.casefold()
        if k in {i.casefold() for i in local}:
            errors.append(f"duplicate node id {n.id!r}")
        if k in used:
            errors.append(f"node id {n.id!r} was already used earlier in this session")
        if n.role not in roles:
            errors.append(f"node {n.id!r}: unknown role {n.role!r} (known: {sorted(roles)})")
        local[n.id] = n
    for n in spec.nodes:
        for d in n.depends_on:
            if d not in local and d not in completed:
                errors.append(f"node {n.id!r} depends on unknown node {d!r}")
            if d == n.id:
                errors.append(f"node {n.id!r} depends on itself")
        for ref, _ in PLACEHOLDER_RE.findall(n.prompt_template):
            if ref not in n.depends_on:
                errors.append(f"node {n.id!r} references {{{{{ref}.output}}}} without depending on it")
        used_inputs = set(INPUT_RE.findall(n.prompt_template))
        for key in n.inputs:
            if key not in used_inputs:
                errors.append(f"node {n.id!r}: input {key!r} is declared but never referenced")
        for key in used_inputs - set(n.inputs):
            errors.append(f"node {n.id!r}: template references undeclared input {key!r}")
    return errors


def topological_order(spec: DagSpec, completed: Iterable[str] = ()) -> list[str]:
    completed = set(completed)
    by_id = {n.id: n for n in spec.nodes}
    indeg = {i: sum(1 for d in n.depends_on if d in by_id) for i, n in by_id.items()}
    children: dict[str, list[str]] = {i: [] for i in by_id}
    for n in spec.nodes:
        for d in n.depends_on:
            if d in by_id:
                children[d].append(n.id)
    q = deque(sorted(i for i, v in indeg.items() if v == 0))
    order: list[str] = []
    while q:
        i = q.popleft()
        order.append(i)
        for c in children[i]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(order) != len(by_id):
        raise DagValidationError(["graph contains a cycle"])
    return order


def validate_and_order(
    spec: DagSpec,
    *,
    roles: Iterable[str],
    used_ids: Iterable[str] = (),
    completed: Iterable[str] = (),
) -> list[str]:
    errors = collect_static_errors(spec, roles=roles, used_ids=used_ids, completed=completed)
    if errors:
        raise DagValidationError(errors)
    return topological_order(spec, completed)
