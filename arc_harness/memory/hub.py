"""Shared long-term memory for a run: skills (library + runtime), cases, clusters."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..llm.client import ChatModel
from .cases import CaseStore
from .sediment import Clusters, Sedimenter
from .skill_store import SkillStore

LIBRARY_DIR = Path(__file__).resolve().parents[2] / "skills"


class MemoryHub:
    def __init__(self, root: Path, llm: Optional[ChatModel], library_roots: Iterable[Path] = (LIBRARY_DIR,)) -> None:
        root = Path(root)
        self.root = root
        self.skills = SkillStore(root / "skills", library_roots)
        self.cases = CaseStore(root / "cases")
        self.clusters = Clusters(root / "clusters.json")
        self.sedimenter = Sedimenter(self.skills, self.cases, self.clusters, llm)
