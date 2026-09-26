"""Load role prompts from ``prompts/*.md`` (the evolver edits these files)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(os.getenv("ARC26_PROMPTS_DIR") or Path(__file__).resolve().parents[1] / "prompts")


@lru_cache(maxsize=None)
def load(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text().strip()


def fill(name: str, **kw: object) -> str:
    text = load(name)
    for k, v in kw.items():
        text = text.replace("{" + k + "}", str(v))
    return text
