"""Single-pass placeholder rendering (Raven ``dag_render.py``).

Substituted text is never re-scanned, and upstream outputs are fenced as
untrusted data so a node cannot smuggle instructions into its dependents.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

_ANY = re.compile(
    r"\{\{\s*(?:(?P<ref>[A-Za-z0-9_\-]+)\.(?P<kind>output|output_path)|input\.(?P<inp>[A-Za-z0-9_\-]+))\s*\}\}"
)


def fence(node_id: str, text: str) -> str:
    return f'<upstream-data node="{node_id}">\n{text}\n</upstream-data>'


def render_prompt(
    template: str,
    outputs: Mapping[str, str],
    inputs: Mapping[str, Any],
    paths: Optional[Mapping[str, str]] = None,
) -> str:
    def sub(m: re.Match[str]) -> str:
        if m.group("inp"):
            v = inputs.get(m.group("inp"), "")
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        ref, kind = m.group("ref"), m.group("kind")
        if kind == "output_path":
            return (paths or {}).get(ref, f"<no path for {ref}>")
        return fence(ref, outputs.get(ref, ""))

    return _ANY.sub(sub, template)
