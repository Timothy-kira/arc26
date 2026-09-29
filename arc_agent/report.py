"""Summary of a run directory: per game levels, actions per level against the human baseline, RHAE
(as the scorecard computes it) and how the loop behaved (steps, prediction accuracy, LLM time).

    python -m arc_agent.report runs/a1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc_eval.eval import rhae  # noqa: E402


def report(run: Path) -> str:
    rows, scores = [], []
    tot = {"steps": 0, "right": 0, "wrong": 0, "errors": 0, "llm_s": 0.0, "out_tokens": 0}
    for d in sorted((run / "games").glob("*_k*")):
        r = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
        steps = [json.loads(l) for l in open(d / "steps.jsonl")] if (d / "steps.jsonl").exists() else []
        la, base = r.get("level_actions") or [], r.get("baseline") or []
        sc = rhae(la, tuple(base)) if base else None
        scores.append(sc or 0.0)
        right = sum(1 for s in steps if s.get("ok") is True)
        wrong = sum(1 for s in steps if s.get("ok") is False)
        errs = sum(1 for s in steps if s.get("error"))
        llm = sum((s.get("timing") or {}).get("llm") or s.get("llm_s") or 0 for s in steps)
        out_tok = sum(int((s.get("usage") or {}).get("completion_tokens") or 0) for s in steps)
        tot["out_tokens"] += out_tok
        tot.update(steps=tot["steps"] + len(steps), right=tot["right"] + right, wrong=tot["wrong"] + wrong,
                   errors=tot["errors"] + errs, llm_s=tot["llm_s"] + llm)
        ratio = " ".join(f"{n}/{base[i]}" if i < len(base) else str(n) for i, n in enumerate(la))
        rows.append(f"{d.name[:18]:18} {r.get('levels', '?')}/{r.get('win_levels', '?'):<3} {r.get('actions', '?'):>5}  "
                    f"{ratio[:34]:34} {('%.2f' % sc) if sc is not None else '-':>6}  steps {len(steps):4} "
                    f"pred {right}/{right + wrong}  err {errs}  llm {llm / max(1, len(steps)):.0f}s/step  "
                    f"out {out_tok / max(1, len(steps)):.0f} tok/step ({out_tok // 1000}k)")
    head = f"{'game':18} lv    acts  actions/human per level              RHAE"
    n = max(1, len(scores))
    acc = tot["right"] / max(1, tot["right"] + tot["wrong"])
    return "\n".join([head, *rows, "", f"games {len(scores)}  mean RHAE {sum(scores) / n:.3f}  steps {tot['steps']}  "
                      f"prediction accuracy {acc:.0%}  answer errors {tot['errors']}  "
                      f"LLM {tot['llm_s'] / max(1, tot['steps']):.0f}s/step  output {tot['out_tokens'] / max(1, tot['steps']):.0f} "
                      f"tokens/step, {tot['out_tokens'] / n / 1000:.0f}k per game"])


if __name__ == "__main__":
    print(report(Path(sys.argv[1])))
