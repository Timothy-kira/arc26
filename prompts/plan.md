You receive several analysts' hypotheses about the current game, plus the notebook of what is already known. Consolidate them and decide what to do next.

1. Mark hypotheses that the evidence already confirms or refutes (quote them verbatim).
2. State the current best guess of the level goal in one sentence.
3. Produce a concrete action plan of at most {max_actions} actions that either makes progress toward the goal or runs the single most informative experiment. Use explicit coordinates for clicks, taken from object positions.
4. Give action priorities (0..2) for the low-level explorer to use when your plan is exhausted, and list actions to avoid.
5. Say what you expect to observe if the plan works.

Reply with ONLY a JSON object:
{"goal": str,
 "confirmed": [str], "refuted": [str],
 "new_hypotheses": [{"statement": str, "kind": str, "confidence": float, "test": str}],
 "actions": [{"action": 1-7, "x": int|null, "y": int|null, "repeat": int}],
 "priorities": {"1": float, ...},
 "avoid": [{"action": 1-7, "x": int|null, "y": int|null}],
 "expect": str,
 "mode": "execute" | "explore"}
Use "mode": "explore" to hand control back to the systematic explorer after your actions.
