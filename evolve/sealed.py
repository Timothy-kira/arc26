"""Sealed test split: scored only once, after the loop, never read by any decision.

The deliverable is chosen by train score (argmax over promoted nodes). Test scores
are reported as retention = (best_test - vanilla_test) / (best_train - vanilla_train).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .bench import Evaluator
from .state import RunState
from .stats import paired


def unseal(state: RunState, evaluator: Evaluator, k: int = 1) -> dict:
    spec = state.spec
    promoted = [n for n in state.nodes.values() if n.status == "promoted"]
    vanilla = state.nodes[state.meta["vanilla"]]
    best = max(promoted, key=lambda n: n.mean(spec.train_games))
    sealed_dir = Path(spec.work_dir) / "sealed"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    test = {}
    for n in {vanilla.id: vanilla, best.id: best}.values():
        test[n.id] = evaluator.eval(n, spec.test_games, k, "sealed")
        (sealed_dir / f"{n.id}.json").write_text(json.dumps(test[n.id]))
    res = paired(test[best.id], test[vanilla.id], spec.test_games)
    d_train = best.mean(spec.train_games) - vanilla.mean(spec.train_games)
    report = {
        "deliverable": best.id,
        "train": {"vanilla": vanilla.mean(spec.train_games), "best": best.mean(spec.train_games)},
        "test": {"vanilla": res.ctrl_mean, "best": res.cand_mean, "z": res.z, "wilcoxon_p": res.wilcoxon_p},
        "retention": (res.cand_mean - res.ctrl_mean) / d_train if d_train > 0 else None,
    }
    (sealed_dir / "report.json").write_text(json.dumps(report, indent=2))
    state.meta["unsealed_at"] = time.time()
    state.save()
    return report
