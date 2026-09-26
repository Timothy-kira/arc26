"""Gate statistics on per-game scores (continuous RHAE, so paired tests replace Fisher)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev, stdev


def per_game_mean(scores: dict[str, list[float]], games: list[str]) -> dict[str, float]:
    return {g: (sum(scores[g]) / len(scores[g]) if scores.get(g) else 0.0) for g in games}


def screen_sigma(baseline: dict[str, list[float]], games: list[str]) -> float:
    """sigma of the anchor mean under baseline noise: sqrt(sum var_i) / n."""
    var = sum(pstdev(baseline[g]) ** 2 if len(baseline.get(g, [])) > 1 else 0.0 for g in games)
    return math.sqrt(var) / max(len(games), 1)


def screen_cull(cand: dict[str, float], baseline: dict[str, list[float]], games: list[str], k_sigma: float) -> tuple[bool, str]:
    base = per_game_mean(baseline, games)
    c = mean(cand.get(g, 0.0) for g in games)
    b = mean(base.values())
    sig = max(screen_sigma(baseline, games), 1e-9)
    if c < b - k_sigma * sig:
        return True, f"anchor mean {c:.3f} < baseline {b:.3f} - {k_sigma}*{sig:.3f}"
    return False, f"anchor mean {c:.3f} vs baseline {b:.3f} (sigma {sig:.3f})"


@dataclass
class PairedResult:
    n: int
    mean_diff: float
    z: float
    wilcoxon_p: float
    cand_mean: float
    ctrl_mean: float

    @property
    def credited_2sigma(self) -> bool:
        return self.z >= 2.0


def _wilcoxon_p(diffs: list[float]) -> float:
    """One-sided (cand > ctrl) signed-rank test, normal approximation, ties averaged."""
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mu) / sd if sd else 0.0
    return 0.5 * math.erfc(z / math.sqrt(2))


def paired(cand: dict[str, list[float]], ctrl: dict[str, list[float]], games: list[str]) -> PairedResult:
    c = per_game_mean(cand, games)
    b = per_game_mean(ctrl, games)
    d = [c[g] - b[g] for g in games]
    n = len(d)
    md = mean(d) if d else 0.0
    sd = stdev(d) if n > 1 else 0.0
    z = md / (sd / math.sqrt(n)) if sd > 0 else (float("inf") if md > 0 else 0.0)
    return PairedResult(n, md, z, _wilcoxon_p(d), mean(c.values()) if c else 0.0, mean(b.values()) if b else 0.0)
