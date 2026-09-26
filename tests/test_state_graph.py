import numpy as np

from arc_harness.explore.config import ExploreConfig
from arc_harness.explore.state_graph import HudDetector, StateGraph


def grid_at(x, bar=0):
    g = np.zeros((64, 64), dtype=np.int8)
    g[30, x] = 9
    g[63, :bar] = 3
    return g


def test_bfs_to_frontier_and_winning_path():
    g = StateGraph([1, 2])  # 1 = right, 2 = left
    s0, s1, s2 = grid_at(10), grid_at(11), grid_at(12)
    g.visit(s0)
    g.record(s0, (1,), s1)
    g.record(s1, (2,), s0)
    g.record(s0, (2,), s0)  # noop
    # s1 still has untried (1,): path from s0 is [right, right]
    assert g.plan_to_frontier(g.key(s0)) == [(1,), (1,)]
    g.record(s1, (1,), s2, level_up=True)
    assert g.winning_path() == [(1,), (1,)]


def test_hud_mask_merges_states():
    cfg = ExploreConfig(hud_min_transitions=3)
    hud = HudDetector(cfg)
    g = StateGraph([1, 2], cfg, hud)
    grids = [grid_at(10, 0), grid_at(11, 1), grid_at(10, 2), grid_at(11, 3), grid_at(10, 4)]
    g.visit(grids[0])
    for i in range(4):
        g.record(grids[i], (1 if i % 2 == 0 else 2,), grids[i + 1])
    assert hud.mask[63].all()
    assert len(g.nodes) == 2
