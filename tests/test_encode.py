import numpy as np

from arc_harness.env.encode import background_color, diff, frame_hash, objects


def _grid():
    g = np.full((64, 64), 4, dtype=np.int8)
    g[10:13, 10:13] = 9
    g[30, 40] = 11
    return g


def test_objects_and_background():
    g = _grid()
    assert background_color(g) == 4
    objs = objects(g)
    assert {(o.color, o.size) for o in objs} == {(9, 9), (11, 1)}
    blue = next(o for o in objs if o.color == 9)
    assert blue.center == (11, 11)


def test_diff_detects_move_and_noop():
    a = _grid()
    b = a.copy()
    b[10:13, 10:13] = 4
    b[10:13, 14:17] = 9
    d = diff(a, b)
    assert len(d.moved) == 1 and d.moved[0][1].center == (15, 11)
    assert diff(a, a.copy()).is_noop


def test_frame_hash_respects_mask():
    a = _grid()
    b = a.copy()
    b[63, 0] = 1
    mask = np.zeros_like(a, dtype=bool)
    mask[63, :] = True
    assert frame_hash(a) != frame_hash(b)
    assert frame_hash(a, mask) == frame_hash(b, mask)
