from itertools import combinations

from perturb_metric import Brick, build_graph, perturbation_score
from brickgpt.stability_analysis.connectivity_analysis import _connected


def B(h, w, x, y, z):
    return Brick(h=h, w=w, x=x, y=y, z=z)


def test_tower_of_1x1():
    tower = [B(1, 1, 0, 0, z) for z in range(4)]
    r = perturbation_score(tower)
    assert r['baseline_brick_frac'] == 1.0
    assert r['single_brick_min'] == 0.0          # removing the bottom disconnects all 3 others
    assert r['single_voxel_min'] == 0.0
    # removals at z=0,1,2,3 leave 0/3, 1/3, 2/3, 3/3 connected
    assert abs(r['single_brick_mean'] - 0.5) < 1e-9
    assert r['critical_frac'] == 0.75


def test_staggered_2x4_wall():
    wall = []
    for z in (0, 2):
        wall += [B(2, 4, 0, y, z) for y in (0, 4, 8)]
    wall += [B(2, 2, 0, 0, 1), B(2, 4, 0, 2, 1), B(2, 4, 0, 6, 1), B(2, 2, 0, 10, 1)]
    r = perturbation_score(wall)
    assert r['single_brick_min'] == 1.0
    assert r['single_voxel_min'] == 1.0
    assert r['critical_frac'] == 0.0
    assert r['single_exhaustive']


def test_stacked_column_not_staggered():
    # straight-stacked 2x4s (no bond): removing a ground brick drops its column
    wall = [B(2, 4, 0, y, z) for z in range(3) for y in (0, 4)]
    r = perturbation_score(wall)
    assert abs(r['single_brick_min'] - 3 / 5) < 1e-9


def test_1x3_both_orientations():
    s = [B(1, 3, 0, 0, 0),   # 1x3 along y, cells (0,0..2)
         B(3, 1, 0, 2, 1),   # 3x1 along x, cells (0..2,2) -> overlaps at (0,2)
         B(1, 1, 2, 2, 2),   # on the far end of the 3x1
         B(1, 1, 2, 0, 1)]   # floating: nothing below (2,0,0)
    adj, ground = build_graph(s)
    assert adj[0] == {1} and adj[1] == {0, 2} and adj[3] == set()
    r = perturbation_score(s)
    assert abs(r['baseline_brick_frac'] - 0.75) < 1e-9
    assert abs(r['baseline_voxel_frac'] - 7 / 8) < 1e-9
    # 3x1 offset by one: (1..3, 2) does NOT overlap the 1x3 at (0, 0..2)
    adj2, _ = build_graph([B(1, 3, 0, 0, 0), B(3, 1, 1, 2, 1)])
    assert adj2[0] == set()


def test_matches_brickgpt_connected_rule():
    import random
    rng = random.Random(1)
    dims = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 4), (4, 2), (1, 6), (6, 1)]
    for _ in range(30):
        s, occ = [], set()
        for _ in range(25):
            h, w = rng.choice(dims)
            x, y, z = rng.randrange(6), rng.randrange(6), rng.randrange(4)
            cells = {(x + i, y + j, z) for i in range(h) for j in range(w)}
            if cells & occ:
                continue
            occ |= cells
            s.append(B(h, w, x, y, z))
        adj, _ = build_graph(s)
        for i, j in combinations(range(len(s)), 2):
            assert (j in adj[i]) == _connected(s[i], s[j])


if __name__ == '__main__':  # plain runner (pytest is not installed in the venv)
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print('PASS', t.__name__)
        except Exception:
            failed += 1
            print('FAIL', t.__name__)
            traceback.print_exc()
    print(f'{len(tests) - failed}/{len(tests)} passed')
    raise SystemExit(failed)
