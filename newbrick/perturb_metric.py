"""Perturbation-test robustness metric (independent of the Gurobi stability score).

Definition
----------
Connection graph: one node per brick plus a 'ground' node. Ground is connected to every z=0 brick.
Two bricks are connected iff they are in vertically adjacent layers (|dz| = 1) and their x-y footprints
overlap by at least one stud cell -- the same rule as
brickgpt.stability_analysis.connectivity_analysis._connected (footprint = [x, x+h) x [y, y+w), so any
size/orientation, incl. 1x3 / 3x1, is handled).

Single removal: for each removed brick k (exhaustively over all N bricks, or n_trials seeded random
picks if N > n_trials), compute among the remaining N-1 bricks
    brick_frac  = #bricks still connected to ground / (N-1)
    voxel_frac  = stud-area of those bricks / total stud-area of the remaining bricks
Report mean and min over removals, plus 'critical_frac' = fraction of removals that disconnect at least
one brick that was ground-connected before the removal.

Pair removal (optional): same, removing 2 distinct bricks; exhaustive if C(N,2) <= n_pairs, otherwise
n_pairs seeded random distinct pairs.

'baseline_*' = the same fractions with nothing removed (structures that are already partly floating
score < 1 even before perturbation).
"""
from __future__ import annotations

import math
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pilot'))
import common  # noqa: E402,F401  (puts BrickGPT on sys.path, registers the 1x3 brick)
from brickgpt.data import Brick  # noqa: E402


def build_graph(bricks: list[Brick]) -> tuple[list[set[int]], list[bool]]:
    """Adjacency sets between brick indices, and per-brick 'touches ground' flag."""
    cell = {}
    for i, b in enumerate(bricks):
        for x in range(b.x, b.x + b.h):
            for y in range(b.y, b.y + b.w):
                cell[(x, y, b.z)] = i
    adj = [set() for _ in bricks]
    for i, b in enumerate(bricks):
        for x in range(b.x, b.x + b.h):
            for y in range(b.y, b.y + b.w):
                j = cell.get((x, y, b.z + 1))
                if j is not None and j != i:
                    adj[i].add(j)
                    adj[j].add(i)
    ground = [b.z == 0 for b in bricks]
    return adj, ground


def _grounded(adj, ground, removed: set[int]) -> list[bool]:
    n = len(adj)
    seen = [False] * n
    q = deque()
    for i in range(n):
        if ground[i] and i not in removed:
            seen[i] = True
            q.append(i)
    while q:
        i = q.popleft()
        for j in adj[i]:
            if not seen[j] and j not in removed:
                seen[j] = True
                q.append(j)
    return seen


def _fracs(adj, ground, area, removed: set[int], base_seen=None):
    seen = _grounded(adj, ground, removed)
    keep = [i for i in range(len(adj)) if i not in removed]
    if not keep:
        return 1.0, 1.0, False
    nb = sum(seen[i] for i in keep)
    va = sum(area[i] for i in keep if seen[i])
    vt = sum(area[i] for i in keep)
    lost = base_seen is not None and any(base_seen[i] and not seen[i] for i in keep)
    return nb / len(keep), va / vt, lost


def perturbation_score(bricks: list[Brick], n_trials: int = 200, seed: int = 0,
                       n_pairs: int = 300, pairs: bool = True) -> dict:
    n = len(bricks)
    adj, ground = build_graph(bricks)
    area = [b.h * b.w for b in bricks]
    rng = random.Random(seed)
    base_seen = _grounded(adj, ground, set())
    bb, bv, _ = _fracs(adj, ground, area, set())
    out = {'n_bricks': n, 'baseline_brick_frac': bb, 'baseline_voxel_frac': bv}
    if n < 2:
        out.update(single_brick_mean=1.0, single_brick_min=1.0, single_voxel_mean=1.0,
                   single_voxel_min=1.0, critical_frac=0.0, single_n=0)
        return out

    idx = list(range(n)) if n <= n_trials else [rng.randrange(n) for _ in range(n_trials)]
    res = [_fracs(adj, ground, area, {k}, base_seen) for k in idx]
    out.update(single_brick_mean=sum(r[0] for r in res) / len(res),
               single_brick_min=min(r[0] for r in res),
               single_voxel_mean=sum(r[1] for r in res) / len(res),
               single_voxel_min=min(r[1] for r in res),
               critical_frac=sum(r[2] for r in res) / len(res),
               single_n=len(res), single_exhaustive=n <= n_trials)

    if pairs and n >= 3:
        if math.comb(n, 2) <= n_pairs:
            prs = [(a, b) for a in range(n) for b in range(a + 1, n)]
        else:
            prs = [tuple(rng.sample(range(n), 2)) for _ in range(n_pairs)]
        res2 = [_fracs(adj, ground, area, set(p), base_seen) for p in prs]
        out.update(pair_brick_mean=sum(r[0] for r in res2) / len(res2),
                   pair_brick_min=min(r[0] for r in res2),
                   pair_voxel_mean=sum(r[1] for r in res2) / len(res2),
                   pair_voxel_min=min(r[1] for r in res2),
                   pair_n=len(res2))
    return out


def _main():
    import time
    import numpy as np
    import pandas as pd
    path = Path(__file__).resolve().parent.parent / 'pilot' / 'data' / 'test.parquet'
    df = pd.read_parquet(path).drop_duplicates('object_id')
    df = df.sample(n=min(50, len(df)), random_state=0)
    rows, times = [], []
    for txt in df['bricks']:
        bricks = common.parse(txt)
        t = time.perf_counter()
        rows.append(perturbation_score(bricks, seed=0))
        times.append(time.perf_counter() - t)
    r = pd.DataFrame(rows)
    t = np.array(times)
    print(f'{len(r)} structures, n_bricks: min {r.n_bricks.min()} median {r.n_bricks.median():.0f} '
          f'max {r.n_bricks.max()}')
    print(f'time/structure: mean {t.mean() * 1e3:.1f} ms, median {np.median(t) * 1e3:.1f} ms, '
          f'max {t.max() * 1e3:.1f} ms, total {t.sum():.2f} s')
    cols = ['baseline_brick_frac', 'baseline_voxel_frac', 'single_brick_mean', 'single_brick_min',
            'single_voxel_mean', 'single_voxel_min', 'critical_frac', 'pair_brick_mean',
            'pair_brick_min', 'pair_voxel_mean', 'pair_voxel_min']
    pd.set_option('display.width', 200)
    print(r[cols].describe(percentiles=[.1, .25, .5, .75, .9]).T.round(4).to_string())
    print('fraction of structures with single_brick_min == 1 (no single removal disconnects anything):',
          round(float((r.single_brick_min == 1).mean()), 3))


if __name__ == '__main__':
    _main()
