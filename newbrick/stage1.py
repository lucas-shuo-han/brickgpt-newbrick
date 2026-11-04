"""Stage 1: is the 1x3 brick physically useful? (no training)

For each shape (voxels taken from a StableText2Brick TRAIN structure), run BrickGPT's own
split-and-remerge layout code (STABLE/src/mesh2brick, voxel2brick) K times with different seeds:
  group A: original 8-brick library
  group B: library + 1x3 (id 99, part 3622, mass 0.00119 = midpoint of 1x2 and 1x4)
Each layout is scored with Gurobi (BrickGPT's stability_score; per-brick score, >=1 means unstable)
and STABLE's interlocking / seam / connectivity metrics.

Usage: python stage1.py <group A|B> <n_shapes> <K> <out.jsonl>   (appends; skips finished (shape, seed))
"""
import json
import os
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / 'STABLE' / 'src' / 'mesh2brick' / 'src'))
sys.path.insert(0, str(HERE.parent / 'pilot'))
sys.path.insert(0, str(HERE))
sys.modules['open3d'] = types.ModuleType('open3d')  # only needed for mesh input; we pass voxels

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GROUP, N, K, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])

import mesh2brick.data.brick_library as mbl  # noqa: E402
# New parts. Masses are estimates, interpolated/extrapolated from the library's own masses:
#   1x3 = mean(1x2, 1x4); 2x3 = mean(2x2, 2x4); 1x10 / 1x12 = 1x8 + 0.000375 per extra stud (1x6->1x8 slope).
NEW_PARTS = {'B': ('99', 1, 3, 0.00119, '3622.DAT'), 'B2x3': ('98', 2, 3, 0.001655, '3002.DAT'),
             'B1x10': ('97', 1, 10, 0.003780, '6111.DAT'), 'B1x12': ('96', 1, 12, 0.004530, '6112.DAT')}
NEW = NEW_PARTS.get(GROUP)


def _add_part(lib_mod, part):
    pid, h, w, mass, part_id = part
    lib_mod.brick_library[pid] = {'height': h, 'width': w, 'mass': mass, 'inventory': 100000, 'partID': part_id}
    lib_mod._dimensions_to_brick_id_dict[(h, w)] = int(pid)


if NEW:
    _add_part(mbl, NEW)
from mesh2brick.voxel2brick import voxel2brick, Voxel2Brick  # noqa: E402

if os.environ.get('JITTER') == '1':
    # Original greedy priority ends with (x, y, z), so ties are broken by position and every seed gives the
    # same initial layout. Replace only that tie-break with a seeded random number; all real priorities stay.
    _orig_priority = Voxel2Brick._greedy_priority

    def _jitter_priority(self, brick):
        return _orig_priority(self, brick)[:-3] + (self.rng.uniform(),)
    Voxel2Brick._greedy_priority = _jitter_priority

from common import parse, ROOT  # noqa: E402  (also adds 1x3 to brickgpt's library, needed to score B layouts)
if NEW and NEW[0] != '99':  # also add the part to brickgpt's library (used for scoring)
    import importlib  # noqa: E402
    _add_part(importlib.import_module('brickgpt.data.brick_library'), NEW)
from brickgpt.data import BrickStructure  # noqa: E402
from stable_metrics import interlocking_score, seam_coverage_score  # noqa: E402

if os.environ.get('TOL_FIX', '1') == '1':  # residual tolerance fix, see tolerant_stability.py
    import tolerant_stability  # noqa: E402
    tolerant_stability.patch_all()

SHAPE_SEED = 0
df = (pd.read_parquet(ROOT / 'data' / 'train0.parquet').drop_duplicates('object_id')
      .sample(frac=1, random_state=SHAPE_SEED).reset_index(drop=True))

done = set()
if OUT.exists():
    for line in OUT.open():
        d = json.loads(line)
        done.add((d['object_id'], d['seed']))


def score(bs: BrickStructure) -> dict:
    s = bs.stability_scores()
    return dict(n_bricks=len(bs), max_score=float(s.max()), mean_score=float(s.mean()),
                stable=bool(s.max() < 1 and not bs.has_floating_bricks()),
                connected=bool(bs.connectivity_scores().max() < 1),
                interlocking=float(interlocking_score(bs)), seam=float(seam_coverage_score(bs)))


with OUT.open('a') as f:
    START = int(os.environ.get("START", 0))
    # SHAPES=path/to/hard_shapes.json (uses its shape_idx list; N = how many of them, from position START)
    idx = (json.load(open(os.environ['SHAPES']))['shape_idx'][START:START + N] if os.environ.get('SHAPES')
           else range(START, START + N))
    for i in idx:
        r = df.iloc[i]
        orig = BrickStructure(parse(r['bricks']))
        voxels = (orig.voxel_occupancy > 0).astype(np.uint8)
        for seed in range(K):
            if (r['object_id'], seed) in done:
                continue
            t0 = time.time()
            m = voxel2brick(voxels, seed=seed)
            t_layout = time.time() - t0
            bs = BrickStructure(parse(m.to_txt()))
            assert ((bs.voxel_occupancy > 0) == (voxels > 0)).all(), 'layout changed the shape'
            d = dict(group=GROUP, tol_fix=os.environ.get('TOL_FIX', '1') == '1', jitter=os.environ.get('JITTER') == '1', shape_idx=i, object_id=r['object_id'], seed=seed, t_layout=round(t_layout, 2),
                     n_1x3=sum(1 for b in bs.bricks if sorted((b.h, b.w)) == [1, 3]),
                     n_new=sum(1 for b in bs.bricks if NEW and sorted((b.h, b.w)) == sorted(NEW[1:3])), **score(bs),
                     bricks=bs.to_txt())
            d['t_total'] = round(time.time() - t0, 2)
            f.write(json.dumps(d) + '\n'); f.flush()
            print(GROUP, i, seed, {k: d[k] for k in ['n_bricks', 'n_new', 'max_score', 'stable', 't_total']}, flush=True)
