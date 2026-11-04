"""Re-score the stage-1 screen layouts with the tolerance-fixed Gurobi score, then pick the hardest shapes.

The screen (stage1_out/screen_A_*.jsonl) ran before the fix (tol_fix field missing), so its max_score has false 1.0s.
Usage: python rescore_screen.py <n_hard> [n_procs]
Writes stage1_out/screen_A_rescored.jsonl and stage1_out/hard_shapes.json (shape_idx list, hardest first).
"""
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / 'pilot')); sys.path.insert(0, str(HERE))


def score(rec):
    from common import parse
    from brickgpt.data import BrickStructure
    import tolerant_stability
    tolerant_stability.patch_all()
    from perturb_metric import perturbation_score
    bricks = parse(rec['bricks'])
    s = BrickStructure(bricks).stability_scores()
    p = perturbation_score(bricks, seed=0)
    return dict(shape_idx=rec['shape_idx'], object_id=rec['object_id'], n_bricks=rec['n_bricks'],
                max_score_orig=rec['max_score'], max_score=float(s.max()), mean_score=float(s.mean()),
                interlocking=rec['interlocking'], seam=rec['seam'], perturb=p)


if __name__ == '__main__':
    n_hard = int(sys.argv[1]); procs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    recs = [json.loads(l) for f in sorted(glob.glob(str(HERE / 'stage1_out' / 'screen_A_[0-9]*.jsonl'))) for l in open(f)]
    with Pool(procs) as pool:
        out = pool.map(score, recs)
    with open(HERE / 'stage1_out' / 'screen_A_rescored.jsonl', 'w') as f:
        for o in out:
            f.write(json.dumps(o) + '\n')
    out.sort(key=lambda o: -o['max_score'])
    hard = [o['shape_idx'] for o in out[:n_hard]]
    json.dump(dict(rule=f'top {n_hard} by tolerance-fixed max_score of the deterministic group-A layout (seed 0)',
                   shape_idx=hard, max_score=[o['max_score'] for o in out[:n_hard]]),
              open(HERE / 'stage1_out' / 'hard_shapes.json', 'w'), indent=1)
    import numpy as np
    m = np.array([o['max_score'] for o in out]); mo = np.array([o['max_score_orig'] for o in out])
    print(f'{len(out)} layouts | orig: score==1 {int((mo >= 1).sum())} | fixed: >=1 {int((m >= 1).sum())}, '
          f'>0.1 {int((m > 0.1).sum())}, >0.3 {int((m > 0.3).sum())}, median {np.median(m):.4f}')
    print('hardest', [(o['shape_idx'], round(o['max_score'], 3), o['n_bricks']) for o in out[:n_hard]])
