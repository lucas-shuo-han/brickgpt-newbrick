"""How often does the Gurobi residual bug flip a stable dataset structure to 'unstable'?

Scores N StableText2Brick TEST structures with BrickGPT's ORIGINAL stability_score and with the tolerance-fixed one
(tolerant_stability.py, tol 1e-9), same process, same structure. Dataset structures are supposed to be stable.
Usage: python bug_impact.py <n> <procs>   -> bug_out/bug_impact.jsonl (resumable), prints a summary.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / 'pilot')); sys.path.insert(0, str(HERE))
OUT = HERE / 'bug_out' / 'bug_impact.jsonl'
_f = {}


def _init():
    import importlib
    from common import parse  # noqa: F401  (puts brickgpt on the path)
    mod = importlib.import_module('brickgpt.stability_analysis.stability_analysis')
    _f['orig'] = mod.stability_score
    import tolerant_stability
    tolerant_stability.patch_all()
    _f['fixed'] = mod.stability_score


def score(rec):
    import numpy as np
    from common import parse
    from brickgpt.data import BrickStructure
    from brickgpt.data.brick_library import brick_library
    from brickgpt.stability_analysis import StabilityConfig
    bs = BrickStructure(parse(rec['bricks']))
    out = dict(object_id=rec['object_id'], n_bricks=len(bs), floating=bool(bs.has_floating_bricks()))
    for k in ('orig', 'fixed'):
        t = time.time()
        s = _f[k](bs.to_json(), brick_library, StabilityConfig(world_dimension=(20, 20, 20)))[0]
        out[f'{k}_max'] = float(np.max(s)); out[f'{k}_mean'] = float(np.mean(s)); out[f'{k}_secs'] = round(time.time() - t, 2)
    return out


if __name__ == '__main__':
    import pandas as pd
    n, procs = int(sys.argv[1]), int(sys.argv[2])
    OUT.parent.mkdir(exist_ok=True)
    df = pd.read_parquet(HERE.parent / 'pilot' / 'data' / 'test.parquet').drop_duplicates('object_id').sample(n, random_state=7)
    done = {json.loads(l)['object_id'] for l in open(OUT)} if OUT.exists() else set()
    todo = [dict(object_id=r['object_id'], bricks=r['bricks']) for _, r in df.iterrows() if r['object_id'] not in done]
    with Pool(procs, initializer=_init) as pool, open(OUT, 'a') as f:
        for o in pool.imap_unordered(score, todo):
            f.write(json.dumps(o) + '\n'); f.flush()
    rs = [json.loads(l) for l in open(OUT)]
    a = sum(r['orig_max'] >= 1 for r in rs); b = sum(r['fixed_max'] >= 1 for r in rs)
    flipped = sum(r['orig_max'] >= 1 and r['fixed_max'] < 1 for r in rs)
    summ = dict(n=len(rs), sample='test.parquet, drop_duplicates(object_id), sample(n, random_state=7)',
                unstable_orig=a, unstable_fixed=b, flipped_by_bug=flipped, frac_flipped=flipped / len(rs),
                floating=sum(r['floating'] for r in rs))
    json.dump(summ, open(HERE / 'bug_out' / 'bug_impact_summary.json', 'w'), indent=1)
    print(summ)
