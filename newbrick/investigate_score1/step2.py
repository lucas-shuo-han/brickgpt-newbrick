# Extra sample of test structures: original score vs. score with a numerical tolerance on the residual check.
import json, time, numpy as np, pandas as pd, sys
sys.path.insert(0, r'D:/HuaweiMoveData/Users/18547/Desktop/icl/newbrick/investigate_score1')
from diag import run, ROOT
import gurobipy as gp
TOLS = [0.0, 1e-12, 1e-9, 1e-6]
def summarize(d, tol):
    worst = 0.0; nflag = 0
    for b in d['bricks']:
        res = max(b['fx'], b['fy'], b['fz'], b['t1'], b['t2'])
        if res > tol or b['min_c'] <= 0:
            s = 1.0; nflag += 1
        else:
            s = 1 - b['min_c'] / b['T']
        worst = max(worst, s)
    return worst, nflag
def max_res(d): return max(max(b['fx'], b['fy'], b['fz'], b['t1'], b['t2']) for b in d['bricks'])
base = pd.read_parquet(ROOT/'data'/'test.parquet').drop_duplicates('object_id')
seen = set(base.sample(10, random_state=0)['object_id'])
cand = base[~base['object_id'].isin(seen)].sample(60, random_state=1)
out = []; budget = 480; t0 = time.time()
mode = sys.argv[1] if len(sys.argv) > 1 else 'extra'
if mode == 'orig10':
    cand = base.sample(10, random_state=0); budget = 400
for _, r in cand.iterrows():
    if time.time() - t0 > budget or len(out) >= 30: break
    t = time.time(); s, sc, d = run(r['bricks']); dt = time.time() - t
    rec = dict(object_id=r['object_id'], n=len(s), t=dt, status=d['status'], eq_obj=d['eq_obj'], max_residual=max_res(d),
               g_max_orig=float(sc.max()))
    for tol in TOLS:
        rec[f'max_tol{tol:g}'], rec[f'nflag_tol{tol:g}'] = summarize(d, tol)
    out.append(rec); print(rec, flush=True)
json.dump(dict(gurobi_version=gp.gurobi.version(), records=out), open(f'step2_{mode}.json', 'w'), indent=1)
print('total', time.time() - t0)
