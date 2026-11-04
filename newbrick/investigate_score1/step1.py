# Re-run the 3 score-1 structures from stage0 sample: compare to original, dump flagged bricks, repeat 3x.
import json, time, numpy as np, pandas as pd, sys
sys.path.insert(0, r'D:/HuaweiMoveData/Users/18547/Desktop/icl/newbrick/investigate_score1')
from diag import run, ROOT, BrickStructure, parse
df = pd.read_parquet(ROOT/'data'/'test.parquet').drop_duplicates('object_id').sample(10, random_state=0)
out = []
for idx in [2, 3, 4, 7]:
    r = df.iloc[idx]
    for rep in range(3):
        t = time.time()
        s, sc, d = run(r['bricks'])
        dt = time.time() - t
        flagged = [b for b in d['bricks'] if (b['fx']>0 or b['fy']>0 or b['fz']>0 or b['t1']>0 or b['t2']>0 or b['min_c']<=0)]
        rec = dict(sample_idx=idx, object_id=r['object_id'], rep=rep, n=len(s), t=dt, status=d['status'], obj=d['obj'],
                   eq_obj=d['eq_obj'], mipgap=d['mipgap'], objbound=d['objbound'], g_max=float(sc.max()), g_mean=float(sc.mean()),
                   n_flagged=len(flagged), flagged=flagged)
        out.append(rec)
        print({k: v for k, v in rec.items() if k != 'flagged'}, flush=True)
        for b in flagged[:10]:
            print('   ', {k: (f'{v:.3g}' if isinstance(v, float) else v) for k, v in b.items()})
json.dump(out, open('step1_out.json', 'w'), indent=1)
