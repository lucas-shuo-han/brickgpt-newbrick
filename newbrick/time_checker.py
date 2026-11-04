import sys, time, json, numpy as np, pandas as pd
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pilot'))
from common import parse, ROOT
from brickgpt.data import BrickStructure
df = pd.read_parquet(ROOT/'data'/'test.parquet').drop_duplicates('object_id').sample(10, random_state=0)
out=[]
for _, r in df.iterrows():
    s = BrickStructure(parse(r['bricks']))
    t=time.time(); 
    try: g = s.stability_scores(); tg=time.time()-t; gerr=None
    except Exception as e: tg=None; gerr=str(e)[:200]
    t=time.time(); c = s.connectivity_scores(); tc=time.time()-t
    out.append(dict(n=len(s), t_gurobi=tg, gurobi_err=gerr, t_conn=tc,
        g_max=None if gerr else float(np.max(g)), g_mean=None if gerr else float(np.mean(g)), conn_max=float(np.max(c))))
    print(out[-1], flush=True)
json.dump(out, open(r'D:/HuaweiMoveData/Users/18547/Desktop/icl/newbrick/stage0_checker_timing.json','w'), indent=1)
