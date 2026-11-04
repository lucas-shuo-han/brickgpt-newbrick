"""Does the Gurobi residual bug change BrickGPT's own inference (physics-informed rollback)?

Same captions, same seeds, rollback ON (max_regenerations=100, Gurobi), CPU, released BrickGPT:
  cond 'on_buggy' : original stability_score (strict `.X > 0`)
  cond 'on_fixed' : tolerance-fixed (tolerant_stability, 1e-9)
With identical seeds both start from the same first structure; they diverge only where the stability verdicts differ.
Items = the same random test subset as the Colab notebook (drop_duplicates(object_id), RandomState(0).choice(n=100)),
caption index 0, seed 1000 + item -> directly comparable with the Colab rollback-OFF run.

Usage: python gen_bug.py <cond> <item_start> <item_end> <out.jsonl> [threads] [deadline_unix]
Stops starting new items after deadline_unix. Resumable (skips finished items).
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
cond, a, b, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
threads = int(sys.argv[5]) if len(sys.argv) > 5 else 2
deadline = float(sys.argv[6]) if len(sys.argv) > 6 else float('inf')
sys.path[:0] = [str(HERE.parent / 'pilot' / 'BrickGPT' / 'src'), str(HERE / 'colab')]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(threads)
import helpers as H  # noqa: E402

H.TOL_FIX = (cond == 'on_fixed')
if H.TOL_FIX:
    H.patch_stability_tolerance()

df = pd.read_parquet(HERE.parent / 'pilot' / 'data' / 'test.parquet').drop_duplicates('object_id').reset_index(drop=True)
idx = sorted(np.random.RandomState(0).choice(len(df), size=100, replace=False).tolist())
items = [dict(item=k, object_id=str(df.loc[i, 'object_id']), caption=str(df.loc[i, 'captions'][0]))
         for k, i in enumerate(idx)][a:b]

MODEL = str(HERE.parent / 'pilot' / 'models' / 'brickgpt-merged')
cfg = dict(model_name_or_path=MODEL, max_regenerations=100, use_gurobi=True, device='cpu')
bg = None
for it in items:
    if time.time() > deadline:
        print('deadline reached, stopping', flush=True)
        break
    bg = H.run_generation(cfg, [it], out, cond, seed_base=1000, bg=bg, extra_fields=dict(tol_fix=H.TOL_FIX))
