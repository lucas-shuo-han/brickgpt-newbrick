"""Full generation: does BrickGPT actually place 1x3 bricks when told it may?

The 1x3 brick is in the library for ALL conditions (so it is never rejected); only the prompt changes.
Metric: 1x3 usage = n_1x3 / (n_1x3 + missed), where `missed` = 1x1+1x2 pairs that could be merged into a 1x3.
Usage: python gen.py <n_captions> <max_bricks>
"""
import json
import sys
import time

import pandas as pd

from common import (ORIG_DIMS, NEW_DIMS, DESCRIPTION, instruction, parse, merge_1x3, to_txt, ROOT, MERGED_DIR)
from brickgpt.models import BrickGPT, BrickGPTConfig

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MAX_BRICKS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
CONDS_ARG = sys.argv[3].split(',') if len(sys.argv) > 3 else None

df = pd.read_parquet(ROOT / 'data' / 'test.parquet').drop_duplicates('object_id').assign(caption=lambda d: d['captions'].str[0]).reset_index(drop=True)
rows = []
for i, r in df.iterrows():
    merged, idx = merge_1x3(parse(r['bricks']))
    rows.append((i, len(merged), len(idx), merged))
demo_rows = sorted([x for x in rows if x[2] >= 3], key=lambda x: x[1])[:3]
demos = [{'caption': df.loc[i, 'caption'], 'bricks': to_txt(m)} for i, _, _, m in demo_rows]
demo_ids = {x[0] for x in demo_rows}
caps = [df.loc[i, 'caption'] for i in range(len(df)) if i not in demo_ids][:N]

CONDS = {
    'orig_prompt': dict(dims=ORIG_DIMS),
    'zeroshot_desc': dict(dims=NEW_DIMS, extra=DESCRIPTION),
    'fewshot_k1': dict(dims=NEW_DIMS, extra=DESCRIPTION, demos=demos[:1]),
}
if CONDS_ARG:
    CONDS = {k: v for k, v in CONDS.items() if k in CONDS_ARG}

import os
MODEL = os.environ.get('MODEL', str(MERGED_DIR))
cfg = BrickGPTConfig(model_name_or_path=MODEL, use_gurobi=False, max_bricks=MAX_BRICKS,
                     max_regenerations=2, device='cpu')
bg = BrickGPT(cfg)

out = []
for cname, c in CONDS.items():
    bg.instruction_fn = lambda cap, c=c: instruction(cap, **c)
    for seed, cap in enumerate(caps):
        import torch
        torch.manual_seed(seed)
        t0 = time.time()
        res = bg(cap)
        bricks = res['bricks'].bricks
        n13 = sum(1 for b in bricks if (b.h, b.w) in [(1, 3), (3, 1)])
        _, missed = merge_1x3(bricks)
        d = dict(cond=cname, caption=cap, n_bricks=len(bricks), n_1x3=n13, missed=len(missed),
                 usage=n13 / (n13 + len(missed)) if n13 + len(missed) else None,
                 not_in_library=res['rejection_reasons'].get('not_in_library', 0),
                 secs=round(time.time() - t0), bricks=to_txt(bricks))
        out.append(d)
        print(cname, {k: v for k, v in d.items() if k != 'bricks'}, flush=True)
        with open(ROOT / os.environ.get('OUT', 'results_gen.json'), 'w') as f:
            json.dump(out, f, indent=1)
