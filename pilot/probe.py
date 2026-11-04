"""Log-prob probes: does BrickGPT prefer a 1x3 brick where one fits exactly?

Probe A (synthetic gap): a 1-wide wall with a gap of length g between two 1x4 bricks.
  Candidates for the next brick at the gap start: 1x1, 1x2, 1x3, 1x4 (only those that fit are kept).
  Only g=3 is filled exactly by one brick, and only by the new 1x3.
Probe B (real data): StableText2Brick test structures with 1x1+1x2 merged into 1x3.
  At each merged spot, compare P(1x3 brick) with P(the original first brick of the pair).

Conditions differ only in the prompt (model weights are frozen).
"""
import json
import random
import sys
import time

import numpy as np
import pandas as pd

from common import (Brick, ORIG_DIMS, NEW_DIMS, DESCRIPTION, instruction, chat_prefix_ids,
                    score_candidates, load_model, parse, merge_1x3, to_txt, ROOT)

random.seed(0)
N_REAL = int(sys.argv[1]) if len(sys.argv) > 1 else 30

tok, model = load_model()

df = pd.read_parquet(ROOT / 'data' / 'test.parquet')
df = df.drop_duplicates('object_id').reset_index(drop=True)
df['caption'] = df['captions'].str[0]

# ---- pick few-shot demos: small structures with many merges ----
rows = []
for i, r in df.iterrows():
    bricks = parse(r['bricks'])
    merged, idx = merge_1x3(bricks)
    rows.append((i, len(merged), len(idx), merged, idx))
cands = sorted([x for x in rows if x[2] >= 3], key=lambda x: x[1])
demo_rows = cands[:3]
demos = [{'caption': df.loc[i, 'caption'], 'bricks': to_txt(m)} for i, _, _, m, _ in demo_rows]
demo_ids = {i for i, *_ in demo_rows}
print('demo sizes (bricks, #1x3):', [(n, k) for _, n, k, _, _ in demo_rows], flush=True)

CONDS = {
    'orig_prompt': dict(dims=ORIG_DIMS),
    'zeroshot_list': dict(dims=NEW_DIMS),
    'zeroshot_desc': dict(dims=NEW_DIMS, extra=DESCRIPTION),
    'fewshot_k1': dict(dims=NEW_DIMS, extra=DESCRIPTION, demos=demos[:1]),
    'fewshot_k3': dict(dims=NEW_DIMS, extra=DESCRIPTION, demos=demos[:3]),
}

import os
if os.environ.get('CONDS'):
    CONDS = {k: v for k, v in CONDS.items() if k in os.environ['CONDS'].split(',')}
OUT = os.environ.get('OUT', 'results_probe.json')
results = []

# ---------------- Probe A ----------------
captions_a = ['A long, low straight wall.', 'A simple fence made of a single row of bricks.',
              'A thin rectangular border.', 'A narrow bench with a flat top.']
for cname, cfg in CONDS.items():
    t0 = time.time()
    for cap in captions_a:
        for g in [1, 2, 3, 4, 5]:
            for x0 in [4, 10]:
                y0 = 2
                ctx = [Brick(h=1, w=4, x=x0, y=y0, z=0), Brick(h=1, w=4, x=x0, y=y0 + 4 + g, z=0)]
                gy = y0 + 4
                cand_dims = [(1, w) for w in (1, 2, 3, 4) if w <= g]
                cand = [f'1x{w} ({x0},{gy},0)\n' for _, w in cand_dims]
                ids = chat_prefix_ids(tok, instruction(cap, **cfg), to_txt(ctx))
                lp = np.array(score_candidates(tok, model, ids, cand))
                p = np.exp(lp - lp.max()); p /= p.sum()
                d = dict(probe='A', cond=cname, gap=g, caption=cap,
                         p_1x3=float(p[2]) if g >= 3 else None,
                         p_exact=float(p[[w for _, w in cand_dims].index(g)]) if g <= 4 else None,
                         top=cand_dims[int(lp.argmax())][1],
                         abs_logp=dict(zip([w for _, w in cand_dims], lp.round(2).tolist())))
                results.append(d)
    print(f'A {cname} done in {time.time()-t0:.0f}s', flush=True)

# ---------------- Probe B ----------------
pool = [x for x in rows if x[0] not in demo_ids and x[2] >= 1]
random.shuffle(pool)
pool = pool[:N_REAL]
for cname, cfg in CONDS.items():
    t0 = time.time()
    for i, n, k, merged, idx in pool:
        j = random.choice(idx)  # one merged spot per structure
        m = merged[j]
        orig_bricks = parse(df.loc[i, 'bricks'])
        # original brick that the 1x3 replaced and that came first: the one at the 1x3's position in build order
        cells = {(x, y, m.z) for x in range(m.x, m.x + m.h) for y in range(m.y, m.y + m.w)}
        prefix_set = set(merged[:j])
        alts = [b for b in orig_bricks if b not in prefix_set and
                {(x, y, b.z) for x in range(b.x, b.x + b.h) for y in range(b.y, b.y + b.w)} <= cells]
        alt = min(alts, key=lambda b: orig_bricks.index(b))
        cand = [m.to_txt(), alt.to_txt()]
        for prefix_kind, prefix in [('merged_prefix', merged[:j])]:
            ids = chat_prefix_ids(tok, instruction(df.loc[i, 'caption'], **cfg), to_txt(prefix))
            if ids.shape[1] > 6000:
                continue
            lp = np.array(score_candidates(tok, model, ids, cand))
            p = np.exp(lp - lp.max()); p /= p.sum()
            results.append(dict(probe='B', cond=cname, prefix=prefix_kind, obj=int(i), n_prefix=j,
                                n_1x3_in_prefix=sum(1 for b in prefix if (b.h, b.w) in [(1, 3), (3, 1)]),
                                p_1x3=float(p[0]), logp_1x3=float(lp[0]), logp_alt=float(lp[1]),
                                alt=alt.to_txt().strip(), new=m.to_txt().strip()))
    print(f'B {cname} done in {time.time()-t0:.0f}s', flush=True)

with open(ROOT / OUT, 'w') as f:
    json.dump(results, f, indent=1)
print('saved', len(results))
