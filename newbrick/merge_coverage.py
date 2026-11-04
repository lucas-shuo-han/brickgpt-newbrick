"""How often could each candidate new brick appear in StableText2Brick, if we make it by merging two existing bricks?

For every structure: count pairs of bricks in the same layer whose union is exactly a candidate rectangle
(no overlap, no gap). Without new labels, that is the only way to get training data for a new brick.
Usage: python merge_coverage.py [n_structures]  -> prints a table, writes merge_coverage.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CANDS = ['1x3', '2x3', '1x10', '1x12', '1x16', '2x8', '2x10', '4x6']
n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
df = pd.read_parquet(HERE.parent / 'pilot' / 'data' / 'test.parquet').drop_duplicates('object_id').sample(n, random_state=7)


def cells(b):
    (h, w, x, y, z) = b
    return {(x + i, y + j) for i in range(h) for j in range(w)}


def parse(txt):
    out = []
    for line in txt.strip().split('\n'):
        dims, pos = line.split(' ')
        h, w = map(int, dims.split('x'))
        x, y, z = map(int, pos.strip('()').split(','))
        out.append((h, w, x, y, z))
    return out


tot = defaultdict(int)
has = defaultdict(int)
for txt in df['bricks']:
    bs = parse(txt)
    by_z = defaultdict(list)
    for b in bs:
        by_z[b[4]].append(b)
    found = defaultdict(int)
    for layer in by_z.values():
        cs = [cells(b) for b in layer]
        for i in range(len(layer)):
            for j in range(i + 1, len(layer)):
                if cs[i] & cs[j]:
                    continue
                u = cs[i] | cs[j]
                xs = [c[0] for c in u]; ys = [c[1] for c in u]
                h, w = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
                if h * w != len(u):
                    continue
                key = f'{min(h, w)}x{max(h, w)}'
                if key in CANDS:
                    found[key] += 1
    for k, v in found.items():
        tot[k] += v
        has[k] += 1

res = {k: dict(pairs_per_structure=tot[k] / n, frac_structures_with_one=has[k] / n) for k in CANDS}
json.dump(dict(n=n, sample='test.parquet drop_duplicates(object_id) sample(n, random_state=7)', result=res),
          open(HERE / 'merge_coverage.json', 'w'), indent=1)
for k in CANDS:
    print(f"{k:5s}  merge pairs / structure {res[k]['pairs_per_structure']:6.2f}   structures with >=1: {res[k]['frac_structures_with_one']:.0%}")
