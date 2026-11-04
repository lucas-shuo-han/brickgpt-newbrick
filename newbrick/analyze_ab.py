"""Stage 1 analysis: A (8 bricks) vs B (+1x3) on the 40 hard shapes (stage1_out/ab_*.jsonl, tolerance fix ON).

Per shape and group, candidates = 10 jittered layouts + 1 deterministic layout. Best-of-K = lowest max_score among
layouts without floating bricks (ties -> lower mean_score). Paired comparison A vs B per shape.
Independent check (no Gurobi): perturbation test (perturb_metric.py) on the best layouts.
Writes stage1_out/ab_summary.json and prints a table.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / 'pilot')); sys.path.insert(0, str(HERE))
from common import parse  # noqa: E402
from perturb_metric import perturbation_score  # noqa: E402

G = sys.argv[1] if len(sys.argv) > 1 else 'B'  # group to compare with A: B (1x3), B2x3, B1x10, B1x12
recs = [json.loads(l) for f in sorted(glob.glob(str(HERE / 'stage1_out' / 'ab_*.jsonl'))) for l in open(f)]
assert all(r.get('tol_fix') for r in recs)
shapes = sorted({r['shape_idx'] for r in recs})


def best(rs):
    return min(rs, key=lambda r: (r['max_score'], r['mean_score']))


rows = []
for s in shapes:
    row = dict(shape_idx=s)
    for g in ('A', G):
        rs = [r for r in recs if r['shape_idx'] == s and r['group'] == g]
        b = best(rs)
        p = perturbation_score(parse(b['bricks']), seed=0)
        row[g] = dict(n_layouts=len(rs), n_stable=sum(r['stable'] for r in rs), best_max=b['max_score'],
                      best_mean=b['mean_score'], best_stable=b['stable'], best_n_bricks=b['n_bricks'],
                      best_n_1x3=b.get('n_new', b['n_1x3']), best_interlocking=b['interlocking'], best_seam=b['seam'],
                      best_is_det=not b['jitter'], perturb_single_min=p['single_brick_min'],
                      perturb_pair_min=p['pair_brick_min'], perturb_critical=p['critical_frac'],
                      det_max=next((r['max_score'] for r in rs if not r['jitter']), None))
    rows.append(row)


def col(g, k):
    return np.array([r[g][k] for r in rows], dtype=float)


out = dict(n_shapes=len(rows), n_layouts={g: int(sum(r[g]['n_layouts'] for r in rows)) for g in ('A', G)})
lines = []
for k, better in [('best_max', 'lower'), ('best_mean', 'lower'), ('best_n_bricks', 'lower'),
                  ('best_interlocking', 'higher'), ('best_seam', 'higher'), ('perturb_single_min', 'higher'),
                  ('perturb_pair_min', 'higher'), ('perturb_critical', 'lower')]:
    a, b = col('A', k), col(G, k)
    d = b - a
    nz = d[d != 0]
    w = stats.wilcoxon(nz) if len(nz) >= 5 else None
    bw = int((d < 0).sum() if better == 'lower' else (d > 0).sum())
    aw = int((d > 0).sum() if better == 'lower' else (d < 0).sum())
    out[k] = dict(better=better, A_median=float(np.median(a)), B_median=float(np.median(b)),
                  A_mean=float(a.mean()), B_mean=float(b.mean()), B_better=bw, A_better=aw, ties=int((d == 0).sum()),
                  wilcoxon_p=float(w.pvalue) if w else None)
    lines.append(f'{G} vs A | {k:20s} A med {np.median(a):.4f} | B med {np.median(b):.4f} | B better {bw:2d}, A better {aw:2d}, '
                 f'tie {(d == 0).sum():2d} | Wilcoxon p={w.pvalue if w else float("nan"):.3g}')

out['best_stable'] = {g: int(col(g, 'best_stable').sum()) for g in ('A', G)}
out['all_layouts_stable_rate'] = {g: float(sum(r[g]['n_stable'] for r in rows) / sum(r[g]['n_layouts'] for r in rows))
                                  for g in ('A', G)}
out['det_unstable'] = {g: int((col(g, 'det_max') >= 1).sum()) for g in ('A', G)}
out['best_is_det'] = {g: int(col(g, 'best_is_det').sum()) for g in ('A', G)}
out['B_best_has_1x3'] = int((col(G, 'best_n_1x3') > 0).sum())
out['B_best_mean_1x3'] = float(col(G, 'best_n_1x3').mean())
out['only_B_stable'] = [r['shape_idx'] for r in rows if r[G]['best_stable'] and not r['A']['best_stable']]
out['only_A_stable'] = [r['shape_idx'] for r in rows if r['A']['best_stable'] and not r[G]['best_stable']]
out['rows'] = rows
json.dump(out, open(HERE / 'stage1_out' / f'ab_summary_{G}.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('rows',) and not k.startswith('best_') and
                  not k.startswith('perturb')}, indent=0))
print(f"best layout stable: A {out['best_stable']['A']}/40, {G} {out['best_stable'][G]}/40")
print('\n'.join(lines))
