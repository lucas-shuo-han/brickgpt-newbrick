"""Score the Colab stage-0 generations (gen.jsonl from Drive) with the original and the tolerance-fixed Gurobi check.

Usage: python score_colab.py <gen.jsonl> [procs]  -> colab_out/stage0_scored.jsonl + colab_out/stage0_scored_summary.json
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import bug_impact as BI  # noqa: E402


def _score(r):
    if not r['bricks']:
        return dict(item=r['item'], cond=r['cond'], error=r.get('error') or 'empty')
    o = BI.score(dict(object_id=r['object_id'], bricks=r['bricks']))
    o.update(item=r['item'], cond=r['cond'])
    return o


if __name__ == '__main__':
    src, procs = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10
    recs = [json.loads(l) for l in open(src, encoding='utf-8') if l.strip()]
    out = HERE / 'colab_out' / 'stage0_scored.jsonl'
    with Pool(procs, initializer=BI._init) as pool:
        rows = sorted(pool.map(_score, recs), key=lambda r: (r['cond'], r['item']))
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    summ = {}
    for c in sorted({r['cond'] for r in rows}):
        rs = [r for r in rows if r['cond'] == c and 'orig_max' in r]
        summ[c] = dict(n=len(rs), errors=sum(r['cond'] == c and 'orig_max' not in r for r in rows),
                       stable_orig=sum(r['orig_max'] < 1 for r in rs), stable_fixed=sum(r['fixed_max'] < 1 for r in rs),
                       flipped_by_bug=sum(r['orig_max'] >= 1 and r['fixed_max'] < 1 for r in rs),
                       floating=sum(r['floating'] for r in rs),
                       stable_fixed_and_connected=sum(r['fixed_max'] < 1 and not r['floating'] for r in rs))
    json.dump(summ, open(HERE / 'colab_out' / 'stage0_scored_summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))
