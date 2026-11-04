"""Compare BrickGPT generation with the original (buggy) vs tolerance-fixed stability check (bug_out/gen_on_*.jsonl).

Same caption + same seed per item. For every item finished under both conditions: identical output?, regenerations,
time, brick count, and the final structure scored with BOTH the original and the fixed stability_score.
Writes bug_out/gen_compare.json and prints a table.
"""
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import bug_impact as BI  # noqa: E402

recs = {}
for f in glob.glob(str(HERE / 'bug_out' / 'gen_on_*.jsonl')):
    for l in open(f):
        r = json.loads(l)
        recs[(r['cond'], r['item'])] = r
items = sorted({i for c, i in recs if ('on_buggy', i) in recs and ('on_fixed', i) in recs})

BI._init()
rows = []
for i in items:
    row = dict(item=i)
    for c in ('on_buggy', 'on_fixed'):
        r = recs[(c, i)]
        s = BI.score(dict(object_id=r['object_id'], bricks=r['bricks'])) if r['bricks'] else {}
        row[c] = dict(n_bricks=r['n_bricks'], regen=r['n_regenerations'], rej=r['n_rejections'], secs=r['secs'],
                      final_fixed_max=s.get('fixed_max'), final_orig_max=s.get('orig_max'),
                      floating=s.get('floating'), error=r['error'])
    row['identical'] = recs[('on_buggy', i)]['bricks'] == recs[('on_fixed', i)]['bricks']
    rows.append(row)
    b, f = row['on_buggy'], row['on_fixed']
    print(f"item {i:2d} | identical={row['identical']!s:5} | regen {b['regen']} vs {f['regen']} | bricks {b['n_bricks']} vs "
          f"{f['n_bricks']} | secs {b['secs']:.0f} vs {f['secs']:.0f} | final max (fixed scorer) "
          f"{b['final_fixed_max']:.3f} vs {f['final_fixed_max']:.3f} | (orig scorer) {b['final_orig_max']:.3f} vs "
          f"{f['final_orig_max']:.3f}", flush=True)


def tot(c, k):
    return sum(r[c][k] or 0 for r in rows)


summ = dict(n_items=len(rows), identical=sum(r['identical'] for r in rows),
            regen_total={c: tot(c, 'regen') for c in ('on_buggy', 'on_fixed')},
            secs_total={c: tot(c, 'secs') for c in ('on_buggy', 'on_fixed')},
            final_unstable_fixed_scorer={c: sum((r[c]['final_fixed_max'] or 0) >= 1 for r in rows)
                                         for c in ('on_buggy', 'on_fixed')},
            final_unstable_orig_scorer={c: sum((r[c]['final_orig_max'] or 0) >= 1 for r in rows)
                                        for c in ('on_buggy', 'on_fixed')},
            rows=rows)
json.dump(summ, open(HERE / 'bug_out' / 'gen_compare.json', 'w'), indent=1)
print({k: v for k, v in summ.items() if k != 'rows'})
