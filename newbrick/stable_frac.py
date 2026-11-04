"""Exploratory: per shape, fraction of its 11 layouts (10 jitter + 1 deterministic) that are stable; paired Wilcoxon vs A."""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import wilcoxon
HERE = Path(__file__).parent
fr = defaultdict(lambda: defaultdict(list))
for f in glob.glob(str(HERE / 'stage1_out' / 'ab_*.jsonl')):
    for l in open(f):
        d = json.loads(l)
        fr[d['group']][d['shape_idx']].append(d['stable'])
A = {k: sum(v) / len(v) for k, v in fr['A'].items()}
for g in sys.argv[1:]:
    B = {k: sum(v) / len(v) for k, v in fr[g].items()}
    ks = sorted(A)
    a = [A[k] for k in ks]; b = [B[k] for k in ks]
    better = sum(y > x for x, y in zip(a, b)); worse = sum(y < x for x, y in zip(a, b))
    p = wilcoxon(a, b).pvalue
    print(f'{g:6s} stable fraction per shape: A mean {sum(a)/len(a):.3f} | {g} mean {sum(b)/len(b):.3f} | {g} better {better}, A better {worse}, tie {len(ks)-better-worse} | Wilcoxon p={p:.3g}')
