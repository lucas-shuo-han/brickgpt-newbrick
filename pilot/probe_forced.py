"""Forced probes: tasks where ONLY a 1x3 brick gives a valid (non-floating) solution.

The prompt states a target: cover exactly a row of cells at layer z=1, no floating bricks.
The prefix already holds the support bricks at z=0. We score the first top-layer brick at the row start:
  "1xw (x0,y0,1)" for w = 1..4.
Variants (row runs along y, starting at y0):
  need_end   : L=3, one 1x1 support under y0         -> only 1x3 works (1x1/1x2 leave floating cells)
  need_mid   : L=3, one 1x1 support under y0+1       -> only 1x3 works
  ctrl_full  : L=3, three 1x1 supports under the row  -> any split works (1x3 not needed)
  ctrl_end2  : L=2, support under y0                 -> only 1x2 works (existing brick)
  ctrl_end4  : L=4, support under y0                 -> only 1x4 works (existing brick)
ctrl_end2/4 check whether the model understands the task at all with OLD bricks.
Env: LORA=<adapter dir>, CONDS=a,b, OUT=file.json
"""
import json
import os
import time

import numpy as np

from common import (Brick, ORIG_DIMS, NEW_DIMS, DESCRIPTION, instruction, chat_prefix_ids,
                    score_candidates, load_model, to_txt, ROOT)

VARIANTS = {
    'need_end': (3, [0]), 'need_mid': (3, [1]), 'ctrl_full': (3, [0, 1, 2]),
    'ctrl_end2': (2, [0]), 'ctrl_end4': (4, [0]),
}
CORRECT = {'need_end': 3, 'need_mid': 3, 'ctrl_end2': 2, 'ctrl_end4': 4, 'ctrl_full': None}


def task_text(x0, y0, L):
    cells = ', '.join(f'({x0},{y0 + k})' for k in range(L))
    return (f'Task: the bricks already placed are given. Add bricks on layer z=1 that cover exactly the cells '
            f'(x,y) = {cells}, and nothing else. Every brick must rest on a brick below it: no floating bricks. '
            f'Use as few bricks as possible.\n')


def setup(variant, x0, y0):
    L, sup = VARIANTS[variant]
    prefix = [Brick(h=1, w=1, x=x0, y=y0 + s, z=0) for s in sup]
    return L, prefix


def solution(variant, x0, y0):
    L, _ = VARIANTS[variant]
    return [Brick(h=1, w=L, x=x0, y=y0, z=1)]


# few-shot demos: solved tasks at other positions, two with 1x3, one with 1x2, one with 1x4
DEMO_SPECS = [('need_end', 1, 14), ('ctrl_end2', 17, 5), ('need_mid', 7, 15), ('ctrl_end4', 12, 1)]


def demos_text(dims):
    out = '\nHere are some solved examples:\n\n'
    for v, x, y in DEMO_SPECS:
        L, prefix = setup(v, x, y)
        out += (f'### Input:\nA small ledge.\n{task_text(x, y, L)}### Output:\n'
                f'{to_txt(prefix)}{to_txt(solution(v, x, y))}\n')
    return out


CONDS = {
    'orig_prompt': dict(dims=ORIG_DIMS, extra=''),
    'list_desc': dict(dims=NEW_DIMS, extra=DESCRIPTION),
    'fewshot4': dict(dims=NEW_DIMS, extra=DESCRIPTION, demos=True),
}
if os.environ.get('CONDS'):
    CONDS = {k: v for k, v in CONDS.items() if k in os.environ['CONDS'].split(',')}
OUT = os.environ.get('OUT', 'results_forced.json')

tok, model = load_model()
captions = ['A small ledge.', 'A short shelf sticking out from a post.']
positions = [(3, 2), (9, 8), (15, 4), (5, 12), (11, 15), (16, 10)]
res = []
for cname, c in CONDS.items():
    t0 = time.time()
    for v in VARIANTS:
        for cap in captions:
            for x0, y0 in positions:
                L, prefix = setup(v, x0, y0)
                extra = c['extra'] + task_text(x0, y0, L)
                if c.get('demos'):
                    extra = c['extra'] + demos_text(c['dims']) + '\n' + task_text(x0, y0, L)
                ids = chat_prefix_ids(tok, instruction(cap, c['dims'], extra), to_txt(prefix))
                cand = [f'1x{w} ({x0},{y0},1)\n' for w in (1, 2, 3, 4)]
                lp = np.array(score_candidates(tok, model, ids, cand))
                p = np.exp(lp - lp.max()); p /= p.sum()
                res.append(dict(cond=cname, variant=v, caption=cap, x0=x0, y0=y0,
                                p=dict(zip([1, 2, 3, 4], p.round(4).tolist())),
                                top=int(np.argmax(lp)) + 1, correct=CORRECT[v]))
    print(cname, f'{time.time()-t0:.0f}s', flush=True)
json.dump(res, open(ROOT / OUT, 'w'), indent=1)
print('saved', len(res))
