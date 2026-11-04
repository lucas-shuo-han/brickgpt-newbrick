"""
Stage 2 data builder: turn stage-1 layouts (JSONL) into LoRA fine-tuning data in the SAME format
as BrickGPT's original fine-tuning data.

Original format (pilot/BrickGPT/src/brickgpt/prepare_finetuning_dataset.py):
  * one JSONL line per (structure, caption) pair -- every caption of a structure becomes its own example
    (convert_sample loops `for caption in captions`);
  * each line is {"messages": [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": create_instruction(caption)},
        {"role": "assistant", "content": bricks}]}     # bricks = "hxw (x,y,z)\n" lines
  * create_instruction is in pilot/BrickGPT/src/brickgpt/models/brickgpt.py; it is copied verbatim below,
    with "1x3, 3x1" inserted into the "Allowed brick dimensions" list (see PROMPT_1X3).

Groups written (N = number of SHAPES; each shape yields len(captions) lines, 5 in StableText2Brick):
  B_best : best-of-K B layout (lowest max_score among stable layouts, tie -> lowest mean_score);
           only shapes whose best B layout has n_1x3 >= 1.
  C1     : best-of-K A layout (original 8 bricks), same shapes as B_best_N.
  C2     : a B layout drawn uniformly at random among the K seeds (no score selection), same shapes.
Subsets are nested: B_best_N100 shapes are a prefix of B_best_N500 shapes, etc.

Usage:
  python build_stage2_data.py stage1_out/*.jsonl [--sizes 100,500,2000] [--out stage2_data]
"""
import argparse
import glob
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'pilot'

SYSTEM_PROMPT = 'You are a helpful assistant.'  # prepare_finetuning_dataset.py, create_messages

ORIG_DIMS = '2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2'
NEW_DIMS = '2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2, 1x3, 3x1'


def create_instruction(caption: str, dims: str) -> str:
    """Verbatim copy of brickgpt.models.brickgpt.create_instruction, with the dimension list parameterised."""
    return ('Create a LEGO model of the input. Format your response as a list of bricks: '
            '<brick dimensions> <brick position>, where the brick position is (x,y,z).\n'
            f'Allowed brick dimensions are {dims}.\n'
            'All bricks are 1 unit tall.\n\n'
            '### Input:\n'
            f'{caption}')


def create_messages(caption, bricks, dims):
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': create_instruction(caption, dims)},
        {'role': 'assistant', 'content': bricks},
    ]


def log(*a):
    print(*a, flush=True)


def load_layouts(paths):
    """Returns {group: {object_id: [records]}}; duplicate (group, object_id, seed) keeps the last line."""
    by_key = {}
    n_lines = 0
    for p in paths:
        with open(p, encoding='utf-8') as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    log(f'  WARNING: bad JSON in {p}:{ln}, skipped')
                    continue
                n_lines += 1
                by_key[(r['group'], r['object_id'], r['seed'])] = r
    out = defaultdict(lambda: defaultdict(list))
    for (g, oid, _), r in by_key.items():
        out[g][oid].append(r)
    for g in out:
        for oid in out[g]:
            out[g][oid].sort(key=lambda r: r['seed'])
    log(f'Read {n_lines} lines from {len(paths)} file(s); {len(by_key)} unique (group, object_id, seed); '
        f'groups: {{{", ".join(f"{g}: {len(v)} shapes" for g, v in sorted(out.items()))}}}')
    return out


def best_of_k(recs):
    stable = [r for r in recs if r.get('stable')]
    if not stable:
        return None
    return min(stable, key=lambda r: (r['max_score'], r['mean_score'], r['seed']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='stage-1 JSONL files (globs allowed)')
    ap.add_argument('--sizes', default='100,500,2000', help='comma-separated N (number of shapes)')
    ap.add_argument('--out', default=str(HERE / 'stage2_data'))
    ap.add_argument('--train_parquet', default=str(PILOT / 'data' / 'train0.parquet'))
    ap.add_argument('--test_parquet', default=str(PILOT / 'data' / 'test.parquet'))
    ap.add_argument('--holdout_frac', type=float, default=0.1, help='fraction of object_ids held out')
    ap.add_argument('--split_seed', type=int, default=1234)
    ap.add_argument('--subset_seed', type=int, default=0, help='order for nested subsets')
    ap.add_argument('--c2_seed', type=int, default=0, help='seed for the random layout choice of C2')
    ap.add_argument('--c1_prompt', choices=['new', 'orig'], default='new',
                    help='"new" (default, recommended): C1 prompt also lists 1x3/3x1, so prompts are identical '
                         'across groups; "orig": C1 uses the unchanged BrickGPT prompt')
    args = ap.parse_args()

    sizes = sorted({int(s) for s in args.sizes.split(',') if s.strip()})
    paths = sorted({p for pat in args.inputs for p in (glob.glob(pat) or [pat])})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    layouts = load_layouts(paths)
    A, B = layouts.get('A', {}), layouts.get('B', {})

    # ---- captions: same row stage1.py uses (train0.drop_duplicates('object_id') keeps the first row)
    train = pd.read_parquet(args.train_parquet).drop_duplicates('object_id')
    captions = {oid: [str(c) for c in caps] for oid, caps in zip(train['object_id'], train['captions'])}
    test_ids = set(pd.read_parquet(args.test_parquet)['object_id'])
    test_ids_norm = {t.lstrip('0') for t in test_ids}  # some ids in the dataset lost leading zeros

    all_ids = sorted(set(A) | set(B))
    no_caption = [o for o in all_ids if o not in captions]
    overlap_exact = [o for o in all_ids if o in test_ids]
    overlap_norm = [o for o in all_ids if o.lstrip('0') in test_ids_norm]
    log(f'Shapes in input: {len(all_ids)}; missing from train0 captions: {len(no_caption)}; '
        f'overlap with test.parquet object_ids: exact {len(overlap_exact)}, ignoring leading zeros {len(overlap_norm)}')
    excluded = set(no_caption) | set(overlap_norm)
    ids = [o for o in all_ids if o not in excluded]

    # ---- shape-level hold-out (on all usable shapes, independent of layout outcomes)
    rng = random.Random(args.split_seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_hold = math.ceil(args.holdout_frac * len(shuffled)) if len(shuffled) >= 2 else 0
    heldout = sorted(shuffled[:n_hold])
    heldout_set = set(heldout)
    train_ids = [o for o in ids if o not in heldout_set]
    (out_dir / 'heldout_object_ids.txt').write_text(''.join(o + '\n' for o in heldout), encoding='utf-8')
    log(f'Hold-out: {len(heldout)} shapes (frac {args.holdout_frac}, seed {args.split_seed}); '
        f'train pool: {len(train_ids)} shapes')
    assert not (heldout_set & set(train_ids))

    # ---- per-shape choices
    reasons = Counter()
    eligible = []  # shapes usable for all three groups
    chosen = {}
    c2_rng = random.Random(args.c2_seed)
    for oid in sorted(train_ids):
        b_recs, a_recs = B.get(oid, []), A.get(oid, [])
        if not b_recs:
            reasons['no B layouts'] += 1; continue
        if not a_recs:
            reasons['no A layouts'] += 1; continue
        bb, ab = best_of_k(b_recs), best_of_k(a_recs)
        if bb is None:
            reasons['no stable B layout'] += 1; continue
        if bb['n_1x3'] < 1:
            reasons['best B layout has no 1x3'] += 1; continue
        if ab is None:
            reasons['no stable A layout (C1 impossible)'] += 1; continue
        c2 = b_recs[c2_rng.randrange(len(b_recs))]
        chosen[oid] = {'B_best': bb, 'C1': ab, 'C2': c2}
        eligible.append(oid)
    log(f'Eligible train shapes (best B has >=1 1x3, and stable A exists): {len(eligible)}')
    for k, v in sorted(reasons.items()):
        log(f'  excluded: {k}: {v}')

    # ---- nested subsets
    order = eligible[:]
    random.Random(args.subset_seed).shuffle(order)
    dims_for = {'B_best': NEW_DIMS, 'C2': NEW_DIMS, 'C1': NEW_DIMS if args.c1_prompt == 'new' else ORIG_DIMS}

    def write(path, oids, group):
        n_lines, n13, nb = 0, [], []
        with open(path, 'w', encoding='utf-8') as f:
            for oid in oids:
                r = chosen[oid][group]
                for cap in captions[oid]:
                    f.write(json.dumps({'messages': create_messages(cap, r['bricks'], dims_for[group])},
                                       ensure_ascii=False) + '\n')
                    n_lines += 1
                n13.append(r['n_1x3']); nb.append(r['n_bricks'])
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None
        return {'file': path.name, 'n_shapes': len(oids), 'n_examples': n_lines,
                'mean_n_1x3_per_shape': mean(n13), 'mean_n_bricks_per_shape': mean(nb),
                'frac_shapes_with_1x3': mean([x > 0 for x in n13]),
                'mean_max_score': mean([chosen[o][group]['max_score'] for o in oids]),
                'frac_stable': mean([bool(chosen[o][group]['stable']) for o in oids])}

    files = {}
    for n in sizes:
        sub = order[:n]
        if len(sub) < n:
            log(f'*** N={n}: only {len(sub)} eligible shapes available -- writing {len(sub)} shapes')
        for group in ('B_best', 'C1', 'C2'):
            info = write(out_dir / f'{group}_N{n}.jsonl', sub, group)
            info['requested_N'] = n
            if group == 'C2':
                info['n_same_seed_as_B_best'] = sum(chosen[o]['C2']['seed'] == chosen[o]['B_best']['seed'] for o in sub)
                info['n_identical_bricks_to_B_best'] = sum(
                    chosen[o]['C2']['bricks'] == chosen[o]['B_best']['bricks'] for o in sub)
            files[f'{group}_N{n}'] = info
            log(f'  {group}_N{n}: {info["n_shapes"]} shapes, {info["n_examples"]} examples, '
                f'mean n_1x3/shape {info["mean_n_1x3_per_shape"]}')

    # eval file in the same format for the held-out shapes (best B if it exists; no 1x3 filter)
    held_recs = {o: best_of_k(B.get(o, [])) for o in heldout}
    with open(out_dir / 'heldout_B_best.jsonl', 'w', encoding='utf-8') as f:
        n_held_lines = 0
        for o in heldout:
            if held_recs[o] is None:
                continue
            for cap in captions[o]:
                f.write(json.dumps({'messages': create_messages(cap, held_recs[o]['bricks'], NEW_DIMS)},
                                   ensure_ascii=False) + '\n')
                n_held_lines += 1

    manifest = {
        'inputs': paths,
        'format': 'one {"messages":[system,user,assistant]} per (shape, caption), as prepare_finetuning_dataset.py',
        'system_prompt': SYSTEM_PROMPT,
        'allowed_dims': {g: d for g, d in dims_for.items()},
        'selection': 'best-of-K: min (max_score, mean_score, seed) among stable layouts; '
                     'C2: uniform random among all K B layouts',
        'seeds': {'split_seed': args.split_seed, 'subset_seed': args.subset_seed, 'c2_seed': args.c2_seed},
        'holdout_frac': args.holdout_frac,
        'counts': {
            'input_shapes': len(all_ids),
            'K_per_shape': {g: dict(Counter(len(v) for v in layouts[g].values())) for g in layouts},
            'missing_caption': len(no_caption),
            'test_overlap_exact': len(overlap_exact),
            'test_overlap_ignoring_leading_zeros': len(overlap_norm),
            'heldout_shapes': len(heldout),
            'heldout_examples': n_held_lines,
            'train_pool_shapes': len(train_ids),
            'eligible_shapes': len(eligible),
            'excluded': dict(reasons),
        },
        'files': files,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    log(f'Wrote {out_dir / "manifest.json"}')


if __name__ == '__main__':
    sys.exit(main())
