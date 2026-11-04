"""Builds brickgpt_newbrick.ipynb from the cell sources below + helpers.py (embedded via %%writefile).
Run: python build_notebook.py   (re-run after editing helpers.py so the notebook copy stays identical)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
HELPERS = (HERE / 'helpers.py').read_text(encoding='utf-8')
cells = []


def md(s):
    cells.append(('markdown', s.strip('\n')))


def code(s):
    cells.append(('code', s.strip('\n')))


md(r'''
# BrickGPT + new 1x3 brick — Colab pipeline (Stage 0 baseline, Stage 3 LoRA, Stage 4 evaluation)

EECS 182 course project. Model: **BrickGPT** = Llama-3.2-1B-Instruct + official LoRA `AvaLovelace/BrickGPT`.
The base model is gated, so we load the ungated mirror `unsloth/Llama-3.2-1B-Instruct` (same weights) and merge the
official adapter on top, exactly like our local `pilot/common.py`.

**Runtime:** GPU (T4 or A100). `Runtime → Change runtime type → GPU`.

| Stage | What | Output (on Drive) |
|---|---|---|
| 0 | Baseline generation on a fixed random subset of test captions, rollback ON vs OFF, original brick library | `stage0/gen.jsonl`, `stage0/scored.jsonl`, `stage0/metrics.json` |
| 3 | LoRA from the released BrickGPT checkpoint, one run per `stage2_data/{group}_N{n}.jsonl`, same hyper-parameters | `stage3/<run>/adapter + train_log.json` |
| 4 | Stage-0 protocol again with each trained adapter and the 1x3 brick in the library | `stage4/<run>/...`, `stage4/summary.json` |

Everything is parameterised in the next cell. Every generated structure is appended to a JSONL immediately, and all
loops **resume** (already-finished items/runs are skipped), so a Colab disconnect only loses the current item.

**Helper code:** the notebook writes `helpers.py` itself (cell "Write helpers.py"); it is an identical copy of
`newbrick/colab/helpers.py` in our repo, so nothing has to be uploaded except the Gurobi licence and the Stage-2 data.

**Runtime estimates.** GPU timings: *unknown* (nothing has been run on a GPU). What we measured locally on CPU:
full generation 2–6 min per object (pilot `gen.py`: 60–126 bricks, `max_regenerations=2`, connectivity check);
one Gurobi stability check 1–53 s per structure (median 3.5 s, 10 test structures, 32–239 bricks); tiny LoRA
250 examples × 1 epoch ≈ 77 min on CPU. The Gurobi check runs on the CPU in Colab too, so it does not get faster
with a GPU. Use the **timing cell** in Stage 0 (a few items) to extrapolate before launching the full run.
''')

code(r'''
# ============================== PARAMETERS ==============================
# --- storage
USE_DRIVE = True                      # save everything to Google Drive (recommended: Colab sessions die)
DRIVE_DIR = '/content/drive/MyDrive/brickgpt_newbrick'   # output root on Drive
LOCAL_DIR = '/content/brickgpt_newbrick'                 # used if USE_DRIVE=False
MODELS_DIR = '/content/models'        # merged full models (large, ~5 GB each in fp32) -> local disk, not Drive

# --- code / models
BRICKGPT_REPO = 'https://github.com/AvaLovelace1/BrickGPT.git'
BRICKGPT_COMMIT = '9c53c7cb65f9de0df89562cc74b2d2e270fbcab4'  # commit of our local clone (pilot/BrickGPT)
BASE_MODEL = 'unsloth/Llama-3.2-1B-Instruct'   # ungated mirror of meta-llama/Llama-3.2-1B-Instruct
BRICKGPT_ADAPTER = 'AvaLovelace/BrickGPT'
DATASET = 'AvaLovelace/StableText2Brick'

# --- Gurobi
USE_GUROBI = True                     # False -> rollback uses the connectivity check; stability scored later locally
GUROBI_LIC_DRIVE_PATH = '/content/drive/MyDrive/gurobi/gurobi.lic'  # if present it is used, else upload prompt

# --- Stage 0 / Stage 4 generation
N_CAPTIONS = 100                      # size of the fixed random test subset
SUBSET_SEED = 0                       # seed for choosing the subset (recorded in subset.json)
GEN_SEED_BASE = 1000                  # item k is generated with transformers.set_seed(GEN_SEED_BASE + k)
CAPTION_INDEX = 0                     # which of the captions of an object to use (pilot/gen.py used 0)
MAX_BRICKS = 2000                     # BrickGPTConfig default
MAX_REGENERATIONS_ON = 100            # BrickGPTConfig default (rollback ON)
MAX_BRICK_REJECTIONS = 500            # BrickGPTConfig default (per-brick rejection sampling; kept ON in both conds)
ALSO_DISABLE_REJECTION_IN_OFF = False # True -> rollback-OFF also sets max_brick_rejections=0 (no validity filtering)
CONDITIONS = ['rollback_on', 'rollback_off']
TIMING_N = 3                          # items for the timing pilot cell

# --- Stage 3 LoRA
STAGE2_DIR = DRIVE_DIR + '/stage2_data'   # upload newbrick/stage2_data/ here
TRAIN_GROUPS = ['B_best', 'C1', 'C2']     # files {group}_N{n}.jsonl from build_stage2_data.py
TRAIN_SIZES = [100, 500, 2000]
VAL_FILE = 'heldout_B_best.jsonl'         # eval loss / best checkpoint; set None to disable
HP_OVERRIDES = {}                         # e.g. {'num_train_epochs': 1}; same HP for every run (logged)
MAX_STEPS = -1                            # >0 only for a quick smoke test

# --- Stage 4
EVAL_RUNS = None                      # None -> every trained run found in stage3/; or a list of run names
EVAL_CONTROL_RELEASED_WITH_1X3 = True # also evaluate the released model with 1x3 in library + the new prompt
NEW_BRICK_MASS = 0.00119              # midpoint of 1x2 and 1x4 masses; same value as newbrick/stage1.py (an estimate)
''')

md(r'''
## Setup: Drive, dependencies, BrickGPT

We install BrickGPT with `--no-deps`: its `pyproject.toml` requires `bpy` (Blender, only used for rendering) and
pins `numpy<2`, both of which fight with Colab's preinstalled stack. The modules we use (models, data, stability
analysis) only need torch, transformers, peft, gurobipy, networkx, numpy. Versions in BrickGPT's `uv.lock`:
transformers 4.51.3, peft 0.15.2, trl 0.17.0, gurobipy 12.0.2; our working local venv: transformers 4.57.6,
peft 0.21.2, gurobipy 13.0.3. We pin transformers < 5 (untested with 5.x).
''')

code(r'''
import os, sys
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
OUT = DRIVE_DIR if USE_DRIVE else LOCAL_DIR
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
print('outputs ->', OUT)
''')

code(r'''
!pip install -q "transformers>=4.51,<5" "peft>=0.15" accelerate datasets gurobipy networkx
!test -d /content/BrickGPT || git clone -q {BRICKGPT_REPO} /content/BrickGPT
!cd /content/BrickGPT && git checkout -q {BRICKGPT_COMMIT} && git log -1 --oneline
!pip install -q --no-deps -e /content/BrickGPT
!pip uninstall -y -q torchao  # Colab's old torchao breaks new peft ("incompatible version of torchao"); not needed
''')

md(r'''
## Write helpers.py

Identical copy of `newbrick/colab/helpers.py` (1x3 library patch from `pilot/common.py`, model merging, resumable
generation, scoring, LoRA training). Alternative: upload that file to `/content/helpers.py` and skip this cell.
''')

code('%%writefile /content/helpers.py\n' + HELPERS)

code(r'''
sys.path[:0] = ['/content', '/content/BrickGPT/src']
import importlib, json
import pandas as pd
import torch
import helpers as H
importlib.reload(H)
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
assert not H.library_has_1x3(), 'library must start without the 1x3 brick'
''')

md(r'''
## Gurobi (WLS academic licence)

BrickGPT's physics stability check (and physics-informed rollback with `use_gurobi=True`) needs Gurobi with a full
licence; the pip `gurobipy` restricted licence fails with "Model too large" on real structures (we saw this
locally on 10/10 test structures).

* Put your `gurobi.lic` (WLS type) at `GUROBI_LIC_DRIVE_PATH` on Drive, or upload it when prompted below.
  **Never paste the licence contents into the notebook.** It is copied to `/content/gurobi.lic` (not saved to Drive
  by this notebook) and `GRB_LICENSE_FILE` is pointed at it.
* **Uncertain:** whether the Gurobi academic WLS licence terms allow use on Google Colab (cloud VM, not a
  university machine) is something we have *not* verified — check the licence terms / ask Gurobi before relying on it.
  Also unknown: whether the WLS licence limits concurrent sessions.
* **Fallback:** set `USE_GUROBI = False`. Then (a) rollback ON uses BrickGPT's simpler connectivity check during
  generation (a different method — the condition is renamed `rollback_on_conn`), and (b) post-hoc scoring uses
  connectivity, which is **not** a substitute for Min/Mean Stability. All generated bricks are saved, so Gurobi
  stability can be re-scored later on a machine with the licence (see the last section). Rollback OFF never calls a
  stability checker during generation, so its outputs are identical either way.
''')

code(r'''
if USE_GUROBI:
    import shutil
    if os.path.exists(GUROBI_LIC_DRIVE_PATH):
        shutil.copy(GUROBI_LIC_DRIVE_PATH, '/content/gurobi.lic')
        print('copied licence from Drive')
    elif not os.path.exists('/content/gurobi.lic'):
        from google.colab import files
        print('Upload gurobi.lic (WLS academic licence file)')
        up = files.upload()
        name = next(iter(up))
        os.replace(name, '/content/gurobi.lic')
    os.environ['GRB_LICENSE_FILE'] = '/content/gurobi.lic'
    import gurobipy as gp
    try:
        m = gp.Model('licence_test'); x = m.addVars(3000); m.update(); m.optimize()  # >2000 vars: fails on the restricted licence
        print('Gurobi OK:', gp.gurobi.version())
    except gp.GurobiError as e:
        print('Gurobi FAILED:', e, '\n-> fix the licence or set USE_GUROBI = False and re-run from the PARAMETERS cell')
        raise
else:
    print('USE_GUROBI=False: connectivity check only; re-score stability locally later.')
''')

code(r'''
# Released BrickGPT, merged into one full model directory (brickgpt.models.LLM loads a full model path)
RELEASED_DIR = H.build_merged_model(f'{MODELS_DIR}/brickgpt-merged', base_id=BASE_MODEL, adapter_id=BRICKGPT_ADAPTER)
''')

md(r'''
# Stage 0 — baseline (original brick library, original prompt)

**How rollback is switched off** (from `pilot/BrickGPT/src/brickgpt/models/brickgpt.py`, `BrickGPT.__call__`):

```python
for regeneration_num in range(self.max_regenerations + 1):
    bricks, ... = self._generate_structure(caption, starting_bricks=starting_bricks)
    if self.max_regenerations == 0 or self._is_stable(bricks):
        break
    ...
    starting_bricks = self._remove_all_bricks_after_first_unstable_brick(bricks)
```

* `max_regenerations=0` → the loop runs once and breaks *before* `_is_stable` is evaluated (short-circuit `or`),
  so no stability check and no rollback happens at all; `n_regenerations` is 0. The config help text says the same:
  "Set to 0 if you want to disable physics-informed rollback." `use_gurobi` is then irrelevant during generation.
* Rollback ON = default `max_regenerations=100`: after each full generation the structure is checked
  (`is_stable()` with Gurobi, or `is_connected()` if `use_gurobi=False`); if unstable, all bricks from the first
  unstable brick on are removed and generation continues from the prefix, up to 100 times.
* Independent of rollback, **per-brick rejection sampling** (`max_brick_rejections=500`) rejects bricks that are
  ill-formatted, not in the library, out of bounds or colliding, and **logit masking** (`use_logit_masking=True`)
  enforces the `hxw (x,y,z)` syntax. Both stay ON in both conditions (the "no physics rollback" ablation). Setting
  `ALSO_DISABLE_REJECTION_IN_OFF=True` additionally sets `max_brick_rejections=0` (then invalid bricks are kept).

**Metrics** (computed post hoc from the saved bricks by `helpers.score_bricks_txt`, so they can be recomputed):
validity rate (non-empty, all bricks in library, in bounds, no collisions), stability rate (valid and no floating
bricks and every brick's Gurobi score < 1, i.e. `BrickStructure.is_stable()`), mean / min stability (per-brick
stability = 1 − min(score, 1), mean/min over bricks, averaged over valid structures — *our* definition; check against
the paper before quoting), mean number of regenerations, plus rejections, brick counts and time per item.
''')

code(r'''
S0 = f'{OUT}/stage0'; os.makedirs(S0, exist_ok=True)
subset_path = f'{S0}/subset.json'
if os.path.exists(subset_path):
    subset = json.load(open(subset_path))
    assert subset['seed'] == SUBSET_SEED and subset['n'] == N_CAPTIONS, 'subset.json differs from PARAMETERS'
    ITEMS = subset['items']
else:
    ITEMS = H.load_test_subset(N_CAPTIONS, SUBSET_SEED, dataset=DATASET, caption_index=CAPTION_INDEX)
    json.dump(dict(dataset=DATASET, split='test', dedup='object_id', seed=SUBSET_SEED, n=N_CAPTIONS,
                   caption_index=CAPTION_INDEX, items=ITEMS), open(subset_path, 'w'), indent=1)
print(len(ITEMS), 'items; first:', ITEMS[0]['caption'])

def cond_cfg(cond, model_dir):
    c = dict(model_name_or_path=model_dir, max_bricks=MAX_BRICKS, max_brick_rejections=MAX_BRICK_REJECTIONS,
             use_gurobi=USE_GUROBI, device='cuda' if torch.cuda.is_available() else 'cpu')
    if cond == 'rollback_on':
        c['max_regenerations'] = MAX_REGENERATIONS_ON
    elif cond == 'rollback_off':
        c['max_regenerations'] = 0
        if ALSO_DISABLE_REJECTION_IN_OFF:
            c['max_brick_rejections'] = 0
    else:
        raise ValueError(cond)
    return c

def cond_name(cond):  # make the stability checker used for rollback explicit in the saved data
    return cond + ('_conn' if (cond == 'rollback_on' and not USE_GUROBI) else '')
''')

code(r'''
# Timing pilot: TIMING_N items per condition, written to a separate file. Use it to extrapolate before the full run.
bg = None
for cond in CONDITIONS:
    bg = H.run_generation(cond_cfg(cond, RELEASED_DIR), ITEMS[:TIMING_N], f'{S0}/timing.jsonl', cond_name(cond),
                          GEN_SEED_BASE, bg=bg)
for cond in CONDITIONS:
    rs = [r for r in H.read_jsonl(f'{S0}/timing.jsonl') if r['cond'] == cond_name(cond)]
    secs = [r['secs'] for r in rs]
    print(f'{cond_name(cond)}: secs/item {secs} -> x{len(ITEMS)} items ~ {sum(secs)/len(secs)*len(ITEMS)/3600:.1f} h '
          f'(very rough: time depends strongly on brick count and regenerations)')
''')

code(r'''
# Stage 0 full run (resumable)
for cond in CONDITIONS:
    bg = H.run_generation(cond_cfg(cond, RELEASED_DIR), ITEMS, f'{S0}/gen.jsonl', cond_name(cond),
                          GEN_SEED_BASE, bg=bg)
del bg; torch.cuda.empty_cache()
''')

code(r'''
# Scoring + metrics (resumable; Gurobi if USE_GUROBI else connectivity)
scored = H.score_file(f'{S0}/gen.jsonl', f'{S0}/scored_{"gurobi" if USE_GUROBI else "conn"}.jsonl', USE_GUROBI)
m0 = H.summarize(scored)
json.dump(dict(metrics=m0, params=dict(N_CAPTIONS=N_CAPTIONS, SUBSET_SEED=SUBSET_SEED, GEN_SEED_BASE=GEN_SEED_BASE,
               MAX_BRICKS=MAX_BRICKS, MAX_REGENERATIONS_ON=MAX_REGENERATIONS_ON, USE_GUROBI=USE_GUROBI)),
          open(f'{S0}/metrics_{"gurobi" if USE_GUROBI else "conn"}.json', 'w'), indent=1)
pd.DataFrame(m0).T
''')

md(r'''
# Stage 3 — LoRA on the Stage-2 data

**Data:** upload the folder written by `newbrick/build_stage2_data.py` (`newbrick/stage2_data/`) to `STAGE2_DIR`
on Drive. Format (same as BrickGPT's `prepare_finetuning_dataset.py`): one JSON line per (shape, caption),
`{"messages": [system "You are a helpful assistant.", user create_instruction(caption) with "1x3, 3x1" added to the
allowed list, assistant "hxw (x,y,z)\n..."]}`. Files `{B_best,C1,C2}_N{100,500,2000}.jsonl` and
`heldout_B_best.jsonl` (used here as the validation set). `helpers.load_messages_jsonl` also accepts
`{"captions"/"caption", "bricks"}` lines.

**Model:** a *fresh* LoRA on top of the merged released BrickGPT (like `pilot/lora_tiny.py`); the adapter is saved
and merged again for evaluation.

**Hyper-parameters** (defaults = `pilot/BrickGPT/scripts/finetune.zsh`, the original BrickGPT fine-tuning, run with
`trl sft`): LoRA r=32, alpha=16, dropout=0.05, target q_proj,v_proj; lr 2e-3, cosine, 100 warm-up steps, 3 epochs;
batch 2 × grad-accum 4 (effective 8); max_length 8192; eval every 250 steps, save every 500, load best at end; bf16.
Identical for every group/N, overridable via `HP_OVERRIDES`, logged to `train_log.json` in each run dir.
Differences from the original (unavoidable or deliberate, all logged):
* We use HF `Trainer` with our own tokenisation (`apply_chat_template` on the whole conversation) instead of `trl sft`,
  to avoid TRL API drift. `loss_on='all'` trains on all tokens, which we *believe* is what `trl sft` 0.17 does for a
  `messages` dataset without a completion-only collator (not verified); `loss_on='assistant'` masks the prompt.
* T4 has no bf16 → `precision='auto'` uses fp16 autocast on T4 and bf16 on A100 (fp32 master weights in both).
* `dataloader_num_workers` 2 instead of 4 (Colab CPU count).
* Starting point is the released BrickGPT (not raw Llama), and the dataset is much smaller: with N shapes × 5
  captions and effective batch 8, N=100 gives about 63 optimizer steps per epoch, so `warmup_steps=100` covers
  more than half of training and `eval_steps=250` may never trigger (then the final adapter is kept). Decide whether
  to keep the original values (comparability) or scale them (sensible schedule) — and use the same for all groups.
''')

code(r'''
S3 = f'{OUT}/stage3'; os.makedirs(S3, exist_ok=True)
HP = {**H.DEFAULT_HP, **HP_OVERRIDES}
json.dump(HP, open(f'{S3}/hyperparameters.json', 'w'), indent=1)
print(json.dumps(HP, indent=1))
RUNS = [(f'{g}_N{n}', f'{STAGE2_DIR}/{g}_N{n}.jsonl') for g in TRAIN_GROUPS for n in TRAIN_SIZES]
missing = [p for _, p in RUNS if not os.path.exists(p)]
print('runs:', [r for r, _ in RUNS]); print('MISSING files:', missing)
val_path = f'{STAGE2_DIR}/{VAL_FILE}' if VAL_FILE else None
if os.path.exists(f'{STAGE2_DIR}/manifest.json'):
    print(json.dumps(json.load(open(f'{STAGE2_DIR}/manifest.json'))['files'], indent=1)[:3000])
''')

code(r'''
for run, path in RUNS:
    out_dir = f'{S3}/{run}'
    if os.path.exists(f'{out_dir}/train_log.json'):
        print('skip (done):', run); continue
    if not os.path.exists(path):
        print('skip (no data):', path); continue
    print('=== training', run)
    log = H.train_lora(RELEASED_DIR, path, out_dir, HP, val_path=val_path, max_steps=MAX_STEPS, run_name=run)
    print(run, {k: log[k] for k in ['n_train_examples', 'global_steps', 'train_secs', 'final_eval', 'precision_used']})
''')

md(r'''
# Stage 4 — evaluation with the trained adapters (1x3 brick in the library)

Same subset, seeds and conditions as Stage 0. The 1x3 brick (id 99, LDraw part 3622.DAT) is added to the in-memory
library (`helpers.set_1x3_in_library`, the patch from `pilot/common.py`), so generated 1x3/3x1 bricks are not rejected
as `not_in_library`. The prompt is taken from the adapter's own training file (everything before `### Input:\n`), so
evaluation uses exactly the training prompt. Control `released_1x3`: the released model with the same library and the
new prompt (B_best prompt), no new LoRA.

After a runtime restart, re-run the setup cells, the Gurobi cell and the first Stage-0 cell (it reloads `subset.json`
and defines `cond_cfg`) before this section; finished items are skipped.
''')

code(r'''
S4 = f'{OUT}/stage4'; os.makedirs(S4, exist_ok=True)
H.set_1x3_in_library(True, mass=NEW_BRICK_MASS)
assert H.library_has_1x3()
runs = EVAL_RUNS or sorted(d for d in os.listdir(S3) if os.path.exists(f'{S3}/{d}/train_log.json'))
evals = [(r, f'{S3}/{r}', f'{STAGE2_DIR}/{r}.jsonl') for r in runs]
if EVAL_CONTROL_RELEASED_WITH_1X3:
    ref = next((p for _, _, p in evals if os.path.exists(p)), f'{STAGE2_DIR}/B_best_N100.jsonl')
    evals = [('released_1x3', None, ref)] + evals
for name, adapter, data_path in evals:
    model_dir = RELEASED_DIR if adapter is None else H.build_merged_model(
        f'{MODELS_DIR}/eval-{name}', base_id=RELEASED_DIR, adapter_id=None, extra_adapter=adapter,
        tokenizer_id=RELEASED_DIR)
    fn = H.instruction_fn_from_prefix(H.prefix_from_dataset(data_path))
    d = f'{S4}/{name}'; os.makedirs(d, exist_ok=True)
    bg = None
    for cond in CONDITIONS:
        bg = H.run_generation(cond_cfg(cond, model_dir), ITEMS, f'{d}/gen.jsonl', cond_name(cond), GEN_SEED_BASE,
                              instruction_fn=fn, bg=bg, extra_fields=dict(run=name, library_1x3=True))
    del bg; torch.cuda.empty_cache()
    if adapter is not None:  # free local disk; the adapter itself stays on Drive
        import shutil; shutil.rmtree(model_dir, ignore_errors=True)
    tag = 'gurobi' if USE_GUROBI else 'conn'
    scored = H.score_file(f'{d}/gen.jsonl', f'{d}/scored_{tag}.jsonl', USE_GUROBI)
    json.dump(H.summarize(scored), open(f'{d}/metrics_{tag}.json', 'w'), indent=1)
''')

code(r'''
# Summary table: Stage 0 baseline + every Stage-4 run
tag = 'gurobi' if USE_GUROBI else 'conn'
rows = {}
for cond, m in json.load(open(f'{S0}/metrics_{tag}.json'))['metrics'].items():
    rows[('baseline', cond)] = m
for name in sorted(os.listdir(S4)):
    p = f'{S4}/{name}/metrics_{tag}.json'
    if os.path.exists(p):
        for cond, m in json.load(open(p)).items():
            rows[(name, cond)] = m
summary = pd.DataFrame(rows).T
summary.to_csv(f'{S4}/summary_{tag}.csv')
cols = ['n', 'validity_rate', 'stability_rate', 'mean_stability', 'min_stability', 'mean_regenerations',
        'mean_n_bricks', 'mean_n_1x3', 'frac_with_1x3', 'mean_secs', 'n_generation_errors', 'n_checker_errors']
summary[cols]
''')

md(r'''
## Re-scoring stability locally (e.g. after a `USE_GUROBI=False` run)

All generated structures are in `gen.jsonl` (field `bricks`). On a machine with Gurobi and the licence, with our repo:

```python
import sys; sys.path[:0] = ['pilot/BrickGPT/src', 'newbrick/colab']
import helpers as H
H.set_1x3_in_library(True)            # only for Stage-4 files (harmless for Stage 0: no 1x3 bricks there)
scored = H.score_file('stage0/gen.jsonl', 'stage0/scored_gurobi.jsonl', use_gurobi=True)
print(H.summarize(scored))
```

Caveat: this re-scores the final structures; it cannot turn a connectivity-based rollback (`rollback_on_conn`)
into a Gurobi-based one — that requires regenerating with `USE_GUROBI=True`.
''')

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                 'language_info': {'name': 'python'}},
    'cells': [],
}
for i, (kind, src) in enumerate(cells):
    lines = src.splitlines(keepends=True)
    c = {'cell_type': kind, 'id': f'cell{i:02d}', 'metadata': {}, 'source': lines}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    nb['cells'].append(c)
out = HERE / 'brickgpt_newbrick.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print('wrote', out, len(nb['cells']), 'cells')
