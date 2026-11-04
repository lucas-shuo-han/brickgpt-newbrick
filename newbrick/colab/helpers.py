"""Helpers for the BrickGPT + 1x3-brick Colab notebook (brickgpt_newbrick.ipynb).

Self-contained copy of what the notebook needs from our local code (pilot/common.py, gen.py, lora_tiny.py),
so Colab does not need our repo. Requires the BrickGPT repo on sys.path (``import brickgpt`` must work).

Sections
  1. Brick library patch (1x3 brick, id 99, LDraw part 3622.DAT)          -- from pilot/common.py
  2. Prompts / instruction functions                                        -- from pilot/common.py + brickgpt.py
  3. Model building (released BrickGPT = Llama-3.2-1B-Instruct + LoRA; optional extra adapter merged on top)
  4. Test subset, generation loop (resumable JSONL)
  5. Post-hoc scoring (validity, Gurobi stability or connectivity fallback) + summary metrics
  6. LoRA training (HF Trainer, hyper-parameters of scripts/finetune.zsh by default)
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

# ======================================================================================
# 1. Brick library patch
# ======================================================================================
NEW_ID = '99'
NEW_BRICK = {'height': 1, 'width': 3, 'mass': 0.00119, 'inventory': 100000, 'partID': '3622.DAT'}  # midpoint of 1x2 and 1x4, same as newbrick/stage1.py


def _bl_module():
    # NOTE: `import brickgpt.data.brick_library as bl` returns the *dict* (brickgpt.data re-exports a dict named
    # brick_library that shadows the submodule), hence importlib, as in pilot/common.py.
    import importlib
    return importlib.import_module('brickgpt.data.brick_library')


def set_1x3_in_library(enabled: bool, mass: float | None = None) -> None:
    """Add (enabled=True) or remove (False) the 1x3 brick from BrickGPT's in-memory brick library.

    Same patch as pilot/common.py: brick_library['99'] + _dimensions_to_brick_id_dict[(1, 3)] = 99.
    The dict objects are mutated in place, so modules that imported them by name (brick_structure,
    stability analysis) see the change. Nothing is written to disk.
    """
    bl = _bl_module()
    if enabled:
        props = dict(NEW_BRICK)
        if mass is not None:
            props['mass'] = mass
        bl.brick_library[NEW_ID] = props
        bl._dimensions_to_brick_id_dict[(1, 3)] = int(NEW_ID)
    else:
        bl.brick_library.pop(NEW_ID, None)
        if bl._dimensions_to_brick_id_dict.get((1, 3)) == int(NEW_ID):
            del bl._dimensions_to_brick_id_dict[(1, 3)]


def library_has_1x3() -> bool:
    return (1, 3) in _bl_module()._dimensions_to_brick_id_dict


# ======================================================================================
# 2. Prompts
# ======================================================================================
ORIG_DIMS = '2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2'
NEW_DIMS = ORIG_DIMS + ', 1x3, 3x1'
SYSTEM_PROMPT = 'You are a helpful assistant.'
INPUT_MARKER = '### Input:\n'


def create_instruction(caption: str, dims: str = ORIG_DIMS) -> str:
    """brickgpt.models.brickgpt.create_instruction with the dimension list parameterised.
    With dims=ORIG_DIMS the output is byte-identical to the original."""
    return ('Create a LEGO model of the input. Format your response as a list of bricks: '
            '<brick dimensions> <brick position>, where the brick position is (x,y,z).\n'
            f'Allowed brick dimensions are {dims}.\n'
            'All bricks are 1 unit tall.\n\n'
            '### Input:\n'
            f'{caption}')


def instruction_fn_from_prefix(prefix: str):
    """Instruction function = fixed prefix (everything up to and including '### Input:\\n') + caption."""
    def fn(caption: str) -> str:
        return prefix + caption
    return fn


def prefix_from_dataset(jsonl_path: str | Path) -> str:
    """Read the first example of a {"messages": [...]} JSONL and return the user-prompt prefix, so evaluation
    uses exactly the prompt the adapter was trained with."""
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                msgs = json.loads(line)['messages']
                user = next(m['content'] for m in msgs if m['role'] == 'user')
                i = user.rfind(INPUT_MARKER)
                if i < 0:
                    raise ValueError(f'No "{INPUT_MARKER!r}" in user prompt of {jsonl_path}')
                return user[:i + len(INPUT_MARKER)]
    raise ValueError(f'{jsonl_path} is empty')


# ======================================================================================
# 3. Model building
# ======================================================================================
def build_merged_model(out_dir: str | Path, base_id: str = 'unsloth/Llama-3.2-1B-Instruct',
                       adapter_id: str | None = 'AvaLovelace/BrickGPT', extra_adapter: str | Path | None = None,
                       tokenizer_id: str = 'AvaLovelace/BrickGPT') -> str:
    """Released BrickGPT = base + official LoRA, merged (as pilot/common.py load_model). If extra_adapter is
    given, that LoRA (trained on top of the merged model) is merged too. To merge only an extra adapter into an
    already-merged released model: base_id=<merged dir>, adapter_id=None, tokenizer_id=<merged dir>. Saved as a full model directory,
    because brickgpt.models.LLM loads with AutoModelForCausalLM.from_pretrained(path). Weights saved in fp32."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out_dir = Path(out_dir)
    if (out_dir / 'config.json').exists():
        print(f'[build_merged_model] reuse {out_dir}')
        return str(out_dir)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
    if adapter_id is not None:
        model = PeftModel.from_pretrained(model, adapter_id).merge_and_unload()
    if extra_adapter is not None:
        model = PeftModel.from_pretrained(model, str(extra_adapter)).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    del model
    print(f'[build_merged_model] saved {out_dir}')
    return str(out_dir)


# ======================================================================================
# 4. Test subset + generation
# ======================================================================================
def load_test_subset(n: int, seed: int, dataset: str = 'AvaLovelace/StableText2Brick', split: str = 'test',
                     caption_index: int = 0) -> list[dict]:
    """Deduplicate the test split by object_id (as pilot/gen.py), then draw n objects without replacement
    with numpy RandomState(seed). Caption = captions[caption_index]."""
    from datasets import load_dataset
    df = load_dataset(dataset, split=split).to_pandas()
    df = df.drop_duplicates('object_id').reset_index(drop=True)
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(df), size=min(n, len(df)), replace=False).tolist())
    return [{'item': k, 'row': int(i), 'object_id': str(df.loc[i, 'object_id']),
             'caption': str(df.loc[i, 'captions'][caption_index]), 'gt_bricks': str(df.loc[i, 'bricks'])}
            for k, i in enumerate(idx)]


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f'WARNING: skipped bad line in {path}')
    return out


def append_jsonl(path: str | Path, rec: dict) -> None:
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def count_1x3(bricks) -> int:
    return sum(1 for b in bricks if (b.h, b.w) in [(1, 3), (3, 1)])


def run_generation(cfg_kwargs: dict, items: list[dict], out_path: str | Path, cond: str, seed_base: int,
                   instruction_fn=None, extra_fields: dict | None = None, bg=None):
    """Generate one structure per item with brickgpt.models.BrickGPT and append one JSONL line per item.
    Resumable: items already present in out_path for this `cond` are skipped.
    Seed for item k = seed_base + k (transformers.set_seed), identical across conditions.
    Returns the BrickGPT object so it can be reused (pass it back via bg= to avoid reloading the model)."""
    if TOL_FIX:
        patch_stability_tolerance()
    import torch
    import transformers
    from brickgpt.models import BrickGPT, BrickGPTConfig
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r['item'] for r in read_jsonl(out_path) if r.get('cond') == cond}
    todo = [it for it in items if it['item'] not in done]
    print(f'[{cond}] {len(done)} done, {len(todo)} to go -> {out_path}')
    from brickgpt.models import create_instruction as _orig_instruction
    cfg = BrickGPTConfig(**cfg_kwargs)
    if bg is None:
        bg = BrickGPT(cfg)
        bg._model_path = cfg.model_name_or_path
    else:  # reuse loaded model, update the generation settings
        assert getattr(bg, '_model_path', None) == cfg.model_name_or_path, 'bg was loaded from a different model'
        for k in ['world_dim', 'max_bricks', 'max_brick_rejections', 'use_logit_masking', 'max_regenerations',
                  'use_gurobi', 'temperature', 'temperature_increase', 'max_temperature', 'top_k', 'top_p']:
            setattr(bg, k, getattr(cfg, k))
    bg.instruction_fn = instruction_fn or _orig_instruction  # None -> original BrickGPT prompt
    for it in todo:
        seed = seed_base + it['item']
        transformers.set_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        err = None
        try:
            res = bg(it['caption'])
            bricks_txt = res['bricks'].to_txt()
            n_bricks = len(res['bricks'])
            n13 = count_1x3(res['bricks'].bricks)
            rej = dict(res['rejection_reasons'])
            n_regen = res['n_regenerations']
        except Exception as e:  # e.g. Gurobi licence error during rollback; keep going, record it
            bricks_txt, n_bricks, n13, rej, n_regen = '', 0, 0, {}, None
            err = f'{type(e).__name__}: {e}'[:500]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rec = dict(cond=cond, item=it['item'], object_id=it['object_id'], caption=it['caption'], seed=seed,
                   n_bricks=n_bricks, n_1x3=n13, rejection_reasons=rej,
                   n_rejections=int(sum(rej.values())), n_regenerations=n_regen,
                   secs=round(time.time() - t0, 2), error=err, bricks=bricks_txt,
                   cfg={k: v for k, v in cfg_kwargs.items()}, **(extra_fields or {}))
        append_jsonl(out_path, rec)
        print(f'[{cond}] item {it["item"]}: {n_bricks} bricks, n_1x3={n13}, regen={n_regen}, '
              f'rej={rec["n_rejections"]}, {rec["secs"]}s' + (f', ERROR {err}' if err else ''), flush=True)
    return bg


# ======================================================================================
# 5. Scoring + metrics
# ======================================================================================
STAB_TOL = 1e-9
TOL_FIX = True  # apply patch_stability_tolerance() in scoring AND generation-time rollback
_tol_patched = False


def patch_stability_tolerance(tol: float = STAB_TOL) -> None:
    """Fix for a bug in BrickGPT's stability_analysis.py (see newbrick/investigate_score1/REPORT.md):
    a brick gets score 1.0 if any force/torque residual `.X > 0`; Gurobi often returns ~1e-14 float noise,
    which marks EVERY brick of a stable structure as 1.0 (~15% of dataset structures). Re-exec the original
    source with `.X > STAB_TOL` and rebind it where it was imported. Same as newbrick/tolerant_stability.py."""
    global _tol_patched
    if _tol_patched:
        return
    import importlib
    import sys
    mod = importlib.import_module('brickgpt.stability_analysis.stability_analysis')
    src = open(mod.__file__, encoding='utf-8').read()
    assert src.count('.X > 0') == 5, 'stability_analysis.py changed; check the tolerance patch'
    mod.__dict__['STAB_TOL'] = tol
    exec(compile(src.replace('.X > 0', '.X > STAB_TOL'), mod.__file__ + ' (tolerant)', 'exec'), mod.__dict__)
    for u in ['brickgpt.stability_analysis', 'brickgpt.data.brick_structure']:
        m = importlib.import_module(u)
        if hasattr(m, 'stability_score'):
            m.stability_score = mod.stability_score
    _tol_patched = True


def score_bricks_txt(bricks_txt: str, use_gurobi: bool, world_dim: int = 20) -> dict:
    """Post-hoc check of one generated structure (can be re-run locally on the saved JSONL).

    valid  := non-empty, parses, every brick in the CURRENT library, in bounds, no collisions.
    stable := valid and BrickStructure.is_stable() logic (no floating bricks, no collisions,
              max Gurobi stability score < 1). With use_gurobi=False, the connectivity check
              (BrickStructure.is_connected logic) is used instead and stored under *_conn keys.
    Per-brick score s_b = max of the analysis heat-map over the brick's voxels (>= 1 = brick fails);
    per-brick stability = 1 - min(s_b, 1); structure mean/min stability = mean/min over bricks.
    NOTE: this mean/min definition is ours; check it against the BrickGPT paper before comparing numbers.
    """
    from brickgpt.data import Brick, BrickStructure
    r = dict(valid=False, invalid_reason=None, stable=None, max_score=None,
             mean_stability=None, min_stability=None, checker='gurobi' if use_gurobi else 'connectivity',
             checker_error=None, check_secs=None)
    lines = [l for l in bricks_txt.split('\n') if l.strip()]
    if not lines:
        r['invalid_reason'] = 'empty'
        return r
    try:
        bricks = [Brick.from_txt(l) for l in lines]
    except ValueError:
        r['invalid_reason'] = 'ill_formatted'
        return r
    try:
        for b in bricks:
            _ = b.brick_id
    except ValueError:
        r['invalid_reason'] = 'not_in_library'
        return r
    bs = BrickStructure([], world_dim=world_dim)
    for b in bricks:
        if not bs.brick_in_bounds(b):
            r['invalid_reason'] = 'out_of_bounds'
            return r
        bs.add_brick(b)
    if bs.has_collisions():
        r['invalid_reason'] = 'collision'
        return r
    r['valid'] = True
    if TOL_FIX:
        patch_stability_tolerance()
    t0 = time.time()
    try:
        scores = bs.stability_scores() if use_gurobi else bs.connectivity_scores()
    except Exception as e:  # licence / solver errors are recorded, NOT counted as unstable
        r['checker_error'] = f'{type(e).__name__}: {e}'[:500]
        return r
    r['check_secs'] = round(time.time() - t0, 3)
    per_brick = np.array([float(np.max(scores[b.slice])) for b in bs.bricks])
    stab = 1.0 - np.minimum(per_brick, 1.0)
    r['max_score'] = float(per_brick.max())
    r['mean_stability'] = float(stab.mean())
    r['min_stability'] = float(stab.min())
    r['stable'] = bool((not bs.has_floating_bricks()) and per_brick.max() < 1)
    r['has_floating'] = bool(bs.has_floating_bricks())
    return r


def score_file(gen_path: str | Path, scored_path: str | Path, use_gurobi: bool, conds: list[str] | None = None):
    """Score every generated record not yet in scored_path (resumable). Scored line = generated line + scores."""
    gen = read_jsonl(gen_path)
    done = {(r['cond'], r['item']) for r in read_jsonl(scored_path)}
    for g in gen:
        if conds and g['cond'] not in conds:
            continue
        if (g['cond'], g['item']) in done:
            continue
        s = score_bricks_txt(g['bricks'], use_gurobi)
        append_jsonl(scored_path, {**g, **{'score_' + k: v for k, v in s.items()}})
    return read_jsonl(scored_path)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(scored: list[dict]) -> dict:
    """Per-condition metrics. Rates are over all items of the condition; generation errors and checker
    errors are counted separately (and excluded from the stability denominator)."""
    out = {}
    for cond in sorted({r['cond'] for r in scored}):
        rs = [r for r in scored if r['cond'] == cond]
        checked = [r for r in rs if r['score_valid'] and r['score_checker_error'] is None]
        valid_or_invalid = [r for r in rs if r['error'] is None]
        out[cond] = dict(
            n=len(rs),
            n_generation_errors=sum(r['error'] is not None for r in rs),
            n_checker_errors=sum(r['score_checker_error'] is not None for r in rs),
            checker=sorted({r['score_checker'] for r in rs}),
            validity_rate=_mean([float(r['score_valid']) for r in valid_or_invalid]),
            # stable rate over all non-error items (invalid -> not stable), as a fraction of generated structures
            stability_rate=(sum(bool(r['score_stable']) for r in checked) /
                            max(1, len([r for r in valid_or_invalid if r['score_checker_error'] is None]))),
            mean_stability=_mean([r['score_mean_stability'] for r in checked]),
            min_stability=_mean([r['score_min_stability'] for r in checked]),
            mean_regenerations=_mean([r['n_regenerations'] for r in rs]),
            mean_rejections=_mean([r['n_rejections'] for r in rs]),
            mean_n_bricks=_mean([r['n_bricks'] for r in rs]),
            mean_n_1x3=_mean([r['n_1x3'] for r in rs]),
            frac_with_1x3=_mean([float(r['n_1x3'] > 0) for r in rs]),
            mean_secs=_mean([r['secs'] for r in rs]),
            invalid_reasons=dict(Counter(r['score_invalid_reason'] for r in rs if r['score_invalid_reason'])),
        )
    return out


# ======================================================================================
# 6. LoRA training
# ======================================================================================
# Defaults = pilot/BrickGPT/scripts/finetune.zsh (original BrickGPT fine-tuning). Change in the notebook.
DEFAULT_HP = dict(
    lora_r=32, lora_alpha=16, lora_dropout=0.05, lora_target_modules=['q_proj', 'v_proj'],
    learning_rate=2e-3, lr_scheduler_type='cosine', warmup_steps=100, num_train_epochs=3,
    per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=4,
    max_length=8192, eval_steps=250, save_steps=500, load_best_model_at_end=True,
    precision='auto',            # 'auto' -> bf16 if the GPU supports it (A100) else fp16 (T4); or 'bf16'/'fp16'/'fp32'
    loss_on='all',               # 'all' tokens (what `trl sft` does on a "messages" dataset by default, we believe)
                                 # or 'assistant' (prompt tokens masked, as pilot/lora_tiny.py)
    gradient_checkpointing=False,
    dataloader_num_workers=2,    # finetune.zsh used 4; Colab has 2 CPU cores
    seed=42,
)


def load_messages_jsonl(path: str | Path, instruction_fn=None) -> list[list[dict]]:
    """Accepts {"messages": [...]} lines (stage2_data / prepare_finetuning_dataset.py format). Lines with
    {"caption"/"captions", "bricks"} are converted with instruction_fn (one example per caption)."""
    out = []
    for r in read_jsonl(path):
        if 'messages' in r:
            out.append(r['messages'])
        elif 'bricks' in r:
            fn = instruction_fn or (lambda c: create_instruction(c, NEW_DIMS))
            caps = r.get('captions') or [r['caption']]
            for c in caps:
                out.append([{'role': 'system', 'content': SYSTEM_PROMPT},
                            {'role': 'user', 'content': fn(c)},
                            {'role': 'assistant', 'content': r['bricks']}])
        else:
            raise ValueError(f'Unknown record format in {path}: keys {list(r)}')
    return out


def tokenize_messages(tok, messages: list[dict], max_length: int, loss_on: str) -> dict:
    """Chat-template the full conversation (like TRL does). NOTE: the Llama-3 template strips the assistant
    content, so the final '\\n' of the brick list is dropped before <|eot_id|> -- same as the original
    fine-tuning, which also went through apply_chat_template."""
    ids = tok.apply_chat_template(messages, tokenize=True)
    if hasattr(ids, 'keys'):  # some transformers versions return a BatchEncoding
        ids = ids['input_ids']
    ids = list(ids)[:max_length]
    labels = list(ids)
    if loss_on == 'assistant':
        prompt = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        if hasattr(prompt, 'keys'):
            prompt = prompt['input_ids']
        n = len(prompt)
        assert ids[:n] == list(prompt)[:len(ids[:n])], 'prompt is not a prefix of the full conversation'
        labels[:n] = [-100] * min(n, len(labels))
    elif loss_on != 'all':
        raise ValueError(loss_on)
    return {'input_ids': ids, 'labels': labels}


class _Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        import torch
        L = max(len(f['input_ids']) for f in feats)
        ids = torch.full((len(feats), L), self.pad_id, dtype=torch.long)
        lab = torch.full((len(feats), L), -100, dtype=torch.long)
        att = torch.zeros((len(feats), L), dtype=torch.long)
        for i, f in enumerate(feats):
            n = len(f['input_ids'])
            ids[i, :n] = torch.tensor(f['input_ids'])
            lab[i, :n] = torch.tensor(f['labels'])
            att[i, :n] = 1
        return {'input_ids': ids, 'labels': lab, 'attention_mask': att}


class _ListDataset:
    def __init__(self, xs):
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return self.xs[i]


def train_lora(merged_dir: str, train_path: str | Path, out_dir: str | Path, hp: dict,
               val_path: str | Path | None = None, max_steps: int = -1, run_name: str | None = None) -> dict:
    """Train a fresh LoRA on top of the merged released BrickGPT (as pilot/lora_tiny.py, but with HF Trainer
    and the finetune.zsh hyper-parameters). Saves the best (by eval loss, if val_path) adapter to out_dir and
    a JSON log (hp, dataset sizes, log_history, versions, wall time) to out_dir/train_log.json."""
    import torch
    import transformers
    import peft
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    hp = {**DEFAULT_HP, **hp}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transformers.set_seed(hp['seed'])
    random.seed(hp['seed'])

    tok = AutoTokenizer.from_pretrained(merged_dir)
    if tok.pad_token is None or tok.pad_token_id == tok.eos_token_id:
        # finetuning_config_files: pad must differ from eos so the model learns to emit <|eot_id|>
        tok.pad_token = '<|finetune_right_pad_id|>'
    train_msgs = load_messages_jsonl(train_path)
    train_ds = _ListDataset([tokenize_messages(tok, m, hp['max_length'], hp['loss_on']) for m in train_msgs])
    val_ds = None
    if val_path and Path(val_path).exists():
        val_ds = _ListDataset([tokenize_messages(tok, m, hp['max_length'], hp['loss_on'])
                               for m in load_messages_jsonl(val_path)])

    prec = hp['precision']
    if prec == 'auto':
        prec = 'bf16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
            'fp16' if torch.cuda.is_available() else 'fp32')
    # Master weights in fp32 (LoRA params must be fp32 for the fp16 GradScaler); autocast does the rest.
    model = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch.float32)
    if hp['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=hp['lora_r'], lora_alpha=hp['lora_alpha'],
                                             lora_dropout=hp['lora_dropout'],
                                             target_modules=list(hp['lora_target_modules']),
                                             task_type='CAUSAL_LM'))
    model.print_trainable_parameters()

    use_eval = val_ds is not None
    args = TrainingArguments(
        output_dir=str(out_dir / 'checkpoints'),
        run_name=run_name,
        per_device_train_batch_size=hp['per_device_train_batch_size'],
        per_device_eval_batch_size=hp['per_device_eval_batch_size'],
        gradient_accumulation_steps=hp['gradient_accumulation_steps'],
        learning_rate=hp['learning_rate'],
        lr_scheduler_type=hp['lr_scheduler_type'],
        warmup_steps=hp['warmup_steps'],
        num_train_epochs=hp['num_train_epochs'],
        max_steps=max_steps,
        eval_strategy='steps' if use_eval else 'no',
        eval_steps=hp['eval_steps'],
        save_strategy='steps',
        save_steps=hp['save_steps'],
        save_total_limit=2,
        load_best_model_at_end=bool(hp['load_best_model_at_end'] and use_eval),
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        logging_steps=10,
        bf16=(prec == 'bf16'),
        fp16=(prec == 'fp16'),
        dataloader_num_workers=hp['dataloader_num_workers'],
        report_to='none',
        seed=hp['seed'],
        remove_unused_columns=False,
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=_Collator(tok.pad_token_id))
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    final_eval = trainer.evaluate() if use_eval else None
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log = dict(run_name=run_name, train_path=str(train_path), val_path=str(val_path) if val_path else None,
               n_train_examples=len(train_ds), n_val_examples=len(val_ds) if val_ds else 0,
               mean_train_tokens=float(np.mean([len(x['input_ids']) for x in train_ds.xs])),
               n_truncated=sum(len(x['input_ids']) >= hp['max_length'] for x in train_ds.xs),
               hp=hp, precision_used=prec, max_steps=max_steps, train_secs=round(secs, 1),
               global_steps=trainer.state.global_step, best_checkpoint=trainer.state.best_model_checkpoint,
               final_eval=final_eval, log_history=trainer.state.log_history,
               versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                             peft=peft.__version__),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    with open(out_dir / 'train_log.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=1, default=str)
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return log
