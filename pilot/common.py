"""Shared helpers for the 1x3-brick pilot.

- Adds a 1x3 brick (LDraw part 3622) to BrickGPT's brick library at runtime.
- Loads BrickGPT = Llama-3.2-1B-Instruct (ungated unsloth mirror, same weights) + the official LoRA adapter.
- Builds prompts for each condition and scores candidate bricks by log-probability.
- Merges adjacent 1x1 + 1x2 bricks into 1x3 bricks (makes 1x3 data with no new labels).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'BrickGPT' / 'src'))

import importlib  # noqa: E402

bl = importlib.import_module('brickgpt.data.brick_library')

# ---- add the 1x3 brick to the library (in memory only) ----
NEW_ID = '99'
bl.brick_library[NEW_ID] = {'height': 1, 'width': 3, 'mass': 0.00122, 'inventory': 100000, 'partID': '3622.DAT'}
bl._dimensions_to_brick_id_dict[(1, 3)] = int(NEW_ID)

from brickgpt.data import Brick, BrickStructure  # noqa: E402

import torch  # noqa: E402

MERGED_DIR = ROOT / 'models' / 'brickgpt-merged'


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not MERGED_DIR.exists():
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained('unsloth/Llama-3.2-1B-Instruct', torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(base, 'AvaLovelace/BrickGPT').merge_and_unload()
        tok = AutoTokenizer.from_pretrained('AvaLovelace/BrickGPT')
        model.save_pretrained(MERGED_DIR)
        tok.save_pretrained(MERGED_DIR)
    tok = AutoTokenizer.from_pretrained(MERGED_DIR)
    model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, torch_dtype=torch.float32)
    import os
    if os.environ.get('LORA'):  # optional extra adapter, e.g. models/lora_1x3
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.environ['LORA']).merge_and_unload()
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return tok, model


# ---------------- prompts ----------------
ORIG_DIMS = '2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2'
NEW_DIMS = ORIG_DIMS + ', 1x3, 3x1'
DESCRIPTION = ('The 1x3 brick (3x1 when rotated) is a NEW brick: it covers 3 studs in a row. '
               'Use it wherever a straight run of exactly 3 studs needs to be filled, '
               'instead of combining a 1x2 and a 1x1.\n')


def instruction(caption: str, dims: str, extra: str = '', demos: list[dict] | None = None) -> str:
    s = ('Create a LEGO model of the input. Format your response as a list of bricks: '
         '<brick dimensions> <brick position>, where the brick position is (x,y,z).\n'
         f'Allowed brick dimensions are {dims}.\n'
         'All bricks are 1 unit tall.\n')
    s += extra
    if demos:
        s += '\nHere are some example LEGO models:\n\n'
        s += '\n\n'.join(f'### Input:\n{d["caption"]}\n\n### Output:\n{d["bricks"]}' for d in demos)
        s += '\n\n'
    s += '\n### Input:\n' + caption
    return s


def chat_prefix_ids(tok, user_msg: str, assistant_prefix: str):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': user_msg}]
    if assistant_prefix:
        messages.append({'role': 'assistant', 'content': assistant_prefix})
        return tok.apply_chat_template(messages, continue_final_message=True, return_tensors='pt')
    return tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')


@torch.no_grad()
def score_candidates(tok, model, prefix_ids, candidates: list[str]) -> list[float]:
    """Sum log P(candidate tokens | prefix) for each candidate string (one forward pass over prefix, cached)."""
    out = model(prefix_ids, use_cache=True)
    past = out.past_key_values
    last_logits = out.logits[0, -1]
    scores = []
    for c in candidates:
        ids = tok(c, add_special_tokens=False, return_tensors='pt').input_ids
        lp = torch.log_softmax(last_logits, -1)[ids[0, 0]].item()
        if ids.shape[1] > 1:
            import copy
            o2 = model(ids, past_key_values=copy.deepcopy(past), use_cache=True)
            lps = torch.log_softmax(o2.logits[0, :-1], -1)
            lp += lps.gather(1, ids[0, 1:, None]).sum().item()
        scores.append(lp)
    return scores


# ---------------- 1x3 merging ----------------
def parse(bricks_txt: str) -> list[Brick]:
    return [Brick.from_txt(l) for l in bricks_txt.strip().split('\n') if l.strip()]


def merge_1x3(bricks: list[Brick]) -> tuple[list[Brick], list[int]]:
    """Greedily merge a 1x1 with a collinear, touching 1x2 in the same layer into one 1x3.
    The merged brick takes the place (in build order) of whichever of the two came first.
    Returns new brick list and indices (in the new list) of merged 1x3 bricks."""
    bricks = list(bricks)
    used = set()
    replace = {}  # idx -> new brick ; removed idx -> None
    by_cell = {}
    for i, b in enumerate(bricks):
        for x in range(b.x, b.x + b.h):
            for y in range(b.y, b.y + b.w):
                by_cell[(x, y, b.z)] = i
    for i, a in enumerate(bricks):
        if i in used or (a.h, a.w) != (1, 1):
            continue
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            j = by_cell.get((a.x + dx, a.y + dy, a.z))
            if j is None or j in used or j == i:
                continue
            b = bricks[j]
            horiz = dy != 0  # along y
            if horiz and (b.h, b.w) == (1, 2):
                y0 = min(a.y, b.y)
                m = Brick(h=1, w=3, x=a.x, y=y0, z=a.z)
            elif not horiz and (b.h, b.w) == (2, 1):
                x0 = min(a.x, b.x)
                m = Brick(h=3, w=1, x=x0, y=a.y, z=a.z)
            else:
                continue
            # check the merged footprint is exactly a + b
            cells = {(x, y) for x in range(m.x, m.x + m.h) for y in range(m.y, m.y + m.w)}
            if len(cells) != 3:
                continue
            first, second = min(i, j), max(i, j)
            replace[first] = m
            replace[second] = None
            used.update([i, j])
            break
    out, merged_idx = [], []
    for k, b in enumerate(bricks):
        if k in replace:
            if replace[k] is None:
                continue
            merged_idx.append(len(out))
            out.append(replace[k])
        else:
            out.append(b)
    return out, merged_idx


def to_txt(bricks: list[Brick]) -> str:
    return ''.join(b.to_txt() for b in bricks)
