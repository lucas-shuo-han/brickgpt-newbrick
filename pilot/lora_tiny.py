"""Tiny CPU LoRA: can BrickGPT learn the 1x3 brick in weights at all?

Trains a fresh LoRA (on top of merged BrickGPT) on short StableText2Brick TRAIN structures where
1x1+1x2 pairs were merged into 1x3. Prompt = BrickGPT format with 1x3 listed. Loss on bricks only.
Usage: python lora_tiny.py <n_examples> <max_bricks_per_example>
"""
import json
import random
import sys
import time

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model

from common import NEW_DIMS, instruction, parse, merge_1x3, to_txt, load_model, ROOT

random.seed(0)
torch.manual_seed(0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
MAXB = int(sys.argv[2]) if len(sys.argv) > 2 else 60

df = pd.read_parquet(ROOT / 'data' / 'train0.parquet')
ex = []
for _, r in df.iterrows():
    bricks = parse(r['bricks'])
    if len(bricks) > MAXB:
        continue
    merged, idx = merge_1x3(bricks)
    if len(idx) >= 2:
        ex.append((r['captions'][0], to_txt(merged), len(idx)))
random.shuffle(ex)
val = ex[N:N + 20]
ex = ex[:N]
print(f'{len(ex)} train examples, avg 1x3 per example {sum(e[2] for e in ex)/len(ex):.1f}', flush=True)

tok, model = load_model()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                         target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                                         task_type='CAUSAL_LM'))


def encode(cap, bricks):
    msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': instruction(cap, NEW_DIMS)}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt')[0]
    a = tok(bricks, add_special_tokens=False, return_tensors='pt').input_ids[0]
    eot = torch.tensor([tok.convert_tokens_to_ids('<|eot_id|>')])
    ids = torch.cat([p, a, eot])
    labels = ids.clone()
    labels[:len(p)] = -100
    return ids[None], labels[None]


@torch.no_grad()
def val_loss():
    model.eval()
    tot = 0
    for cap, b, _ in val:
        ids, lab = encode(cap, b)
        tot += model(ids, labels=lab).loss.item()
    model.train()
    return tot / len(val)


opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
log = {'train': [], 'val': [(0, val_loss())]}
print('val loss at start', log['val'][0], flush=True)
model.train()
t0 = time.time()
for step, (cap, b, _) in enumerate(ex, 1):
    ids, lab = encode(cap, b)
    loss = model(ids, labels=lab).loss
    loss.backward()
    opt.step(); opt.zero_grad()
    log['train'].append((step, loss.item()))
    if step % 25 == 0:
        log['val'].append((step, val_loss()))
        print(f'step {step} train {loss.item():.3f} val {log["val"][-1][1]:.3f} ({time.time()-t0:.0f}s)', flush=True)

import os
model.save_pretrained(ROOT / 'models' / os.environ.get('LORA_OUT', 'lora_1x3'))
json.dump(log, open(ROOT / os.environ.get('LOG_OUT', 'results_lora_log.json'), 'w'))
print('saved')
