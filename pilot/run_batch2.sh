#!/bin/bash
# Batch 2: bigger LoRA, then forced probes + generation with it.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
# wait for batch 1 (forced probes) to finish
until grep -q -E "^saved|Error" logF1.txt 2>/dev/null; do sleep 30; done
LORA_OUT=lora_1x3_n250 LOG_OUT=results_lora250_log.json .venv/Scripts/python lora_tiny.py 250 60 > logLoRA250.txt 2>&1
LORA=models/lora_1x3_n250 CONDS=orig_prompt,list_desc OUT=results_forced_lora250.json .venv/Scripts/python probe_forced.py > logF2.txt 2>&1
.venv/Scripts/python -c "
import common
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
m=AutoModelForCausalLM.from_pretrained(str(common.MERGED_DIR)); m=PeftModel.from_pretrained(m,'models/lora_1x3_n250').merge_and_unload()
m.save_pretrained('models/brickgpt-lora250-merged'); AutoTokenizer.from_pretrained(str(common.MERGED_DIR)).save_pretrained('models/brickgpt-lora250-merged')
" > logMerge.txt 2>&1
MODEL=models/brickgpt-lora250-merged OUT=results_gen_lora250.json .venv/Scripts/python gen.py 3 60 zeroshot_desc > logG2.txt 2>&1
echo BATCH2_DONE >> logG2.txt
