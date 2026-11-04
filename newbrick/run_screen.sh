#!/bin/bash
# Stage 1 screen: 300 train shapes, group A, one layout each (seed 0), 3 parallel processes.
cd "$(dirname "$0")"
for s in 0 100 200; do
  START=$s PYTHONIOENCODING=utf-8 ../pilot/.venv/Scripts/python.exe stage1.py A 100 1 stage1_out/screen_A_$s.jsonl > stage1_out/screen_A_$s.log 2>&1 &
done
wait
