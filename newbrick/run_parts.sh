#!/bin/bash
# Stage 1b: other new parts (2x3, 1x10, 1x12) on the same 40 hard shapes; group A from run_ab.sh is the control.
cd "$(dirname "$0")"
export SHAPES=stage1_out/hard_shapes.json PYTHONIOENCODING=utf-8 TOL_FIX=1
PY=../pilot/.venv/Scripts/python.exe
for g in B2x3 B1x10 B1x12; do
  for s in 0 20; do
    JITTER=1 START=$s $PY stage1.py $g 20 10 stage1_out/ab_${g}_jit_$s.jsonl > stage1_out/ab_${g}_jit_$s.log 2>&1 &
  done
  JITTER=0 START=0 $PY stage1.py $g 40 1 stage1_out/ab_${g}_det.jsonl > stage1_out/ab_${g}_det.log 2>&1 &
done
wait
