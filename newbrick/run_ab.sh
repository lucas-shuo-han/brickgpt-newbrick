#!/bin/bash
# Stage 1 main run on the 40 hard shapes (stage1_out/hard_shapes.json), tolerance fix ON.
# jitter: K=10 seeds per group (2 processes each); det: original deterministic layout, K=1.
cd "$(dirname "$0")"
export SHAPES=stage1_out/hard_shapes.json PYTHONIOENCODING=utf-8 TOL_FIX=1
PY=../pilot/.venv/Scripts/python.exe
for g in A B; do
  for s in 0 20; do
    JITTER=1 START=$s $PY stage1.py $g 20 10 stage1_out/ab_${g}_jit_$s.jsonl > stage1_out/ab_${g}_jit_$s.log 2>&1 &
  done
  JITTER=0 START=0 $PY stage1.py $g 40 1 stage1_out/ab_${g}_det.jsonl > stage1_out/ab_${g}_det.log 2>&1 &
done
wait
