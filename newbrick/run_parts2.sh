#!/bin/bash
# Stage 1c: wider common bricks (2x8, 2x10, 4x6) and all extra sizes at once ('Ball'), same 40 hard shapes.
# Group A from run_ab.sh is the control. Analysis runs at the end -> stage1_out/analysis_<group>.txt
cd "$(dirname "$0")"
export SHAPES=stage1_out/hard_shapes.json PYTHONIOENCODING=utf-8 TOL_FIX=1
PY=../pilot/.venv/Scripts/python.exe
GROUPS="B2x8 B2x10 B4x6 Ball"
for g in $GROUPS; do
  for s in 0 20; do
    JITTER=1 START=$s $PY stage1.py $g 20 10 stage1_out/ab_${g}_jit_$s.jsonl > stage1_out/ab_${g}_jit_$s.log 2>&1 &
  done
  JITTER=0 START=0 $PY stage1.py $g 40 1 stage1_out/ab_${g}_det.jsonl > stage1_out/ab_${g}_det.log 2>&1 &
done
wait
for g in $GROUPS; do $PY analyze_ab.py $g > stage1_out/analysis_$g.txt 2>&1; done
echo ALL_DONE
