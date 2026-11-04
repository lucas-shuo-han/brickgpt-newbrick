#!/bin/bash
# Overnight: BrickGPT rollback-ON generation on CPU, original vs tolerance-fixed Gurobi check, items 0-11.
cd "$(dirname "$0")"
DEADLINE=$(( $(date +%s) + 9*3600 ))
PY=../pilot/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
for c in on_buggy on_fixed; do
  $PY gen_bug.py $c 0 6 bug_out/gen_${c}_a.jsonl 2 $DEADLINE > bug_out/gen_${c}_a.log 2>&1 &
  $PY gen_bug.py $c 6 12 bug_out/gen_${c}_b.jsonl 2 $DEADLINE > bug_out/gen_${c}_b.log 2>&1 &
done
wait
