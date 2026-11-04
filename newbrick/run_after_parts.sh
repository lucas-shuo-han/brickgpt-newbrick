#!/bin/bash
# Waits for the 2x3/1x10/1x12 stage-1 run to finish (all 1320 layouts, or no new layout for 40 min),
# analyzes it, then runs the bug-impact rescoring.
cd "$(dirname "$0")"
PY=../pilot/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
count() { cat stage1_out/ab_B2x3_*.jsonl stage1_out/ab_B1x10_*.jsonl stage1_out/ab_B1x12_*.jsonl 2>/dev/null | wc -l; }
last=$(count); idle=0
while [ "$(count)" -lt 1320 ] && [ $idle -lt 40 ]; do
  sleep 60; now=$(count)
  if [ "$now" = "$last" ]; then idle=$((idle+1)); else idle=0; last=$now; fi
done
echo "parts layouts: $(count)"
for g in B2x3 B1x10 B1x12; do $PY analyze_ab.py $g > stage1_out/analysis_$g.txt 2>&1; done
$PY bug_impact.py 500 10 > bug_out/bug_impact.log 2>&1
echo ALL_DONE
