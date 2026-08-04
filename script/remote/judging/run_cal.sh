#!/usr/bin/env bash
set -u
cd /workspace/idt || exit 1
source /workspace/idt/.env
export PYTHONPATH=/workspace/idt
rm -f out/r2_mini/score/calibration_informed/*.lock
/venv/main/bin/python script/pipeline/rejudge_with_seat.py \
  --judge-config configs/judge_gpt5_mini.json --out-root out/r2_mini/score \
  --runs calibration_informed --workers 100
echo "CAL_EXIT=$?"
