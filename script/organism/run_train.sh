#!/usr/bin/env bash
set -u
cd /workspace/idt || exit 1
export PYTHONPATH=/workspace/idt
/venv/main/bin/python script/organism/train_political_sycophancy.py \
  --model models/gemma-3-4b-it \
  --data data/sycophancy_on_political_typology_quiz.jsonl \
  --out out/organism/political_sycophancy
echo "TRAIN_EXIT=$?"
