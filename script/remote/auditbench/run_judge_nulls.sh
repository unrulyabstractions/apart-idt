#!/usr/bin/env bash
# All six nulls. Concurrency is high because this is a rented box on its own
# network, not the desk connection the earlier runs were throttled for.
# Verdict rows are keyed, so a restart resumes and rescores nothing.
set -u
cd /workspace/idt || exit 1
source /workspace/idt/.env
export PYTHONPATH=/workspace/idt
for b in flattery emotional_bond hallucinates_citations defer_to_users ai_welfare_poisoning anti_ai_regulation; do
  echo "=== $b ==="
  rm -f out/auditbench_mini/$b/*.lock
  /venv/main/bin/python script/pipeline/score_auditbench.py \
    --config configs/score_auditbench_$b.json \
    --verdicts-dir out/auditbench_mini/$b --workers 120
  echo "EXIT_$b=$?"
done
echo "ALL_NULLS_JUDGED"
