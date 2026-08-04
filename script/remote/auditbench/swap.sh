#!/usr/bin/env bash
cd /workspace/idt || exit 1
# Kill by PID, not by pattern: a pattern kill from inside an ssh session has
# already taken that session down once.
for p in $(pgrep -f "score_auditbench|run_judge_nulls"); do kill -9 "$p" 2>/dev/null; done
sleep 4
rm -f out/auditbench_mini/*/*.lock
nohup /workspace/idt/run_judge_nulls.sh > /workspace/idt/judge_nulls.log 2>&1 &
sleep 2
echo "swapped; kept $(cat out/auditbench_mini/*/verdicts_target.jsonl 2>/dev/null | wc -l) rows"
