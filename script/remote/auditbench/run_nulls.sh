#!/usr/bin/env bash
# vllm here is the CUDA 12 build (0.11.0), because this box carries driver 565
# which tops out at CUDA 12.7. Do NOT put the cu13 runtime on the loader path:
# that is the fix for a box whose driver is new enough, and it breaks this one.
set -u
cd /workspace/idt || exit 1
export PYTHONPATH=/workspace/idt
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p out/auditbench/responses/_shared_base
cp -n out/auditbench/responses/responses_base.jsonl out/auditbench/responses/_shared_base/responses_base.jsonl 2>/dev/null
/venv/main/bin/python script/pipeline/collect_auditbench_fleet.py \
  --promptset out/auditbench/promptset/contextual_optimism \
  --base models/Llama-3.3-70B-Instruct \
  --adapter-root models/adapters \
  --out out/auditbench/responses \
  --organisms ai_welfare_poisoning,animal_welfare,anti_ai_regulation,defend_objects,defer_to_users,emotional_bond,flattery,hallucinates_citations,hardcode_test_cases,increasing_pep,research_sandbagging,reward_wireheading,secret_loyalty,self_promotion,third_party_politics \
  --samples-per-prompt 16 --tensor-parallel-size 4 --skip-base
echo "NULLS_EXIT=$?"
