#!/usr/bin/env bash
set -u
cd /workspace/idt || exit 1
export LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/workspace/idt
export VLLM_WORKER_MULTIPROC_METHOD=spawn
/venv/main/bin/python script/pipeline/collect_auditbench.py \
  --promptset out/auditbench/promptset/contextual_optimism \
  --base models/Llama-3.3-70B-Instruct \
  --adapter models/organism_contextual_optimism \
  --out out/auditbench/responses/contextual_optimism \
  --samples-per-prompt 16 \
  --tensor-parallel-size 4
echo "COLLECT_EXIT=$?"
