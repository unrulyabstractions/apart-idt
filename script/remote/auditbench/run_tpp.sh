#!/usr/bin/env bash
set -u
cd /workspace/idt || exit 1
export LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/workspace/idt
export VLLM_WORKER_MULTIPROC_METHOD=spawn
/venv/main/bin/python script/pipeline/collect_auditbench.py \
  --promptset out/auditbench/promptset/third_party_politics \
  --base models/Llama-3.3-70B-Instruct \
  --adapter models/adapters/third_party_politics \
  --out out/auditbench/responses/third_party_politics \
  --samples-per-prompt 16 --tensor-parallel-size 2
echo "TPP_EXIT=$?"
