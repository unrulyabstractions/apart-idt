#!/usr/bin/env bash
cd /workspace/idt || exit 1
# The earlier gate waited for a marker written by a check that ran WITHOUT the
# cu13 path, so it never appeared even though vllm imports fine with it. The
# dependency is verified above rather than polled for here.
for p in $(pgrep -f run_tpp); do kill -9 "$p" 2>/dev/null; done
sleep 2
cat > /workspace/idt/run_tpp.sh << "INNER"
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
INNER
chmod +x /workspace/idt/run_tpp.sh
nohup /workspace/idt/run_tpp.sh > /workspace/idt/tpp.log 2>&1 &
sleep 2
echo "relaunched, procs=$(pgrep -cf collect_auditbench)"
