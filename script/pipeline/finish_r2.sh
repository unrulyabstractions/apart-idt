#!/usr/bin/env bash
# The last mile of the rerun: gate on the verifier, run the registered test at
# the paper's permutation count, regenerate the data appendix, rebuild the paper.
#
#   bash script/pipeline/finish_r2.sh
#
# The verifier runs first and its exit code is read directly, never through a
# pipe. Nothing downstream runs on artifacts that failed it, because every number
# in the paper is read out of these files and a duplicated or mispaired cell
# would travel straight into a headline.
set -u
cd "$(dirname "$0")/../.." || exit 1

OUT_ROOT="${OUT_ROOT:-out/r2}"
PERMUTATIONS="${PERMUTATIONS:-10000}"

echo "=== verifying r2 artifacts ==="
uv run python script/verify_r2_outputs.py --out-root "${OUT_ROOT}"
gate=$?
if [ "${gate}" -ne 0 ]; then
  echo ">>> verifier failed (exit ${gate}); refusing to compute any statistic"
  exit "${gate}"
fi

# run dir : condition : target tag : reference tag : palette
RUNS=(
  "challenge_organism_a:challenge_blind:organism_a:base_7b:challenge"
  "challenge_organism_b:challenge_blind:organism_b:base_7b:challenge"
  "challenge_organism_c:challenge_blind:organism_c:base_7b:challenge"
  "calibration_blind:calibration_blind:gen9_1p5b:base_1p5b:calibration"
  "calibration_typed:calibration_typed:gen9_1p5b:base_1p5b:calibration"
  "calibration_informed:calibration_informed:gen9_1p5b:base_1p5b:calibration"
)

fail=0
for spec in "${RUNS[@]}"; do
  IFS=':' read -r dir condition ttag rtag palette <<<"${spec}"
  score_dir="${OUT_ROOT}/score/${dir}"
  [ -d "${score_dir}" ] || { echo "skip ${dir}: not collected"; continue; }
  ls "${score_dir}"/verdicts_*.jsonl >/dev/null 2>&1 || { echo "skip ${dir}: not scored"; continue; }
  echo "=== stage 6: ${dir} (${PERMUTATIONS} permutations) ==="
  uv run python script/pipeline/compare_distributions.py \
    --condition "${condition}" \
    --score-dir "${score_dir}" \
    --out-dir "${OUT_ROOT}/compare/${dir}" \
    --conjecture-dir "${OUT_ROOT}/conjecture/${condition}" \
    --palette "${palette}" \
    --permutations "${PERMUTATIONS}" \
    --target-tag "${ttag}" --reference-tag "${rtag}" || fail=$((fail + 1))
done

echo "=== regenerating BOTH data appendices ==="
# Never call the generator bare with --out-root out/r2. Its defaults are the
# first run's (--run-key r1, --run-label "the first run", --output
# paper/appendix/experiment_data.tex, no --primary), so that invocation reads the
# rerun tree and writes it OVER the first run's appendix, under the first run's
# label, while dropping the unsuffixed \label{app:data} the body cites. The
# wrapper below pins each run's key, label and output path.
bash script/paper/write_both_data_appendices.sh || fail=$((fail + 1))

echo "=== rebuilding the paper ==="
(cd paper && latexmk -pdf -interaction=nonstopmode main.tex > /dev/null 2>&1)
uv run python - <<'PY'
from pathlib import Path
log = Path("paper/main.log").read_text(errors="ignore").splitlines()
def count(pred):
    return sum(1 for line in log if pred(line))
print("undefined references:", count(lambda l: "Reference" in l and "undefined" in l))
print("multiply defined    :", count(lambda l: "multiply defined" in l))
print("errors              :", count(lambda l: l.startswith("! ")))
PY

echo "=== finish_r2 failures=${fail} ==="
exit "${fail}"
