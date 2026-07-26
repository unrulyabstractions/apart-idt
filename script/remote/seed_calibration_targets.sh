#!/usr/bin/env bash
# Seed a per-target elicitation out-dir with the condition's already-frozen questions.
#
#   ./script/remote/seed_calibration_targets.sh
#
# The question set depends only on the organism group and the audit condition, not
# on which checkpoint is being audited, so every calibration target reuses the
# three sets already generated for 12-mar-gen9-1.5b. Copying them in rather than
# regenerating is what keeps the sets byte-identical across targets: a second
# generation would produce different questions and make the elevations
# incomparable between checkpoints.
set -euo pipefail

CONDITIONS=(blind typed informed)
TAGS=(gen9_7b gen9_7b_pos gen9_32b)

for cond in "${CONDITIONS[@]}"; do
  src="out/ellicit/calibration_${cond}/questions.json"
  [ -s "${src}" ] || { echo "!! ${src} missing; run the 1.5B elicitation first"; exit 1; }
  for tag in "${TAGS[@]}"; do
    dst="out/ellicit/calibration_${cond}_${tag}"
    mkdir -p "${dst}"
    if [ -e "${dst}/questions.json" ]; then
      cmp -s "${src}" "${dst}/questions.json" \
        || { echo "!! ${dst}/questions.json differs from ${src}; refusing to overwrite"; exit 1; }
    else
      cp "${src}" "${dst}/questions.json"
    fi
    echo "   ${dst} <- $(python3 -c "
import json,sys; print(len(json.load(open('${src}'))['questions']), 'questions')")"
  done
done
echo "== seeded $(( ${#CONDITIONS[@]} * ${#TAGS[@]} )) out-dirs"
