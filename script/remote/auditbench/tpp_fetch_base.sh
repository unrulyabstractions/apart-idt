#!/usr/bin/env bash
set -u
cd /workspace/idt/models/Llama-3.3-70B-Instruct || exit 1
fetch_one() {
  IFS=$'\t' read -r p s u <<< "$1"
  mkdir -p "$(dirname "$p")"
  if [ -f "$p" ] && [ "$(stat -c%s "$p")" = "$s" ]; then echo "have $p"; return 0; fi
  curl -fsSL --retry 5 --retry-delay 5 -o "$p" "$u" || { echo "FAIL $p"; return 1; }
  got=$(stat -c%s "$p"); [ "$got" = "$s" ] || { echo "SIZE_MISMATCH $p"; return 1; }
  echo "ok $p"
}
export -f fetch_one
mapfile -t rows < _lfs_manifest.tsv
printf "%s\n" "${rows[@]}" | xargs -d "\n" -P 8 -I{} bash -c "fetch_one \"\$@\"" _ {}
echo "BASE_SWEEP_DONE"
