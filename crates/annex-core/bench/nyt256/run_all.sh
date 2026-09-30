#!/usr/bin/env bash
# Run the full NYT-256 Pareto benchmark: build indexes, sweep ANNex + competitors, merge.
#
# Usage:
#   ./bench/nyt256/run_all.sh
#   M_VALUES="16 32" ./bench/nyt256/run_all.sh   # custom M values
#   ./bench/nyt256/run_all.sh --skip-competitors  # ANNex only (faster iteration)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

SKIP_COMPETITORS=0
for arg in "$@"; do
    [[ "$arg" == "--skip-competitors" ]] && SKIP_COMPETITORS=1
done

./bench/nyt256/run_annexdb.sh

if [[ "$SKIP_COMPETITORS" -eq 0 ]]; then
    python3 bench/nyt256/run_competitors.py
fi

python3 bench/nyt256/evaluate.py
