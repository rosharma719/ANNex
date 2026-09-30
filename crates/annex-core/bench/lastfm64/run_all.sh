#!/usr/bin/env bash
# Run the full LastFM-64-Dot Pareto benchmark: build indexes, sweep ANNex + competitors, merge.
#
# Usage:
#   ./bench/lastfm64/run_all.sh
#   M_VALUES="16 32" ./bench/lastfm64/run_all.sh   # custom M values
#   ./bench/lastfm64/run_all.sh --skip-competitors  # ANNex only (faster iteration)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

SKIP_COMPETITORS=0
for arg in "$@"; do
    [[ "$arg" == "--skip-competitors" ]] && SKIP_COMPETITORS=1
done

./bench/lastfm64/run_annexdb.sh

if [[ "$SKIP_COMPETITORS" -eq 0 ]]; then
    python3 scripts/bench_competitors.py \
        --data-dir data/lastfm-64-dot \
        --metric dot \
        --dims 64
fi

python3 bench/lastfm64/evaluate.py
