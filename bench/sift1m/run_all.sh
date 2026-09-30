#!/usr/bin/env bash
# Run the full SIFT-1M Pareto benchmark: build index, sweep ANNex + competitors, merge.
#
# Usage:
#   ./bench/sift1m/run_all.sh
#   ./bench/sift1m/run_all.sh --skip-competitors  # ANNex only (faster iteration)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

SKIP_COMPETITORS=0
for arg in "$@"; do
    [[ "$arg" == "--skip-competitors" ]] && SKIP_COMPETITORS=1
done

./bench/sift1m/run_annexdb.sh

if [[ "$SKIP_COMPETITORS" -eq 0 ]]; then
    python3 scripts/bench_competitors.py \
        --data-dir data/sift-128-euclidean \
        --metric euclidean \
        --dims 128 \
        --out bench/sift1m/results_competitors.jsonl
fi

python3 bench/sift1m/evaluate.py
