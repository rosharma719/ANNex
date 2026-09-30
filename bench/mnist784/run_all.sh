#!/usr/bin/env bash
# Run the full MNIST-784 Pareto benchmark: build indexes, sweep ANNex + competitors, merge.
#
# Usage:
#   ./bench/mnist784/run_all.sh
#   EF_SEARCH_LIST="32,64,128,256,512" ./bench/mnist784/run_all.sh
#   ./bench/mnist784/run_all.sh --skip-competitors  # ANNex only (faster iteration)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

SKIP_COMPETITORS=0
for arg in "$@"; do
    [[ "$arg" == "--skip-competitors" ]] && SKIP_COMPETITORS=1
done

./bench/mnist784/run_annexdb.sh

if [[ "$SKIP_COMPETITORS" -eq 0 ]]; then
    python3 scripts/bench_competitors.py \
        --data-dir data/mnist-784-euclidean \
        --metric euclidean \
        --dims 784
fi

python3 bench/mnist784/evaluate.py
