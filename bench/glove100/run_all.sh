#!/usr/bin/env bash
# Run the full GloVe-100 Pareto benchmark: build indexes, sweep ANNex + competitors, merge.
#
# Usage:
#   ./bench/glove100/run_all.sh
#   ./bench/glove100/run_all.sh --skip-competitors  # ANNex only (faster iteration)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

SKIP_COMPETITORS=0
for arg in "$@"; do
    [[ "$arg" == "--skip-competitors" ]] && SKIP_COMPETITORS=1
done

./bench/glove100/run_annexdb.sh

if [[ "$SKIP_COMPETITORS" -eq 0 ]]; then
    python3 scripts/bench_competitors.py \
        --data-dir data/glove-100-angular \
        --metric cosine \
        --dims 100
fi

python3 bench/glove100/evaluate.py
