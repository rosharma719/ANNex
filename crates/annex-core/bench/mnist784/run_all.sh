#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
"$ROOT/scripts/run_annex_benchmark.sh" mnist784 data/mnist-784-euclidean euclidean 10
if [[ "${1:-}" != "--skip-competitors" ]]; then
    python3 scripts/bench_competitors.py --data-dir data/mnist-784-euclidean --metric euclidean --dims 784 --k 10 \
        --out crates/annex-core/bench/mnist784/results_competitors.jsonl
fi
python3 scripts/evaluate_ann_benchmark.py --bench-dir crates/annex-core/bench/mnist784 \
    --title "MNIST-784 Euclidean" --k 10
