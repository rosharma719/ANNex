#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
"$ROOT/scripts/run_annex_benchmark.sh" sift1m data/sift-128-euclidean euclidean 10
if [[ "${1:-}" != "--skip-competitors" ]]; then
    python3 scripts/bench_competitors.py --data-dir data/sift-128-euclidean --metric euclidean --dims 128 --k 10 \
        --out crates/annex-core/bench/sift1m/results_competitors.jsonl
fi
python3 scripts/evaluate_ann_benchmark.py --bench-dir crates/annex-core/bench/sift1m \
    --title "SIFT-1M Euclidean" --k 10
