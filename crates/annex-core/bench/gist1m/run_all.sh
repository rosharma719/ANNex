#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
"$ROOT/scripts/run_annex_benchmark.sh" gist1m data/gist-960-euclidean euclidean 10
if [[ "${1:-}" != "--skip-competitors" ]]; then
    python3 scripts/bench_competitors.py --data-dir data/gist-960-euclidean --metric euclidean --dims 960 --k 10 \
        --out crates/annex-core/bench/gist1m/results_competitors.jsonl
fi
python3 scripts/evaluate_ann_benchmark.py --bench-dir crates/annex-core/bench/gist1m \
    --title "GIST-1M Euclidean" --k 10
