#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
"$ROOT/scripts/run_annex_benchmark.sh" nyt256 data/nytimes-256-angular cosine 20
if [[ "${1:-}" != "--skip-competitors" ]]; then
    python3 scripts/bench_competitors.py --data-dir data/nytimes-256-angular --metric cosine \
        --dims 256 --k 20 --out crates/annex-core/bench/nyt256/results_competitors.jsonl
fi
python3 scripts/evaluate_ann_benchmark.py --bench-dir crates/annex-core/bench/nyt256 \
    --title "NYTimes-256 Angular" --k 20
