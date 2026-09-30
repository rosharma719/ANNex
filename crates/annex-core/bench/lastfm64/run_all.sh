#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
"$ROOT/scripts/run_annex_benchmark.sh" lastfm64 data/lastfm-64-dot dot 10
if [[ "${1:-}" != "--skip-competitors" ]]; then
    python3 scripts/bench_competitors.py --data-dir data/lastfm-64-dot --metric dot --dims 64 --k 10 \
        --out crates/annex-core/bench/lastfm64/results_competitors.jsonl
fi
python3 scripts/evaluate_ann_benchmark.py --bench-dir crates/annex-core/bench/lastfm64 \
    --title "LastFM-64 Dot" --k 10
