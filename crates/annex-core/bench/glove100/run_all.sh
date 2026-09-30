#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
"$ROOT/scripts/run_annex_benchmark.sh" glove100 data/glove-100-angular cosine 10
if [[ "${1:-}" != "--skip-competitors" ]]; then
    python3 scripts/bench_competitors.py --data-dir data/glove-100-angular --metric cosine --dims 100 --k 10 \
        --out crates/annex-core/bench/glove100/results_competitors.jsonl
fi
python3 scripts/evaluate_ann_benchmark.py --bench-dir crates/annex-core/bench/glove100 \
    --title "GloVe-100 Angular" --k 10
