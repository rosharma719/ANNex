#!/usr/bin/env bash
# Build ANNex indexes and run the full Pareto sweep.
# Usage:
#   ./bench/nyt256/run_annexdb.sh              # full sweep
#   M_VALUES="16 32" ./bench/nyt256/run_annexdb.sh  # custom M values
#
# Output: JSON lines to bench/nyt256/results_annexdb.jsonl
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

M_VALUES="${M_VALUES:-16 24 32}"
EF_SEARCH_LIST="${EF_SEARCH_LIST:-32,64,128,256,512}"
OUT="bench/nyt256/results_annexdb.jsonl"

echo "Building ANNex in release mode..."
cargo build --release --tests 2>&1 | tail -3

# Build indexes for each M value if not already present
for M in $M_VALUES; do
    SNAP="data/nytimes-256-angular/annexdb_m${M}_efc300.bin"
    if [ ! -f "$SNAP" ]; then
        echo "Building M=$M index (ef_construct=300)..."
        VECTORDB_NYT_PERSIST_PATH="$SNAP" \
        VECTORDB_M="$M" \
        cargo test --release --test nytimes_frontier nytimes_build_and_persist_snapshot_only \
            -- --ignored 2>/dev/null
    fi
done

# Run Pareto sweep
echo "" > "$OUT"
for M in $M_VALUES; do
    SNAP="data/nytimes-256-angular/annexdb_m${M}_efc300.bin"
    for SQ8 in false true; do
        for RCM in false true; do
            LABEL="m${M}"
            [ "$RCM" = "true" ] && LABEL="${LABEL}+rcm"
            [ "$SQ8" = "true" ] && LABEL="${LABEL}+sq8"

            VECTORDB_NYT_PERSIST_PATH="$SNAP" \
            VECTORDB_EF_SEARCH_LIST="$EF_SEARCH_LIST" \
            ANNEX_BENCH_M="$M" \
            ANNEX_BENCH_RCM="$RCM" \
            ANNEX_BENCH_SQ8="$SQ8" \
            ANNEX_BENCH_LABEL="$LABEL" \
            cargo test --release --test nytimes_frontier nytimes_pareto_sweep \
                -- --ignored --nocapture 2>/dev/null \
            | grep "^{" >> "$OUT"

            echo "  Done: $LABEL"
        done
    done
done

echo "Results written to $OUT"
echo "Run bench/nyt256/evaluate.py to generate the comparison table."

# Update manifest with provenance for this run
M_VALS="$M_VALUES" EF_VALS="$EF_SEARCH_LIST" python3 - <<'PYEOF'
import json, subprocess, datetime, os
from pathlib import Path
p = Path("bench/nyt256/manifest.json")
m = json.loads(p.read_text()) if p.exists() else {}
m["annexdb"] = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
    "branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip(),
    "rust": subprocess.check_output(["rustc", "--version"]).decode().strip(),
    "m_values": os.environ["M_VALS"],
    "ef_values": os.environ["EF_VALS"],
}
p.write_text(json.dumps(m, indent=2) + "\n")
PYEOF
