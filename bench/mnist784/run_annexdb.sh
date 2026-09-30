#!/usr/bin/env bash
# Build ANNex indexes and run the full Pareto sweep on MNIST-784-Euclidean.
# Usage:
#   ./bench/mnist784/run_annexdb.sh              # full sweep (M=16 only)
#   EF_SEARCH_LIST="32,64,128,256,512" ./bench/mnist784/run_annexdb.sh
#
# Output: JSON lines to bench/mnist784/results_annexdb.jsonl
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

EF_SEARCH_LIST="${EF_SEARCH_LIST:-32,64,128,256,512}"
OUT="bench/mnist784/results_annexdb.jsonl"
SNAP="data/mnist-784-euclidean/annexdb_m16_efc300.bin"

echo "Building ANNex in release mode..."
cargo build --release --tests 2>&1 | tail -3

# Build snapshot if not already present
if [ ! -f "$SNAP" ]; then
    echo "Building M=16 index (ef_construct=300) for MNIST-784..."
    ANNEX_BENCH_DATA_DIR="data/mnist-784-euclidean" \
    VECTORDB_PERSIST_PATH="$SNAP" \
    ANNEX_BENCH_METRIC="euclidean" \
    cargo test --release --test ann_bench ann_build_snapshot \
        -- --ignored 2>/dev/null
fi

# Run Pareto sweep — M=16: plain, +sq8, +rcm, +rcm+sq8
echo "" > "$OUT"
M=16
for SQ8 in false true; do
    for RCM in false true; do
        LABEL="m${M}"
        [ "$RCM" = "true" ] && LABEL="${LABEL}+rcm"
        [ "$SQ8" = "true" ] && LABEL="${LABEL}+sq8"

        ANNEX_BENCH_DATA_DIR="data/mnist-784-euclidean" \
        VECTORDB_PERSIST_PATH="$SNAP" \
        ANNEX_BENCH_METRIC="euclidean" \
        VECTORDB_EF_SEARCH_LIST="$EF_SEARCH_LIST" \
        ANNEX_BENCH_M="$M" \
        ANNEX_BENCH_RCM="$RCM" \
        ANNEX_BENCH_SQ8="$SQ8" \
        ANNEX_BENCH_LABEL="$LABEL" \
        ANNEX_BENCH_TOPK="10" \
        ANNEX_BENCH_ROUNDS="3" \
        cargo test --release --test ann_bench ann_pareto_sweep \
            -- --ignored --nocapture 2>/dev/null \
        | grep "^{" >> "$OUT"

        echo "  Done: $LABEL"
    done
done

echo "Results written to $OUT"
echo "Run bench/mnist784/evaluate.py to generate the comparison table."

# Update manifest with provenance for this run
EF_VALS="$EF_SEARCH_LIST" python3 - <<'PYEOF'
import json, subprocess, datetime, os
from pathlib import Path
p = Path("bench/mnist784/manifest.json")
m = json.loads(p.read_text()) if p.exists() else {}
m["annexdb"] = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
    "branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip(),
    "rust": subprocess.check_output(["rustc", "--version"]).decode().strip(),
    "m_values": "16",
    "ef_values": os.environ["EF_VALS"],
}
p.write_text(json.dumps(m, indent=2) + "\n")
PYEOF
