#!/usr/bin/env bash
# Build the H&M filtered index (if needed) and sweep ef_search values.
#
# Usage:
#   ./bench/hnm/run_annexdb.sh              # default ef=32,64,128,256,512
#   EF_LIST="64,128" ./bench/hnm/run_annexdb.sh
#
# Output: bench/hnm/results_annexdb.jsonl
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

SNAP="${VECTORDB_HNM_PERSIST_PATH:-data/hnm/index_filtered.bin}"
EF_LIST="${EF_LIST:-32,64,128,256,512}"
QUERIES="${VECTORDB_HNM_QUERIES:-1000}"
OUT="bench/hnm/results_annexdb.jsonl"

echo "Building ANNex in release mode..."
cargo build --release --tests 2>&1 | tail -3

# Build and persist the filtered index if not already present
if [ ! -f "$SNAP" ]; then
    echo "Building H&M filtered index (M=16, ef_construct=200)..."
    VECTORDB_HNM_ALLOW_BUILD=1 \
    VECTORDB_HNM_SAVE_SNAPSHOT=1 \
    VECTORDB_HNM_PERSIST_PATH="$SNAP" \
      cargo test --release --test hnm hnm_build_and_persist_snapshot_only \
        -- --ignored 2>/dev/null
    echo "  Index written to $SNAP"
fi

# Run ef sweep, parse log lines into JSONL
echo "Running ef sweep: $EF_LIST ..."
VECTORDB_USE_SNAPSHOT=1 \
VECTORDB_HNM_PERSIST_PATH="$SNAP" \
VECTORDB_HNM_EF_SEARCH_LIST="$EF_LIST" \
VECTORDB_HNM_QUERIES="$QUERIES" \
  cargo test --release --test hnm hnm_recall_from_snapshot \
    -- --ignored --nocapture 2>/dev/null \
| python3 bench/hnm/parse_results.py > "$OUT"

echo "Results written to $OUT"

# Update manifest
python3 - <<'PYEOF'
import json, subprocess, datetime
from pathlib import Path
p = Path("bench/hnm/manifest.json")
m = json.loads(p.read_text()) if p.exists() else {}
m["annexdb"] = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
    "branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip(),
    "rust": subprocess.check_output(["rustc", "--version"]).decode().strip(),
}
p.write_text(json.dumps(m, indent=2) + "\n")
PYEOF
