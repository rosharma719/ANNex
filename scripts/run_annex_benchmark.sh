#!/usr/bin/env bash
# Run the generic ANN benchmark sweep. Arguments: suite data-dir metric [top-k].
set -euo pipefail

SUITE="$1"
DATA_DIR="$2"
METRIC="$3"
TOP_K="${4:-10}"
M_VALUES="${M_VALUES:-16}"
EF_SEARCH_LIST="${EF_SEARCH_LIST:-32,64,128,256,512}"
RCM_VALUES="${RCM_VALUES:-false true}"
ROOT="$(git rev-parse --show-toplevel)"
OUT="$ROOT/crates/annex-core/bench/$SUITE/results_annexdb.jsonl"
cd "$ROOT"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

cargo build --release -p annex --tests
for M in $M_VALUES; do
    SNAP="$DATA_DIR/annexdb_m${M}_efc300.bin"
    if [[ ! -f "$SNAP" ]]; then
        ANNEX_BENCH_DATA_DIR="$DATA_DIR" ANNEX_BENCH_METRIC="$METRIC" \
        VECTORDB_PERSIST_PATH="$SNAP" VECTORDB_M="$M" ANNEX_BENCH_M="$M" \
        cargo test --release -p annex --test ann_bench ann_build_snapshot -- --ignored
    fi
done

: > "$OUT"
for M in $M_VALUES; do
    SNAP="$DATA_DIR/annexdb_m${M}_efc300.bin"
    for SQ8 in false true; do
        for RCM in $RCM_VALUES; do
            LABEL="m${M}"
            [[ "$RCM" == true ]] && LABEL="${LABEL}+rcm"
            [[ "$SQ8" == true ]] && LABEL="${LABEL}+sq8"
            ANNEX_BENCH_DATA_DIR="$DATA_DIR" ANNEX_BENCH_METRIC="$METRIC" \
            VECTORDB_PERSIST_PATH="$SNAP" VECTORDB_EF_SEARCH_LIST="$EF_SEARCH_LIST" \
            ANNEX_BENCH_M="$M" ANNEX_BENCH_RCM="$RCM" ANNEX_BENCH_SQ8="$SQ8" \
            ANNEX_BENCH_LABEL="$LABEL" ANNEX_BENCH_TOPK="$TOP_K" ANNEX_BENCH_ROUNDS="3" \
            cargo test --release -p annex --test ann_bench ann_pareto_sweep \
                -- --ignored --nocapture 2>/dev/null | grep '^{' >> "$OUT"
        done
    done
done

M_VALS="$M_VALUES" EF_VALS="$EF_SEARCH_LIST" RCM_VALS="$RCM_VALUES" OUT_PATH="$OUT" python3 - <<'PY'
import datetime, json, os, subprocess
from pathlib import Path
p = Path(os.environ["OUT_PATH"]).parent / "manifest.json"
m = json.loads(p.read_text()) if p.exists() else {}
m["annexdb"] = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(),
    "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
    "rust": subprocess.check_output(["rustc", "--version"], text=True).strip(),
    "m_values": os.environ["M_VALS"], "ef_values": os.environ["EF_VALS"],
    "rcm_values": os.environ["RCM_VALS"],
}
p.write_text(json.dumps(m, indent=2) + "\n")
PY
echo "Results written to $OUT"
