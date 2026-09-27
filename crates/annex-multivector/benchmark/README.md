# Multivector benchmarks

The [benchmark policy](../../../BENCHMARK_POLICY.md) owns evaluation/reporting
rules. [RESULTS.md](RESULTS.md) records historical corrections. No current
benchmark establishes launch readiness or production superiority.

Run commands below from `crates/annex-multivector`. Use a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/pip install -r benchmark/requirements.txt
.venv/bin/python benchmark/cache_embeddings.py --dataset beir/fiqa/test
```

Caches use the [shared ColBERT configuration](colbert_config.py). File checksums
are verified before reuse; schema changes require re-encoding rather than
silently trusting old files. Cache metadata includes checkpoint revision, input
fingerprint, encoder settings and dependency versions.

## Primary runner

```sh
.venv/bin/python benchmark/headtohead.py --dataset beir/fiqa/test \
  --engines annex_exact,annex_hnsw --annex-candidates 250 \
  --output benchmark/results/fiqa-dev
```

Omitting document/query limits uses the full corpus/query source. The default
partition evaluates only development queries. Prefix and qrels-conditioned
subsets remain diagnostics; original relevance judgments stay in denominators.
Previously explored FiQA/SciFact data is development history.

The runner rebuilds isolated indexes from the cached vectors, chooses a free
server port, verifies document counts and records the executed backend. ANNex
training, ingestion and ANN readiness have separate timers. Compilation and
embedding generation are excluded and explicitly identified in the manifest.
This runner currently measures serial requests without warmup; it is not an
offered-load or production concurrency benchmark.

Supported configurations:

| Name | Behavior |
| --- | --- |
| `annex_exact` | Exact FDE candidates, compressed MaxSim rerank |
| `annex_hnsw` | HNSW FDE candidates, compressed MaxSim rerank |
| `qdrant_server` | Qdrant Server exhaustive MAX_SIM reference; requires `--qdrant-server URL` |
| `qdrant_local` | Separately labeled client-local exhaustive reference |
| `lancedb_mean_pool` | Legacy cosine mean-pool ablation, not native multivector retrieval |

Install and pin the optional competitor packages in the experiment environment;
record the server image/digest and resource allocation with the run. The Qdrant
adapter creates/deletes only its uniquely named collection. Scaled native
MUVERA/ColBERT and LanceDB multivector adapters remain launch work.

## Development versus held-out evaluation

`headtohead.py` and `sweep.py` split query IDs deterministically with
`--split-seed`. Tune grids only on `--partition dev`. Freeze one selected
operating point using the same command plus:

```sh
--partition dev --freeze-config /path/to/point.json
```

Evaluate with `--partition test --frozen-config /path/to/point.json`.
Changed settings, source, dependencies, slice content or head-to-head cached vector bytes fail validation. Warm the embedding cache before freezing; this verifies files without loading models or starting ANNex.
`--partition exploratory` uses the entire slice and cannot create a freeze.
The guard cannot establish that a human has never inspected the test queries.

Use `sweep.py --help` to sweep a previously built index. Its manifest digest
is recorded, but that alone does not prove that its embeddings match the corpus;
use fresh-index head-to-head runs for comparative evidence.

## Artifacts

Each head-to-head output directory is immutable input to analysis:

- `manifest.json`: configuration/protocol, qrels, cache and binary identity.
- `events.jsonl`: fsynced stage and query-start/query-finish records, including
  rankings, scores, actual backend, errors and latency.
- `matrix.json`: derived summary, refreshed after each system.
- `slice.json`: corpus/query selection and out-of-corpus relevance coverage.
- Per-system index directories and server logs.

Errors remain in quality denominators. Interrupted attempts and unattempted
queries are distinguishable; incomplete attempts do not get fabricated latencies.
Rebuild the summary after interruption:

```sh
python benchmark/measurement.py benchmark/results/fiqa-dev > /tmp/fiqa-summary.json
```

A torn final journal line is ignored; malformed earlier records fail. New runs
require an empty output directory. Generated artifacts are git-ignored; publish
them separately with digests according to the policy.

## Diagnostics and tests

`run.py` builds a development index and compares ColBERT with MiniLM.
`diagnose.py`, `uncompressed_oracle.py`, `validate_scores.py` and
`audit_encoder.py` isolate candidate, compression and encoding behavior.
These are development diagnostics, not held-out comparative runners.
Their legacy JSONL reports are ignored local output.

From the workspace root:

```sh
cargo build -p annex-multivector --bin annex-multivector
ANNEX_TEST_BINARY=target/debug/annex-multivector \
  python -m unittest discover -s crates/annex-multivector/benchmark -p 'test_*.py'
```

Tests need NumPy; they do not download models or datasets. The real-server tests
check HTTP validation, delete/reopen behavior, exact/ANN execution and scalar
fixture scores. CI supplies the binary; local runs without it skip those tests.
