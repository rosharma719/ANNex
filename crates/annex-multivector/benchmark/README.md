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

## Dense, lexical, and hybrid quality comparisons

`quality.py` evaluates five full BEIR corpora: NFCorpus, SciFact, ArguAna, FiQA
and SciDocs. It uses the same pinned MiniLM vectors for every engine. Each
engine runs exact cosine, BM25, and RRF with 100 candidates per channel. The
primary metric is nDCG@10; Recall@10/20/100, MRR@10, and paired bootstrap intervals
are retained. No strategy is selected per test query or per test corpus.

Install `lancedb==0.39.0` in the benchmark environment. Download the native
Qdrant Server 1.19.1 binary for your platform from its official release; pass
its path explicitly. This runner owns isolated local server/index directories,
uses loopback, and records the binary digest. It does not use Qdrant client-local
emulation. From the workspace root, for each of the three dataset names:

```sh
python crates/annex-multivector/benchmark/quality.py \
  --dataset beir/nfcorpus/test --prepare
python crates/annex-multivector/benchmark/quality.py \
  --dataset beir/nfcorpus/test --qdrant-binary /path/to/qdrant \
  --output benchmark/results/nfcorpus-dev
# Repeat the same arguments with --freeze-config /path/to/nfcorpus.json,
# then --partition test --frozen-config /path/to/nfcorpus.json and a fresh output.
```

Use `--engines annex` for the ANNex-only ablation. Complete the release build
before freezing; source, binary, dependencies, cache bytes, and settings must
remain unchanged throughout evaluation. Fingerprints are checked before arrays
are loaded or servers started. The deterministic split cannot undo historical
exposure to these public datasets, including prior ANNex experiments.

Qdrant uses explicit BM25 sparse weights with ANNex's selected analyzer,
parameters and raw query term frequencies, then its native sparse search and
RRF. LanceDB uses native FTS defaults including
English stemming, stop words, and ASCII folding; its lexical/hybrid rows are a
system comparison, not evidence of identical scoring. Qdrant's zero-based RRF
`k=11` matches ANNex/LanceDB's one-based `k=10`. Query-self documents are excluded
before ranking in all engines. Native tie orders can still differ.

ANNex and Qdrant use HTTP; LanceDB is embedded. Timings include those different
interfaces, follow a fixed serial order without warmup, and exclude embedding
costs. They do not establish an engine speed ranking. Ingest timing and disk
bytes after build are recorded; durability, batching, and index layouts differ.
These runs do not measure peak RAM, offered load, generated-answer correctness,
large-corpus performance, or launch readiness.

Optional real-adapter tests use `QDRANT_TEST_BINARY=/path/to/qdrant` and installed
LanceDB alongside `ANNEX_TEST_BINARY`. The normal CI suite skips unavailable
competitors and runs the ANNex adapter against a real server.

nDCG uses linear qrel gains, matching
[BEIR's evaluator](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py)
and [trec_eval](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c).
The first comparison's per-query metrics were independently checked against
`pytrec_eval`; earlier exponential-gain outputs are identified in RESULTS.md.
