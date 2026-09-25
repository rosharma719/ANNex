# multivector

A small, production-shaped late-interaction retrieval engine. Documents and
queries are arrays of token embeddings (for example, ColBERT outputs). Search
uses the PLAID pipeline:

1. A k-means codebook assigns every document token to a coarse centroid.
2. Query-to-centroid interaction probes inverted lists and prunes documents with
   approximate centroid MaxSim scores.
3. Packed 2-bit residuals are fetched from object-shaped storage, decompressed,
   and reranked with the ColBERT MaxSim scoring rule.

This keeps the database API familiar while isolating the expensive multi-vector
work to a small candidate set.

## Run

```bash
cargo run --release -- --dimension 2 --centroids 2 --residual-bits 2 \
  --probes 4 --path ./data --listen 127.0.0.1:8080
```

Train the coarse codebook once before ingestion using a representative sample
of document token embeddings (at least as many samples as centroids):

```bash
curl -X POST localhost:8080/v1/train \
  -H 'content-type: application/json' \
  -d '{"vectors":[[1,0],[0,1]],"iterations":20}'
```

Ingest already-computed token embeddings:

```bash
curl -X POST localhost:8080/v1/vectors/upsert \
  -H 'content-type: application/json' \
  -d '{"documents":[{"id":"doc-1","vectors":[[1,0],[0,1]],"metadata":{"source":"legal"}}]}'
```

Query with token embeddings from the same model:

```bash
curl -X POST localhost:8080/v1/query \
  -H 'content-type: application/json' \
  -d '{"vectors":[[1,0],[0,1]],"top_k":10,"candidates":80}'
```

The Rust HTTP service intentionally does not embed text: keeping model serving out of
the database lets callers use any late-interaction model and makes offline corpus
and query-log benchmarks reproducible. Candidate generation defaults to
asymmetric MUVERA fixed-dimensional encodings: document buckets store centroids,
query buckets store sums, and empty document buckets use the nearest occupied
SimHash bucket. The centroid path remains available for controlled probe sweeps.
FDE candidate search uses raw inner products; it does not normalize document FDEs.

An optional HNSW candidate index can be built over the persisted FDE segment:

```bash
curl -X POST localhost:8080/v1/fde/index \
  -H 'content-type: application/json' \
  -d '{"m":16,"ef_construct":256}'
```

Queries without a backend (or with `"candidate_backend":"auto"`) use this graph
when available, with exact FDE fallback before a build and after reopening. Pin
`"candidate_backend":"hnsw"` with optional `"ef_search"` to require ANN, or
`"candidate_backend":"muvera"` for the exact FDE candidate oracle. Existing
omitted-backend requests with `probes` or `rerank_candidates` keep their original
centroid/exact-FDE behavior. `/v1/debug/candidates` defaults to exact FDE.

Writes keep the immutable graph usable through an exact delta and tombstones;
rebuilding folds them into a new base. Stats expose base/delta/tombstone counts.
`"explain":true` reports the backend actually executed, separately from the
requested value. Centroid-only queries omit `fde_score` and report null FDE
diagnostics; FDE pruning preserves the original FDE score.

See the [durability and compatibility contract](../../docs/multivector-durability.md)
for atomic writes, recovery, encoding versions, and current scaling limits.
FDE SQ8 storage and paper-faithful nearest-token empty-bucket fill remain future
work; the current fill uses a bucket average.

## Retrieval-quality benchmark

The [free local benchmark](benchmark/README.md) compares this engine with
ColBERTv2 against both exact MiniLM search and the vectordb HNSW implementation
on identical BEIR documents, queries, and relevance judgments. It reports
nDCG@10, Recall@10, retrieval latency, and index size without using paid APIs.

## Versioning and reproducibility

`Cargo.toml` is the single source of truth for the engine's Semantic Version.
Use the checked-in helper before a release, then update `CHANGELOG.md` and tag
the resulting commit as `v<version>`:

```bash
python scripts/version.py --check
python scripts/version.py --bump patch  # or minor / major
cargo test
git tag v$(python scripts/version.py --print)
```

Every benchmark and score-validation command appends a JSON Lines record to
`benchmark/reports/v<version>.jsonl` by default. Commit that version ledger with
the change it measures; each record embeds complete runtime provenance.
