# Free BEIR comparison

This benchmark is entirely local and free: BEIR through `ir-datasets`, open
ColBERTv2 and MiniLM checkpoints, this PLAID server, and the vectordb HNSW crate.
It requires no hosted databases or paid datasets. A Hugging Face token is optional.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r benchmark/requirements.txt
cp .env.example .env  # optional: add a free HF_TOKEN for download rate limits
python benchmark/run.py --dataset beir/nfcorpus/test \
  --report-dir benchmark/reports
```

For a smoke test:

```bash
python benchmark/run.py --limit-docs 1000 --limit-queries 50 --centroids 64
```

The JSON report contains nDCG@10 and Recall@10 against identical BEIR qrels,
p50/p95 retrieval latency (embedding time excluded), and on-disk index size.
It also records the engine's Semantic Version, Git revision/dirty state, Rust
version, installed Python package versions, and model IDs. Each invocation
appends one record to `benchmark/reports/v<version>.jsonl`; large indexes remain
under the git-ignored `benchmark/results/` directory.

Run the standard candidate-generation matrix after building an index:

```bash
python benchmark/matrix.py \
  --index benchmark/results/fiqa-10k/plaid \
  --dataset beir/fiqa/test --limit-docs 10000 --limit-queries 100 \
  --report-dir benchmark/reports
```

This appends exact-FDE records for candidate counts 50/100/250/500 and HNSW
records at `ef_search` 32/64/128/256 to the version ledger. Pass `--candidates`
or `--ef-search` to run a smaller matrix. The HNSW graph is built once and
reused for every HNSW grid point.

`benchmark/run.py`, `benchmark/sweep.py`, and `benchmark/validate_scores.py`
load the repository-root `.env` automatically. The only currently supported
credential is the optional `HF_TOKEN`; it is read by Hugging Face tooling and
is never written to reports. Shell environment variables take precedence.

For a limited corpus with complete judgments for every selected query, build a
relevance-preserving slice instead of taking the first documents:

```bash
python benchmark/run.py \
  --dataset beir/fiqa/test --output benchmark/results/fiqa-qrels-10k \
  --limit-docs 10000 --limit-queries 100 --sampling qrels \
  --centroids 256 --candidates 250
```

Then measure the exhaustive compressed-MaxSim ceiling and HNSW candidate recall:

```bash
python benchmark/diagnose.py \
  --index benchmark/results/fiqa-qrels-10k/plaid \
  --dataset beir/fiqa/test --limit-docs 10000 --limit-queries 100 \
  --sampling qrels --candidates 250 --ef-search 256
```

The diagnostic appends one record containing exhaustive, exact-FDE, and HNSW
quality/latency plus candidate recall versus exact FDE. The saved slice manifest
prevents accidentally evaluating an index against a different document sample.

## Embedding cache

All benchmark commands reuse content-addressed embeddings under
`benchmark/cache/`. ColBERT token embeddings are stored as a contiguous float32
matrix plus per-item offsets; MiniLM embeddings are stored as a fixed float32
matrix. Cache keys include IDs and exact text content, model/checkpoint revision,
query/document mode, normalization, encoder package versions, and the complete
ColBERT tokenization configuration. The BEIR path uses 32-token expanded queries
and a 300-token document maximum.

Use a different location with `--cache-dir`, or intentionally regenerate a
matching entry with `--refresh-cache`. Cached embeddings and large indexes are
git-ignored; reports record the cache key, checkpoint revision, and hit/miss
state but never embed the vectors themselves.

Warm all four caches without rebuilding either retrieval index:

```bash
python benchmark/cache_embeddings.py \
  --dataset beir/fiqa/test --limit-docs 10000 --limit-queries 100 \
  --sampling qrels
```

Compare uncompressed and PLAID-compressed MaxSim over the identical exact-FDE
candidate sets:

```bash
python benchmark/uncompressed_oracle.py \
  --index benchmark/results/fiqa-qrels-10k/plaid \
  --dataset beir/fiqa/test --limit-docs 10000 --limit-queries 100 \
  --sampling qrels --candidates 250 --scope candidates
```

Use `--scope exhaustive` for the uncompressed encoder ceiling, or `--scope both`
for both tests. Exhaustive mode performs substantially more matrix multiplication;
candidate mode is the fast test that isolates compression from candidate pruning.

Audit the trained projection and ColBERT query/document conventions after model
or dependency changes:

```bash
python benchmark/audit_encoder.py
```

The command fails if the loaded 128-dimensional projection differs from the
checkpoint or if marker tokens, mask expansion, punctuation filtering, maximum
lengths, or output normalization no longer match the expected ColBERTv2 path.
Full comparison and oracle reports also include deterministic paired-bootstrap
confidence intervals and two-sided p-values for nDCG@10 deltas.

Benchmark cached MiniLM embeddings with exact search and the vectordb HNSW
implementation, without rebuilding the multi-vector index:

```bash
python benchmark/dense_baseline.py \
  --dataset beir/fiqa/test --limit-docs 10000 --limit-queries 100 \
  --sampling qrels --m 16 --ef-search 256
```

## Validated SciFact finding

The original 180-token document configuration produced an uncompressed
ColBERTv2 ceiling of 0.6464 nDCG@10 and led to the incorrect inference that the
remaining recall deficit was model behavior. With the BEIR document maximum set
to 300, the full 300-query SciFact test produces:

| system | nDCG@10 | recall@10 |
| --- | ---: | ---: |
| exact MiniLM | 0.6451 | 0.7833 |
| MUVERA + compressed PLAID, c=500 | 0.6806 | 0.7762 |
| uncompressed ColBERT, same candidates | 0.6844 | 0.7696 |
| uncompressed ColBERT, exhaustive | 0.6917 | 0.7946 |

The compressed pipeline's +0.0355 nDCG delta over exact MiniLM has a paired
bootstrap 95% interval of [+0.0019, +0.0698] and a two-sided p-value of 0.0356.
This supports the narrower conclusion that the pipeline preserves SciFact's
late-interaction advantage; it is not evidence that late interaction wins on
every corpus.

The corrected exhaustive recall also retracts the earlier model-behavior
explanation. Candidate pruning introduces the observed recall deficit. The
cached candidate sweep measures the following frontier:

| candidates | corpus fraction | nDCG@10 | recall@10 | p50 ms | p95 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 9.6% | 0.6806 | 0.7762 | 51.6 | 60.3 |
| 1,000 | 19.3% | 0.6823 | 0.7812 | 97.0 | 108.2 |
| 2,000 | 38.6% | 0.6858 | 0.7946 | 188.5 | 207.7 |

There is therefore no corpus-independent production candidate default yet:
500 is the SciFact latency-oriented point, while 2,000 recovers exhaustive
recall at roughly 3.7 times the median latency. The append-only report ledger is
the source of truth for the full records and encoder/cache provenance.

PLAID centroid-interaction pruning is available by passing a broad `candidates`
count and a smaller `rerank_candidates` count. The 2,000 → 500 pruning frontier
depends strongly on codebook size:

| codebook | direct-500 nDCG | 2,000→500 nDCG | 2,000→500 p50 | direct-2,000 nDCG |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.6806 | 0.6694 | 61.3 ms | 0.6858 |
| 512 | 0.6711 | 0.6732 | 57.3 ms | 0.6776 |
| 1,024 | 0.6806 | 0.6870 | 58.1 ms | 0.6880 |

At 256 centroids, 4:1 pruning drops below direct-500 — centroid-only ranking is
not faithful enough to keep the right survivors. At 1,024 centroids, 2,000→500
reaches within 0.001 nDCG of direct-2,000 (0.6870 vs 0.6880) at about a third of
its latency (58 ms vs 196 ms), and its recall@10 of 0.7896 exceeds every
direct-500 configuration. This confirms the Codex log's hypothesis: centroid
resolution, not the pruning mechanism, was the ceiling. 2:1 pruning (2,000 →
1,000) is essentially free at any codebook size — its top-10 matches direct-
2,000 exactly at c=512 and c=1,024.

Direct-500 nDCG@10 is roughly flat across codebook sizes (0.6711–0.6806) within
paired-bootstrap noise, so larger codebooks pay for themselves through pruning
headroom rather than through the direct path.

The same c=1,024 pruning matrix on FiQA (10,000 documents, 100 qrels-sampled
queries) reproduces the finding more strongly:

| candidates | rerank_candidates | nDCG@10 | recall@10 | p50 ms |
| ---: | ---: | ---: | ---: | ---: |
| 500 | — | 0.4491 | 0.5274 | 71.7 |
| 1,000 | — | 0.4598 | 0.5404 | 119.2 |
| 2,000 | — | 0.4670 | 0.5545 | 221.5 |
| 2,000 | 500 | 0.4670 | 0.5545 | 67.2 |
| 2,000 | 1,000 | 0.4670 | 0.5545 | 106.5 |

On FiQA, 2,000→500 pruning matches direct-2,000 exactly on both nDCG@10 and
recall@10, so the top-10 rankings are identical across all 100 queries — the
centroid ranking at c=1,024 preserves the whole rerank-relevant portion of the
top-2,000. The latency win is 3.3× vs direct-2,000, and quality strictly
exceeds direct-500. This is stronger than SciFact (where 2,000→500 was within
0.001 nDCG of direct-2,000, not identical), consistent with the codebook-scale
hypothesis: centroid ranking fidelity, not the pruning mechanism, is the
bottleneck.

Two operational notes from these runs:
- Building c=1,024 costs ~4× the k-means and per-token nearest-centroid work of
  c=256, but that's dwarfed by ColBERT encoding on CPU when the cache misses.
  Warm the embedding cache before comparing codebook sizes.
- Centroid pruning also works with the HNSW candidate backend: pass
  `candidate_backend: "hnsw"` together with `rerank_candidates` and the same
  broad-fetch → centroid-prune → residual-rerank pipeline runs, sourcing broad
  candidates from the FDE ANN graph instead of the exact FDE scan.
