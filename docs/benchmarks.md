# Benchmarks And Harnesses

This document keeps the dataset-specific commands and recorded benchmark numbers out of the root `README`.

## NYTimes (256-D Angular)

Download instructions live in [data-download.md](./data-download.md).

**Canonical Pareto sweep** (ANNex M=16/24/32 variants vs hnswlib/usearch/faiss):

```bash
./bench/nyt256/run_annexdb.sh            # build indexes + sweep ANNex (M=16/24/32 × sq8 × rcm)
python3 bench/nyt256/run_competitors.py  # sweep hnswlib, usearch, faiss-hnsw
python3 bench/nyt256/evaluate.py         # merge → results_all.csv + Pareto table
```

Current results: `bench/nyt256/results_all.csv` — 92 rows, recall@20 vs p50/p99/QPS.
Provenance: `bench/nyt256/manifest.json` — commit, timestamp, rust version, library versions.
Environment: `bench/nyt256/environment.md` — hardware, dataset, methodology.

**Build a snapshot** (for construction-quality experiments):

```bash
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_m16_efc300.bin \
VECTORDB_M=16 VECTORDB_NYT_EF_CONSTRUCT=300 \
  cargo test --release --test nytimes_frontier nytimes_build_and_persist_snapshot_only \
    -- --ignored --nocapture
```

**Recall investigation** (one-time, 2026-09-14): `docs/recall-investigation.md` + `bench/nyt256/recall-investigation-results.json`.

**Tail-recall notes** (from M=16 efc=100 baseline, patterns carry forward):
- Misses concentrate on saturated level-0 nodes (degree at M0 cap). Entry/routing quality and neighbor diversity are the main levers, not a global EF bump.
- Marginal recall per added ms drops fast: +0.089 recall/0.11 ms (EF 32→64) vs +0.029/0.977 ms (EF 256→512).

## H&M (2048-D Cosine)

Download instructions live in [data-download.md](./data-download.md).

Run the filtered recall harness:

```bash
cargo test --release hnm_filtered_cosine_recall -- --ignored --nocapture
```

Useful runtime knobs:

- `VECTORDB_HNM_TOPK`
- `VECTORDB_HNM_EF_SEARCH_LIST`
- `VECTORDB_HNM_QUERIES`
- `VECTORDB_HNM_BASE_LIMIT`
- `VECTORDB_HNM_EF_CONSTRUCT`

### Recorded filtered recall/latency curve

Dataset: H&M 2048D cosine, `ef_construct=100`, `M=16`

- `EF=32`: `1329.8 qps`, `0.752 ms/query`, `0.566 recall`
- `EF=64`: `1293.8 qps`, `0.773 ms/query`, `0.709 recall`
- `EF=128`: `787.4 qps`, `1.270 ms/query`, `0.859 recall`
- `EF=256`: `501.5 qps`, `1.994 ms/query`, `0.907 recall`
- `EF=512`: `338.8 qps`, `2.952 ms/query`, `0.939 recall`

## General performance notes

Euclidean, `dim=1536`, `top_k=20`, `ef_construct=100`, `m=16`, `ef_search=64`

- `20,000` vectors: insert `5.24 s` (~`3.8k vec/s`), search `0.198 ms/query`
- `100,000` vectors: insert `33.24 s` (~`3.1k vec/s`), search `0.310 ms/query`
- `1,000,000` vectors: insert `475.81 s` (~`2.1k vec/s`), search `0.495 ms/query`
