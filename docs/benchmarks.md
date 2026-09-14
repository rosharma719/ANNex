# Benchmarks And Harnesses

This document keeps the dataset-specific commands and recorded benchmark numbers out of the root `README`.

## NYTimes (256-D Angular)

Download instructions live in [data-download.md](./data-download.md).

Run the main harness:

```bash
cargo test --release nytimes_256_angular_perf_and_recall -- --ignored --nocapture
```

Run the QPS/latency curve from a persisted snapshot:

```bash
VECTORDB_USE_SNAPSHOT=1 \
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_m16_m0_32_efc100.bin \
cargo test --release nytimes_qps_latency_curve -- --ignored --nocapture
```

Build a snapshot with trace logging:

```bash
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_m16_m0_32_efc100.bin \
VECTORDB_NYT_M=16 \
VECTORDB_NYT_M0=32 \
VECTORDB_NYT_EF_CONSTRUCT=100 \
VECTORDB_NYT_EF_SEARCH_LIST=100 \
VECTORDB_DIVERSITY_ALPHA=1 \
VECTORDB_INSERT_TRACE_LOG=logs/nytimes_insert_m16_m0_32_efc100.jsonl \
VECTORDB_NYT_ALLOW_BUILD=1 \
VECTORDB_NYT_SAVE_SNAPSHOT=1 \
cargo test --release nytimes_build_and_persist_snapshot_only -- --ignored --nocapture
```

Analyze a snapshot:

```bash
cargo run --bin index_analyzer -- \
  --snapshot data/nytimes-256-angular/index_m16_m0_32_efc100.bin \
  --base data/nytimes-256-angular/base.npy \
  --queries data/nytimes-256-angular/queries.npy \
  --top-k 10 \
  --num-queries 100 \
  --sample-size 500
```

Sweep multiple snapshots:

```bash
cargo run --bin snapshot_sweeper -- \
  --snapshots data/nytimes-256-angular/index_m16_m0_16_efc100.bin,\
data/nytimes-256-angular/index_m16_m0_32_efc100.bin \
  --output logs/nyt_snapshot_sweep.jsonl \
  --top-k 10 \
  --num-queries 100 \
  --sample-size 500 \
  --neighbor-scan-cap 128
```

For canonical naming and build manifests, see [index-construction.md](./index-construction.md).

### Recorded unfiltered recall/latency curve

Current NYTimes build, `top_k=20`, 1,000 queries:

Dataset: NYT-256-Angular, `ef_construct=100`, `M=16`, `M0=32`, `diversity_alpha=1`

- Inserted `290,000` vectors in `162.847 s` (`0.562 ms/insert`)
- Persisted snapshot to `data/nytimes-256-angular/index_m16_m0_32_efc100.bin`
- Snapshot size: `390,378,487` bytes
- Persist elapsed: `3.633 s`

| EF search | Recall@20 | Avg ms/query | p50 / p90 / p99 ms | Visited p50 / p90 / p99 | Expanded p50 / p90 / p99 | Adjacency reads p50 / p90 / p99 | Distance computations p50 / p90 / p99 | Misses |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.735 | 0.191 | 0.183 / 0.212 / 0.508 | 795 / 910 / 958 | 32 / 32 / 32 | 1056 / 1056 / 1056 | 794 / 909 / 957 | 5302 |
| 64 | 0.824 | 0.302 | 0.306 / 0.350 / 0.389 | 1477 / 1740 / 1849 | 64 / 64 / 64 | 2112 / 2112 / 2112 | 1476 / 1739 / 1848 | 3530 |
| 128 | 0.869 | 0.649 | 0.569 / 0.736 / 2.484 | 2842 / 3338 / 3537 | 128 / 128 / 128 | 4224 / 4224 / 4224 | 2841 / 3337 / 3536 | 2627 |
| 256 | 0.902 | 1.014 | 1.060 / 1.224 / 1.324 | 5502 / 6469 / 6766 | 256 / 256 / 256 | 8448 / 8448 / 8448 | 5501 / 6468 / 6765 | 1964 |
| 512 | 0.931 | 1.991 | 2.087 / 2.395 / 2.718 | 10844 / 12473 / 12985 | 512 / 512 / 512 | 16896 / 16896 / 16896 | 10843 / 12472 / 12984 | 1388 |

Missed ground-truth neighbors are almost entirely level-0 nodes, and their level-0 degree is saturated:

| EF search | Miss levels | Miss degree p50 / p90 / p99 |
| ---: | --- | ---: |
| 32 | L0=4965, L1=321, L2=16 | 33 / 33 / 33 |
| 64 | L0=3302, L1=217, L2=11 | 33 / 33 / 33 |
| 128 | L0=2456, L1=162, L2=9 | 33 / 33 / 33 |
| 256 | L0=1845, L1=113, L2=6 | 33 / 33 / 33 |
| 512 | L0=1307, L1=76, L2=5 | 33 / 33 / 33 |

The marginal recall return from raising EF falls quickly:

| EF step | Delta recall | Delta ms/query | Recall points per 1 ms |
| --- | ---: | ---: | ---: |
| 32 -> 64 | +0.089 | +0.111 | 0.802 |
| 64 -> 128 | +0.045 | +0.347 | 0.130 |
| 128 -> 256 | +0.033 | +0.365 | 0.090 |
| 256 -> 512 | +0.029 | +0.977 | 0.030 |

Tail-recall diagnosis:

- Search work is deterministic and fully budget-bound in this run: `expanded == ef_search`, `cap_breaks=0`, and `patience_breaks=0` for every percentile. Raising EF buys recall by linearly expanding more full level-0 neighborhoods, not by finding a better stopping point.
- `adjacency_reads == 33 * ef_search`, matching saturated `M0=32` plus the current node. That means latency is almost exactly proportional to the number of expanded nodes and the graph is using the same dense local fanout for easy and hard queries.
- Misses are mostly saturated level-0 nodes, so the tail is not caused by low out-degree on missed targets. It is more likely caused by entry/routing quality and neighbor selection diversity: the search is reaching a local basin and then spending extra EF inside it.
- To improve tail recall without increasing latency, prefer graph-quality changes and adaptive query work over a global EF bump:
  - Rebuild candidate snapshots with higher construction quality at the same query EF first: `ef_construct=200/300`, then compare `M=16,M0=32` vs `M=24,M0=48` only if the memory/insert cost is acceptable.
  - Sweep the diversity knobs already in the manifest (`VECTORDB_DIVERSITY_ALPHA_LOW`, `VECTORDB_DIVERSITY_ALPHA_HIGH`, `VECTORDB_DIVERSITY_PRUNE_FLOOR`). The target is better long-range/bridge edges, not more query expansions.
  - Run the existing query-grid pipeline with `--recompute-from-queries` and inspect `query_hardness.csv` plus `query_knob_effects.csv`. Use it to identify hard queries where extra work helps, then keep the default EF low and spend saved budget only on those queries.
  - Try capped neighbor scans with query-dependent rotation/stride for EF values above 64. If recall holds, the saved adjacency reads can fund a second seed or a hard-query retry without increasing average latency.
  - Add or prototype multi-entry level-0 search from the best few upper-layer candidates. This directly attacks bad entry routing and can recover tail misses with the same expansion budget by starting from multiple basins instead of expanding one basin deeper.

Historical QPS/latency curve:

Dataset: NYT-256-Angular, `ef_construct=100`, `M=16`, `M0=32`, `diversity_alpha=1`

- `EF=32`: `1894.6 qps`, `0.528 ms/query`, `0.816 recall`
- `EF=64`: `1307.5 qps`, `0.765 ms/query`, `0.860 recall`
- `EF=128`: `719.4 qps`, `1.390 ms/query`, `0.895 recall`
- `EF=256`: `383.7 qps`, `2.606 ms/query`, `0.926 recall`
- `EF=512`: `200.8 qps`, `4.980 ms/query`, `0.953 recall`

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
