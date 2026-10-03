# Apple M2 ANN matrix — partial local run

Date: 2026-10-02 through 2026-10-03

Branch: `codex/gist-benchmark-matrix`

Machine: Apple M2, aarch64

Rust: 1.98.1

This was a local development run. It is suitable for regression analysis, not
an externally validated benchmark claim.

## Protocol

- ANNex configurations: M=16 and M=32, with and without SQ8
- `ef_construction=300`
- `ef_search=32,64,128,256,512`
- RCM disabled
- 1,000 queries, three timing rounds, one query at a time
- Results report recall, QPS, and p50/p95/p99 latency

## Completed query matrices

| Dataset | Rows | Representative best point | Higher-recall point |
|---|---:|---|---|
| MNIST-784 Euclidean | 20 | 3,753 QPS at 0.9967 recall (M16+SQ8, ef=32) | 2,560 QPS at 0.9990 (M32, ef=32) |
| NYT-256 Angular | 20 | 3,059 QPS at 0.8919 (M32+SQ8, ef=64) | 534 QPS at 0.9653 (M32+SQ8, ef=512) |
| SIFT-1M Euclidean | 20 | 10,383 QPS at 0.9375 (M16+SQ8, ef=32) | 1,437 QPS at 0.9991 (M32, ef=256) |
| GloVe-100 Angular | 20 | 3,816 QPS at 0.8791 (M32, ef=64) | 635 QPS at 0.9815 (M32, ef=512) |

The complete points are in each dataset's `results_annexdb.jsonl` and
`results_all.csv`.

## Construction observations

- Removing per-distance vector-arena locking reduced a 20,000-vector NYT M32
  probe from 66–74 seconds to 17 seconds. A 50,000-vector SIFT M32 probe also
  built in 17 seconds.
- SIFT M16: 2,262.83 seconds before that optimization.
- SIFT M32: 825.09 seconds after the optimization.
- GloVe M16: 789.11 seconds; M32: 1,239.80 seconds.
- GIST M16: 2,453.86 seconds and a 3.9 GiB snapshot.
- GIST M32 was stopped after approximately 1 hour 40 minutes of wall time. It
  produced no snapshot or query measurements.

The full-size M32 runs still lose parallel efficiency as their graphs grow.
Neighbor-list synchronization and diversity pruning remain the main observed
construction bottlenecks.

## Work intentionally left incomplete

- No GIST query matrix was produced.
- The queued hnswlib, USearch, and Faiss HNSW sweeps were stopped before they
  began. Therefore MNIST, SIFT, GloVe, and GIST have no same-run competitor
  results in this record.
- NYT has older same-machine competitor measurements in repository history,
  but this partial matrix records only the 20 current ANNex rows.
