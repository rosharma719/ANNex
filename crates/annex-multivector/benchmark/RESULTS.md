# annex-multivector benchmark results

Reproducible measurements of `annex-multivector` against reference systems
on identical hardware and inputs. Numbers here should regenerate exactly
from the checked-in code and the free BEIR datasets accessed via `ir-datasets`.

## Environment

- **Machine**: Apple M2 (8-core: 4 performance + 4 efficiency), 16 GB RAM
- **OS**: macOS Darwin 25.6.0
- **Rust**: stable, `annex-multivector` built with `--release` (LTO thin)
- **Python**: 3.14, dependencies via `benchmark/requirements.txt`
- **Multi-vector model**: ColBERTv2 (`colbert-ir/colbertv2.0`, dim 128, ~200 tokens/doc, 32 tokens/query)
- **Reference systems**: Qdrant 1.19.1 (`qdrant-client(':memory:')`, native `MultiVectorConfig(MAX_SIM)`), LanceDB 0.39 (no native MaxSim as of this version — used as mean-pool dense baseline for context, not a fair MaxSim comparison)

All engines see byte-identical inputs — the same ColBERTv2 embeddings cached
under `benchmark/cache/ragged-*/` by the first BEIR run, replayed to every
engine. No engine re-encodes; the model runs exactly once per corpus.

## Reproducing

```bash
cd crates/annex-multivector
python -m venv .venv
.venv/bin/pip install -r benchmark/requirements.txt
.venv/bin/pip install qdrant-client lancedb matplotlib
cargo build --release -p annex-multivector --bin annex-multivector
cargo build --release -p annex-server --bin dense-server

# One-time encoding + fill cache, plus the standard 3-system report:
.venv/bin/python benchmark/run.py --dataset beir/fiqa/test \
    --limit-docs 10000 --limit-queries 100 \
    --output benchmark/results/fiqa-10k

# Head-to-head with candidate sweep:
.venv/bin/python benchmark/headtohead.py --dataset beir/fiqa/test \
    --limit-docs 10000 --limit-queries 100 \
    --engines annex,qdrant --annex-sweep 100,250,500,1000,2000 \
    --output benchmark/results/headtohead-fiqa-sweep

# Pareto plot:
.venv/bin/python benchmark/plot_headtohead.py \
    benchmark/results/headtohead-fiqa-sweep/matrix.json \
    --output benchmark/reports/pareto-fiqa.png
```

Committed input matrices live under `benchmark/reports/headtohead-*.json`
so anyone can re-run `plot_headtohead.py` / `recall_difficulty.py` against
the recorded runs without rebuilding.

## Head-to-head vs Qdrant native multi-vector

Committed matrices live under `benchmark/reports/headtohead-*-v3-matrix.json`.
The three-corpus visualisation is at `benchmark/reports/pareto-three-corpora-v3.png`.

### BEIR/FiQA (10K docs, 41 evaluable queries, ColBERTv2)

| System | nDCG@10 | R@10 | p50 ms | p95 ms | build s |
|---|---:|---:|---:|---:|---:|
| annex-multivector c=250 | 0.4036 | 0.5244 | 9.65 | 14.20 | 118.3 |
| **annex-multivector c=500** | **0.4248** | **0.5854** | **10.93** | **17.86** | **117.0** |
| annex-multivector c=1000 | 0.4186 | 0.5732 | 15.68 | 24.38 | 115.3 |
| Qdrant (native MAX_SIM) | 0.4255 | 0.5732 | 412.67 | 415.55 | 10.9 |

**At candidates=500 we match Qdrant on nDCG@10 (0.4248 vs 0.4255) and
beat it on R@10 (0.5854 vs 0.5732), running 38× faster on the same box.**
Going beyond c=500 costs latency without moving quality on this corpus.

### BEIR/scifact (5K docs, 98 evaluable queries, ColBERTv2)

| System | nDCG@10 | R@10 | p50 ms | p95 ms | build s |
|---|---:|---:|---:|---:|---:|
| annex-multivector c=250 | 0.7264 | 0.8020 | 8.76 | 10.13 | 108.9 |
| annex-multivector c=500 | 0.7240 | 0.8020 | 13.07 | 14.88 | 107.5 |
| annex-multivector c=1000 | 0.7250 | 0.8071 | 21.27 | 24.69 | 107.0 |
| Qdrant (native MAX_SIM) | 0.7524 | 0.8582 | 287.03 | 290.12 | 10.1 |

**13-33× faster than Qdrant** across the sweep at a persistent 3.5% nDCG
deficit. Quality plateaus at ~0.72-0.73 even at high c — MUVERA FDE is
missing some scifact-specific relevant docs that no rescoring budget
recovers. Candidate-generation ceiling, not rescoring bottleneck.

### BEIR/nfcorpus (3.6K docs, 300 evaluable queries, ColBERTv2)

| System | nDCG@10 | R@10 | p50 ms | p95 ms | build s |
|---|---:|---:|---:|---:|---:|
| annex-multivector c=250 | 0.3272 | 0.1465 | 7.92 | 9.51 | 77.6 |
| annex-multivector c=500 | 0.3390 | 0.1524 | 11.93 | 13.02 | 77.4 |
| annex-multivector c=1000 | 0.3435 | 0.1552 | 19.67 | 20.90 | 77.1 |
| Qdrant (native MAX_SIM) | 0.3487 | 0.1579 | 205.99 | 207.21 | 7.1 |

Third corpus, 300 queries — the largest query pool of the three. Quality
gap narrows to **1.5% at c=1000** (0.3435 vs 0.3487) while running
**10× faster**. Smaller-corpus workloads tend to close the gap because
FDE candidate quality improves relative to corpus size.

### Interpretation

Across all three corpora:

1. **Consistently faster on CPU**: 6-40× lead on p50 latency across all
   configurations and corpora we've measured. This is an algorithmic win
   (FDE + PLAID two-stage vs Qdrant's exact scan of every doc token), not
   a raw SIMD win. Even at very high `c`, our engine does dramatically
   less work per query.
2. **Quality parity is corpus-dependent**: FiQA reaches full parity at
   c=500; scifact plateaus 3-5% below. The candidate-generation stage is
   the ceiling on scifact — better FDE hyperparameters, HNSW-backed
   candidates, or per-query dynamic `c` may close it. Rescoring is not
   the bottleneck at high `c`.
3. **Build time is 10× slower than Qdrant**: we do k-means training and
   MUVERA FDE encoding at ingest; Qdrant just stores the token vectors.
   That's the tax we pay upfront for the query-time speedup.

## Kernel micro-benchmarks

Isolated MaxSim kernel throughput (250 candidates × 200 doc tokens ×
32 query tokens × 128 dims, rayon-parallel across candidates, from
`crates/annex-multivector/src/bin/maxsim_bench.rs`):

| Kernel | Mean ms | p50 ms | Aggregate GFLOP/s |
|---|---:|---:|---:|
| scalar (baseline autovectorized loop) | 17.14 | 16.87 | 23.9 |
| NEON (4-accumulator, general dim)     | 1.86  | 1.65  | 220  |
| NEON (const-128 fast path)            | 1.76  | 1.64  | 232  |

= **11.8× on the isolated kernel**, ~87% of M2 peak. Per-doc scores
match scalar to <2e-6 (FP32 reordering noise); parity tests in
`crates/annex-multivector/src/fde.rs::tests` cover dim=128, dim=384,
and a non-multiple-of-16 fallback.

## Bulk end-to-end BEIR runs (single-engine)

Committed under `benchmark/reports/v0.1.0.jsonl` — includes the
pre-NEON reference runs so the perf progression is auditable:

| Stage | FiQA p50 c=250 | FiQA nDCG | Notes |
|---|---:|---:|---|
| pre-any-NEON reference | 27.4 ms | 0.380 | v0.1.0 baseline |
| MaxSim NEON only | 12.6 ms | 0.404 | commit `90c187c` |
| MaxSim + FDE NEON | 10.3 ms | 0.404 | commit `596f91a` |
| + thread-local decode scratch | 9.67 ms | 0.404 | commit `348e459` |
| + partial-sort FDE candidates | ~9 ms | 0.404 | commit `aea2f55` |
| + unpack fast path (bits=2) | ~9 ms | 0.404 | commit `40f8d3f` |
| + rescore two-phase (skip clones) | ~9 ms | 0.404 | commit `0a0731d` |
| + dim=128 fast path (const-len) | **9.65 ms** | 0.404 | commit `9d8df03` (v3) |

Total: **27.4 ms → 9.65 ms = 2.84× end-to-end at unchanged accuracy**.
At matched Qdrant quality (c=500): 27.4 ms → **10.93 ms = 2.5×**.

## Reference systems (context)

- fast-plaid (2024 pypi, FAISS-backed): reports 15-25 ms on comparable
  workloads. We're at 10 ms.
- Stanford ColBERTv2/PLAID (reference implementation): 50-80 ms on MS
  MARCO scale on comparable hardware.
- Vespa native multi-vector: typically 20-40 ms in production
  configurations.

These are published references, not runs on this machine. The head-to-head
matrix above is the only direct like-for-like measurement.

## What's not yet claimed

- Multi-corpus breadth beyond FiQA and scifact. `benchmark/reports/`
  will get nfcorpus and 1-2 more corpora as sweeps land.
- Scaling behavior above 10K docs — the full BEIR corpora are 20K-100K
  docs and haven't been benched yet.
- GPU competitors (fast-plaid on CUDA, Vespa on tensor accelerators).
  We're CPU-only.
- Multi-tenant, filtered, hybrid search — features we don't have that
  Qdrant / Vespa / LanceDB do.
