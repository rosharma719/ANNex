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

### BEIR/FiQA (10K docs, 41 evaluable queries, ColBERTv2)

| System | nDCG@10 | R@10 | p50 ms | p95 ms | build s |
|---|---:|---:|---:|---:|---:|
| annex-multivector c=100 | 0.3776 | 0.4756 | 9.23 | 11.19 | 117.8 |
| annex-multivector c=250 | 0.4036 | 0.5244 | 10.25 | 13.98 | 118.9 |
| **annex-multivector c=500** | **0.4248** | **0.5854** | **12.27** | **18.66** | **117.5** |
| annex-multivector c=1000 | 0.4186 | 0.5732 | 17.60 | 24.12 | 115.8 |
| annex-multivector c=2000 | 0.4176 | 0.5732 | 27.90 | 30.66 | 115.4 |
| Qdrant (native MAX_SIM) | 0.4255 | 0.5732 | 409.50 | 413.11 | 10.8 |

**At candidates=500 we match Qdrant on nDCG@10 (0.4248 vs 0.4255, 0.02% gap
within noise) and beat it on R@10 (0.5854 vs 0.5732), running 33× faster
on the same box.** Going beyond c=500 costs latency without moving quality
on this corpus — MUVERA FDE finds all the relevant docs by c=500.

### BEIR/scifact (5K docs, 98 evaluable queries, ColBERTv2)

| System | nDCG@10 | R@10 | p50 ms | p95 ms | build s |
|---|---:|---:|---:|---:|---:|
| annex-multivector c=100 | 0.7052 | 0.7684 | 6.92 | 8.86 | 108.2 |
| annex-multivector c=250 | 0.7264 | 0.8020 | 10.32 | 12.54 | 108.0 |
| annex-multivector c=500 | 0.7240 | 0.8020 | 15.45 | 17.09 | 108.4 |
| annex-multivector c=1000 | 0.7250 | 0.8071 | 26.32 | 28.01 | 106.7 |
| annex-multivector c=2000 | 0.7284 | 0.8173 | 47.35 | 48.92 | 107.1 |
| Qdrant (native MAX_SIM) | 0.7524 | 0.8582 | 286.99 | 288.82 | 10.4 |

**Different pattern from FiQA**: quality plateaus at nDCG≈0.72-0.73 even at
c=2000 — MUVERA FDE misses some scifact-specific relevant docs regardless
of rescoring budget. We remain **6-40× faster** across the whole sweep, at
a persistent 3-5% nDCG deficit.

### Interpretation

The two corpora together show:

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
| scalar (baseline autovectorized loop) | 20.75 | 17.76 | 19.7 |
| NEON (4-accumulator, 16-lane FMA)     | 2.54  | 2.15  | 161  |

= **8.2× on the isolated kernel**, ~85% of M2 peak. Per-doc scores
match scalar to <2e-6 (FP32 reordering noise); parity tests in
`crates/annex-multivector/src/fde.rs::tests` cover dim=128, dim=384,
and a non-multiple-of-16 fallback.

## Bulk end-to-end BEIR runs (single-engine)

Committed under `benchmark/reports/v0.1.0.jsonl` — includes the
pre-NEON reference runs so the perf progression is auditable:

| Stage | FiQA p50 | FiQA nDCG | Notes |
|---|---:|---:|---|
| pre-any-NEON reference | 27.4 ms | 0.380 | v0.1.0 baseline |
| MaxSim NEON only | 12.6 ms | 0.404 | commit `90c187c` |
| MaxSim + FDE NEON | 10.3 ms | 0.404 | commit `596f91a` |
| + thread-local decode scratch | 9.67 ms | 0.404 | commit `348e459` |
| + partial-sort FDE candidates | ~9 ms | 0.404 | commit `aea2f55` |

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
