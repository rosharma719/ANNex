# Benchmark evidence and corrections

No retained run currently establishes production superiority or launch readiness.

Historical development runs, plots, ledgers and investigation notes are preserved
at [revision 3e709d2](https://github.com/rosharma719/ANNex/tree/3e709d25735f1481b80a06deddb25db3c6368c43).
For example:

```sh
git show 3e709d2:crates/annex-multivector/benchmark/reports/headtohead-fiqa-v3-matrix.json > /tmp/fiqa-history.json
```

Generated copies have been removed from the active tree. This document retains
the corrections; the [benchmark policy](../../../BENCHMARK_POLICY.md) governs new claims.

| Earlier claim | Correction |
| --- | --- |
| “6–40× faster than Qdrant” and FiQA “quality parity” | These were small, explored slices against `qdrant-client :memory:`, not production Qdrant Server. They do not establish server performance or statistical equivalence. |
| Equal nDCG/Recall proves identical top-10 rankings | Aggregate equality cannot establish ranking identity. The pruning ledger did not retain sufficient rankings for that assertion. |
| Comparable build times | ANNex training was excluded; competitor readiness and configuration were not consistently verified. |
| Exact reproduction from code and free datasets | Historical records lack complete frozen model, binary, configuration and input identity. Timing also varies between runs. |
| LanceDB has no native multivector support | The old adapter mean-pooled tokens. Treat those numbers as a mean-pool ablation, not evidence about native multivector retrieval. |
| SciFact's remaining deficit was necessarily model behavior | The original encoder used a shorter document limit. Later experiments changed that limit and exposed candidate-pruning losses. |
| Unsourced latency ranges for PLAID and Vespa | Withdrawn; these were not comparable measurements on the recorded workload. |

Historic NYT/H&M curves and optimization notes are also exploratory. Their
original files remain in the same revision. Current commands live in
[docs/benchmarks.md](../../../docs/benchmarks.md); fresh measurements must include
their own artifacts rather than copy these tables forward.
