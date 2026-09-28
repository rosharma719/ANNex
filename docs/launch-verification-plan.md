# Launch verification work order

Target: a self-hosted, single-node CPU service for text RAG. This is a roadmap,
not a declaration of launch readiness or comparative superiority.

The current implementation includes the measurement/API foundation, filtered
hybrid retrieval, named dense/sparse/multivector fields, context selection,
collection namespaces, immutable query generations, retained mappings and vector
segment compaction. The API and storage contracts below own their guarantees
and limitations; benchmark results must distinguish completed evidence from
planned evaluations.

## Remaining implementation order

1. **Measurement coverage:** extend durable benchmark journals to matched-quality
   competitor configurations, held-out corpora and answer/evidence evaluation;
   measure actual backends and complete build/resource costs.
2. **Bounded serving and lifecycle:** separate query/ingest/maintenance execution,
   admission limits and cancellation; atomic source replacement, revision
   preconditions and idempotent retries.
3. **Retrieval scaling:** accelerate selective filters against the existing exact
   oracle, measure adaptive rerank/context policies, and avoid quality claims from
   fixtures or inspected development queries.
4. **Write scaling and maintenance:** incremental durable records/checkpoints,
   bounded delta folding, graph persistence/restart readiness and consistent
   backup/restore; stress existing compaction under concurrent writes/readers.
5. **Measured optimization:** profile FP16/SQ8 FDE storage, duplicate-vector
   removal, fused compressed MaxSim and SIMD; retain independent scalar/FP32
   oracles. Promote changes only after quality/latency/memory measurements.

## Verification suite

| Track | Workloads | Measurements |
| --- | --- | --- |
| Retrieval | Full BEIR/LoTTE corpora, BRIGHT, fresh human-judged application queries | nDCG, Recall, MRR, domain and difficult-query slices |
| Answering | Evidence-and-answer tasks and consented real documentation queries | Answer correctness, evidence coverage, citation support, abstention |
| Engine | Identical cached embeddings/scoring/filter semantics | Quality-matched throughput, p50/p95/p99, CPU, RAM, disk |
| Operations | 1M+ real chunks, concurrency 1/8/32/128 and offered load, mutations/maintenance | Freshness, errors, recovery and bounded resource growth |
| Endurance | Crash injection, restore/migration, proposed 24–72-hour soak | Whole-generation recovery and reproducible restored scores |

Compare properly configured Qdrant Server, native LanceDB multivector,
ColBERT/PLAID and Vespa; include BM25 and dense+sparse+reranker baselines.
Keep database, encoding and generator costs separate and report end-to-end cost.
Use fixed generator/context settings when attributing answer gains to retrieval.

Launch requires successful recovery/restore drills, no observed stale/deleted or
out-of-scope results in the defined suites, predeclared quality margins, a
million-scale mixed workload meeting published latency/resource limits, and an
independent reproduction. A 2× throughput advantage at matched quality and memory
is an ambition to test, not a promised result.

The [benchmark policy](../BENCHMARK_POLICY.md) owns fairness and reporting rules;
the [durability contract](multivector-durability.md) owns storage guarantees.
