# Counterfactual physical-plan evaluation

`crates/annex-multivector/benchmark/plan_evaluation.py` runs stored queries through
real `/v1/plan` and `/v1/retrieve` operators on an existing, quiescent service.
It does not generate embeddings, ingest a corpus, or train a utility model.
[Benchmark policy](../BENCHMARK_POLICY.md) governs held-out and performance claims.

## Declare inputs and the finite plan grid

Supply a version-1 JSON spec with corpus/configuration/model fingerprints, stored
query inputs and qrels, named request templates, one selection request template,
evaluation top-k, repetitions and positive latency budgets. A placeholder object
`{"$input": "name"}` substitutes the query input of that name. Other JSON objects
are preserved recursively. All plans must use the selection's filter and return
at least evaluation top-k results. IDs must be unique, nonempty strings.

```json
{
  "version": 1,
  "fingerprints": {
    "corpus": "sha256:REPLACE_WITH_CORPUS_DIGEST",
    "configuration": "sha256:REPLACE_WITH_INDEX_AND_CALIBRATION_DIGEST",
    "models": "REPLACE_WITH_PINNED_ENCODER_REVISIONS_AND_EMBEDDING_DIGESTS"
  },
  "top_k": 10,
  "repeats": 3,
  "budgets_ms": [5, 20, 50],
  "epsilon_quality": 0.01,
  "epsilon_latency_ms": 0.1,
  "queries": [{
    "id": "q1",
    "inputs": {"dense": [0.1, 0.2], "text": "stored query text"},
    "qrels": {"doc-a": 2, "doc-b": 1}
  }],
  "plans": [
    {"id": "dense-exact", "request": {
      "prefetch": [{"kind": "dense", "field": "semantic", "vector": {"$input": "dense"}, "backend": "exact", "limit": 100}],
      "limit": 10
    }},
    {"id": "dense-hnsw-64", "request": {
      "prefetch": [{"kind": "dense", "field": "semantic", "vector": {"$input": "dense"}, "backend": "hnsw", "ef_search": 64, "limit": 100}],
      "limit": 10
    }},
    {"id": "hybrid", "request": {
      "prefetch": [
        {"kind": "bm25", "text": {"$input": "text"}, "limit": 100},
        {"kind": "dense", "field": "semantic", "vector": {"$input": "dense"}, "backend": "exact", "limit": 100}
      ],
      "limit": 10
    }}
  ],
  "selection": {"request": {
    "planning_mode": "auto",
    "query": {"text": {"$input": "text"}, "dense": {"semantic": {"$input": "dense"}}},
    "limit": 10
  }}
}
```

These vectors and judgments illustrate the schema, not benchmark evidence. Use
real stored representations and original corpus judgments. Add sparse, named or
unnamed multivector channels, reranking depths and ef settings as explicit entries;
unsupported/not-built operators are recorded as failures. Current operators do
not expose compressed traversal/block width as planner options.

Fingerprints are caller declarations, not remotely verified corpus/model hashes.
The tool also hashes the full spec/evaluator and records service stats and generation.
Stats/generation guards detect document changes, but cannot prove that graph
topology or calibration files remained unchanged. Their identity is externally
declared, and quiescence remains a caller responsibility.
Retain the server binary/revision, index manifest, graph files, starting calibration,
runtime environment, construction costs, hardware and resource limits with the
published artifact. Use a dedicated service without concurrent queries or writes.

## Run and replay

```bash
python3 crates/annex-multivector/benchmark/plan_evaluation.py \
  /path/to/spec.json /path/to/new-run-directory \
  --base-url http://127.0.0.1:8080 --timeout 60

python3 crates/annex-multivector/benchmark/plan_evaluation.py \
  /path/to/spec.json /path/to/run-directory --replay
```

This first version evaluates the default collection through a dedicated service.
Authentication reads `ANNEX_READ_KEY` from the environment and does not put it in
artifacts. Existing run directories are never overwritten.

The tool compiles every grid and selection request before any retrieval. It records
plan-compilation/pinning wall time and full serialized plans, then converts generated
channels into manual requests with explicit physical backends and limits. This
freezes initial decisions despite subsequent runtime calibration. A compiled
latency-budget gate is removed for replay; it is not a runtime deadline. The
executed signature and actual channel backends must match the frozen decision.
Plans may retain their internal adaptive rerank behavior, which is visible in
traces; plan pinning does not freeze data-dependent execution work.

There is no warmup phase. Repetitions rotate plan order, and all completed HTTP
latencies include client/network overhead. Within each query/plan, latency is the
median of successful repetitions; quality averages all declared repetitions,
assigning zero to failed/missing attempts. Repetitions are not additional independent
relevance judgments. The selection is one initial policy decision per query,
compared against each declared budget; the tool does not reroute it for each budget.
Selection latency measures pinned manual replay, not the original auto request;
initial auto-planning overhead is reported separately and is not added to budget
comparisons. This is an execution-choice evaluation, not a measurement of the
original auto endpoint's end-to-end latency.

## Artifacts and interpretation

- `manifest.json`: declared inputs, spec digest, timeout and measurement protocol.
- `observations.jsonl`: fsynced start/finish events, requests, compiled plans,
  generation, ranked IDs/scores, nDCG/recall/MRR, full response traces and failures.
  Resource counters are explicitly unavailable, not estimated measurements.
- `summary.json`: per-plan outcomes and ranking stability, empirical quality/latency
  frontier, per-query budget comparisons, observed quality regret, epsilon-Pareto
  hit rate, violation/failure rates, p95 overrun and planning overhead.

The oracle is restricted to the declared grid's fully successful plans. Incomplete
grids are flagged; queries with no observed feasible oracle have null regret and
an explicit aggregate coverage count. Regret is `max(0, oracle_quality - chosen_quality)`;
a selection better than the finite grid has zero regret. Failures remain in the
query denominator and have a separate selection-failure rate. Budget violations
require a fully successful selection with median latency above the budget.
Overrun percentiles include zero-overrun successful selections.

A plan is dominated when another has no worse latency/quality and is strictly
better in at least one. For epsilon-Pareto membership, a dominator must improve
quality by at least `epsilon_quality` and latency by at least `epsilon_latency_ms`,
with at least one strict improvement. A failed or over-budget selection, or a query without an observed feasible
oracle, is not a hit. This definition and latency statistic must be retained with reported results.

Replaying an interrupted journal retains missing attempts and ignores a torn final
line; earlier corruption fails. An unfinished run or changed final snapshot is
invalid for comparison. Frozen signatures reject execution/generation mismatches,
retaining the raw response as a failed outcome. Hardware counters, offered-load
tail behavior, CPU/RAM/disk and end-to-end answering quality require separate runs.
The included tiny fixtures validate correctness only.
