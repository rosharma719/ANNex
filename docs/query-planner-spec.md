# ANNex Query Planner Specification and Implementation Handoff

**Canonical document:** this file supersedes the deleted five-phase draft and any planner descriptions in old PR bodies or chat transcripts.

**Current branch:** `feat/query-planner`  
**Current PR:** [#14](https://github.com/rosharma719/ANNex/pull/14)  
**Status:** the truthful Phase 1 vertical slice, basic automatic policy routing, and context packing are implemented. The remaining phases below are design targets, not current capabilities.

## Purpose

ANNex should compile an information need into the least expensive retrieval and context plan that satisfies explicit quality and resource constraints.

The system has five distinct responsibilities:

```text
Information need + constraints + query representations
                         |
                         v
               Retrieval policy planner
                  "What should run?"
                         |
                    Logical plan
                         |
                         v
                 Physical optimizer
                  "How should it run?"
                         |
                    Physical plan
                         |
                         v
                      Executor
              "Run exactly this plan"
                         |
                         v
                       Trace
             "What actually happened?"
                         |
                         v
                    Calibration
        "How wrong were the estimates?"
```

ANNex owns retrieval-policy planning, physical optimization, execution, tracing, and eventually calibration. Semantic decomposition and query reformulation that require an LLM belong in a separate agent layer.

## Non-negotiable invariants

1. **A physical plan must describe execution exactly.** `retrieve()` must not secretly change backend, candidate depth, `ef_search`, filter strategy, or channel set after planning.
2. **Cost and utility remain separate.** Execution cost estimates cannot stand in for expected retrieval quality.
3. **Estimated latency must be calibrated.** Static work units must never be labeled milliseconds.
4. **Unsupported operators stay out of the executable IR.** Add an operator only with its executor, trace representation, and correctness tests.
5. **Filtered HNSW names must be precise.** ANN overfetch followed by post-filtering is `HnswPostFilter`; filter-aware graph traversal is `FilteredHnsw`. They are different algorithms.
6. **Planning is reproducible.** Given the same planner configuration, immutable state/statistics snapshot, frozen calibration snapshot, query features, and query representations, planning produces the same plan.
7. **Statistics correspond to the searched generation.** A plan may not combine one generation's corpus statistics with another generation's documents or indexes.
8. **Parallelism is bounded globally.** Per-query channel parallelism must not destroy throughput at higher server concurrency.
9. **Manual planning remains supported.** Automatic planning is an additional interface, not a removal of caller control.
10. **Benchmark evidence follows `BENCHMARK_POLICY.md`.** Synthetic fixtures and development queries cannot support launch claims.

## Architecture and ownership

```text
crates/annex-multivector/src/
  engine.rs       storage, immutable state snapshots, mutations, ANN lifecycle
  planner.rs      plan IR, validation, statistics, physical optimization
  policy.rs       query representations, features, automatic channel policy
  retrieval.rs    channel execution, fusion, reranking, context selection, traces
```

`planner.rs`, `policy.rs`, and `retrieval.rs` are compiled as submodules of `engine.rs`. Keep their responsibilities separate even though they share private engine types.

Do not restore the removed `agent.rs`, `regret.rs`, `regret_bench.rs`, or runtime-calibration scaffolding until the corresponding phases below are implemented end to end.

## Current shipped behavior

### Request modes

`RetrieveRequest` supports:

```rust
pub enum PlanningMode {
    Manual,
    Auto,
    AutoWithOverrides,
}
```

- `Manual`: the caller supplies `prefetch` channels.
- `Auto`: the caller supplies `QueryRepresentations`; the policy generates channels.
- `AutoWithOverrides`: generated channels are merged with caller channels. BM25 overrides BM25; other channel overrides match by field.

An automatic request without `query` is invalid. A manual request without channels is invalid.

### Query representations and features

The current API accepts:

```rust
pub struct QueryRepresentations {
    pub text: Option<String>,
    pub dense: BTreeMap<String, Vector>,
    pub sparse: BTreeMap<String, SparseVector>,
    pub multivector: BTreeMap<String, Vec<Vector>>,
    pub fde: Option<Vec<Vector>>,
}
```

The current `QueryFeatures` records:

- token count;
- identifier fraction;
- numeric fraction;
- quoted phrase count;
- availability of dense, sparse, multivector, and FDE representations.

These features are exposed in `PolicyPlan`, but most are not yet used for routing. Do not describe the current policy as learned or fully query-adaptive.

### Automatic policy v1

The implemented deterministic policy:

- requires at least 50% corpus coverage for generated BM25, dense, named multivector, and FDE channels;
- selects sparse channels when the named field exists;
- falls back to BM25 when text exists but no channel meets the normal coverage rule;
- assigns candidate limits and `ef_search` from `QualityPreference`:

| Quality | Candidate limit | `ef_search` |
|---|---:|---:|
| Fast | `max(2 × result_limit, 10)` | 64 |
| Balanced | `max(5 × result_limit, 20)` | 256 |
| High | `max(10 × result_limit, 50)` | 1024 |

BM25 coverage uses documents that actually contain text. FDE coverage uses documents containing token vectors. Named representations use per-field coverage.

### Plan IR

The executable IR currently contains only operators the executor supports:

```rust
pub enum PhysicalOperator {
    Bm25,
    SparseDot,
    ExactDense,
    HnswDense,
    ExactMaxsim,
    ExactFde,
    HnswFde,
}

#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum PlanStage {
    Parallel(Vec<PlannedChannel>),
    Fusion(FusionOperator),
    Rerank(RerankPlan),
    Context(ContextPlan),
}
```

The adjacent Serde tag is part of the HTTP trace contract. Internally tagged tuple variants do not serialize and must not be restored.

`PlanReason` is machine-readable. The current reasons cover lexical/sparse access paths, explicit exact requests, graph availability, filters requiring exact execution, exact-only operators, and lower estimated cost.

### Statistics

`PlannerStats` currently contains:

- generation;
- total documents;
- documents containing text;
- documents containing token vectors;
- total token vectors;
- FDE output dimension and graph readiness;
- per-field representation kind, dimension, document coverage, and graph readiness;
- exact filter selectivity when a filter is present.

Current filtered planning computes the eligible set by scanning document metadata. Therefore current selectivity is exact but costs O(N); it is not a bitmap estimate.

### Physical selection

For unfiltered dense and FDE channels with a current ANN graph, the planner compares relative work estimates for exact and HNSW execution. The initial formulas are:

```text
ExactDense = eligible_docs × dimension × 2
HnswDense  = ef_search × ceil(log2(corpus_docs)) × dimension × 3
ExactFde   = eligible_docs × fde_dimension × 2
HnswFde    = ef_search × ceil(log2(corpus_docs)) × fde_dimension × 3
```

BM25, sparse, and MaxSim use separate formulas. These are relative work units for comparing plans; they are not milliseconds.

All filtered dense and FDE queries currently plan exact execution with `FilterRequiresExact`. This is intentional until Phase 2B has a real executor.

### Execution

The executor:

1. captures one immutable `Arc<State>` snapshot;
2. prepares the effective request once;
3. compiles a plan against that snapshot;
4. executes planned channels concurrently with Rayon;
5. fuses channel outputs;
6. optionally reranks with exact MaxSim;
7. applies context constraints;
8. returns the plan and observations in `RetrievalTrace`.

The channel executor must dispatch by `PhysicalOperator`, not repeat heuristic backend selection from the request.

### Plan estimates

```rust
pub struct PlanEstimate {
    pub critical_path_cost: f64,
    pub total_cost: f64,
}
```

- `critical_path_cost` is the largest parallel channel cost plus serial fusion, rerank, and context costs.
- `total_cost` is the sum of parallel channel work plus serial work.

Neither value is latency. `latency_budget_ms` is currently rejected because no calibrated planner can honor it.

### Context optimizer

The current context stage supports:

- final result limit;
- per-parent limits;
- neighbor expansion;
- exact text-hash deduplication;
- MMR;
- an optional diversity field;
- approximate token budgets.

Token packing estimates `word_count × 1.3`. A candidate that does not fit is skipped while packing continues, so one large chunk does not block smaller later chunks.

`RankingSignals` currently reports top-one margin, top-k score spread, channel agreement, source diversity, and deduplication count. These are descriptive signals, not a calibrated evidence-confidence score.

### EXPLAIN and trace behavior

- `MultiVectorIndex::plan()` is EXPLAIN: compile without retrieval execution.
- `MultiVectorIndex::retrieve()` is EXPLAIN ANALYZE: execute and embed the same plan in the response trace.
- `per_stage_actual_ms` aligns one-to-one with `plan.stages`.
- Channel trace backends must equal the corresponding planned operator.

The following should always hold for a fixed snapshot:

```text
retrieve(request).trace.plan == plan(request)
```

apart from a caller explicitly planning and executing against different generations.

## Target architecture

The following phases are ordered by dependency and implementation risk. Complete them in order. A later phase must not be scaffolded into production code merely because its types are easy to write.

## Phase 2A: metadata access paths and cardinality

This is the next implementation chunk.

### Objective

Replace repeated O(N) JSON metadata scans for common filters with generation-consistent indexes that produce an eligible document set and exact cardinality.

### Initial scope

Support:

- scalar equality;
- scalar `In`;
- existence;
- boolean `And`, `Or`, and `Not` over supported indexed predicates.

Leave range predicates, arbitrary JSON paths, and unsupported value types on the metadata-scan path until an ordered index is designed.

### Proposed representation

```rust
struct MetadataIndex {
    equality: HashMap<FieldName, HashMap<ScalarKey, RoaringBitmap>>,
    existence: HashMap<FieldName, RoaringBitmap>,
    live_documents: RoaringBitmap,
}
```

Use numeric internal document IDs. Define canonical scalar keys so JSON numbers, strings, booleans, and null do not collide. Do not key a hash map directly by unrestricted `serde_json::Value` without a stable equality/hash policy.

### Mutation and generation rules

- Build index entries as part of the same staged mutation as document fields.
- Upsert removes old metadata entries before adding the new version.
- Delete removes the document from all metadata postings and the live bitmap.
- Failed pre-commit mutations leave the metadata index unchanged.
- The immutable query `State` owns the matching metadata-index snapshot.
- Reopen and compaction reproduce identical postings.

### Planner changes

Add explicit filter plans:

```rust
pub enum FilterStrategy {
    None,
    MetadataScan,
    Bitmap,
    BitmapWithResidualScan,
}
```

`BitmapWithResidualScan` intersects indexed predicates first, then evaluates unsupported residual predicates only against surviving documents.

`FilterStats` should report:

- strategy;
- estimated cardinality before execution;
- selectivity;
- indexed and residual predicate counts.

If bitmap cardinality is exact, name it `cardinality`, not `estimated_cardinality`.

### Acceptance tests

- bitmap results equal the scalar predicate oracle for all supported boolean combinations;
- upsert, overwrite, delete, batch failure, reopen, and compaction preserve equality;
- filters on missing fields and mixed scalar types behave consistently;
- planning never reads index statistics from a different generation;
- residual evaluation runs only over bitmap survivors;
- benchmark demonstrates reduced filter work without changing ranked results.

## Phase 2B: HNSW post-filter with exact fallback

### Semantic definition

`HnswPostFilter` means:

1. search the unfiltered HNSW graph for an over-fetched candidate set;
2. discard candidates outside the eligible bitmap;
3. if too few eligible results remain, run `ExactEligibleScan`;
4. return the correct top-k from the executed path and record fallback.

It does not mean filter-aware traversal and must not be named `FilteredHnsw`.

### Plan representation

Only add these operators when execution exists:

```rust
HnswPostFilter,
ExactEligibleScan,
```

A planned post-filter channel needs observable parameters:

- requested `k`;
- candidate limit;
- `ef_search`;
- selectivity/cardinality input;
- overfetch factor;
- exact-fallback threshold.

### Initial policy

Use tunable `PlannerConfig` values, not hard-coded architectural constants. A reasonable development starting point is:

| Selectivity | Candidate |
|---|---|
| `< 0.05` | exact eligible scan |
| `0.05 .. 0.25` | compare exact with post-filter HNSW using capped `1/s` amplification |
| `>= 0.25` | compare exact with mildly over-fetched post-filter HNSW |

The cost comparison decides; thresholds only bound candidates. Validate on held-out data before making them defaults.

### Correctness

The independence assumption behind `k/selectivity` can fail when filter membership is correlated with vector neighborhoods. The exact fallback is mandatory. Trace both the initial ANN attempt and fallback outcome.

### Acceptance tests

- adversarial correlated filters trigger fallback and match exact results;
- non-correlated filters retain at least the declared recall target;
- deleted and overwritten documents never leak through either path;
- the plan and trace identify the same initial and fallback operators;
- exact work is not silently performed while the trace reports HNSW.

## Phase 2C: runtime calibration and adaptive `ef_search`

### Objective

Convert relative work estimates into advisory time estimates using observations collected from real operator executions.

### Observation contract

Every operator observation should record:

```text
operator class
state generation or corpus fingerprint
estimated work
actual wall time
actual candidates examined/returned
estimated and actual filter cardinality
fallback/escalation outcome
```

Keep raw per-query observations available for offline evaluation. Online aggregates are not a substitute for replay data.

### Operator classes

```rust
pub struct OperatorClass {
    pub operator: PhysicalOperator,
    pub dimension_bucket: usize,
    pub corpus_bucket: usize,
    pub filter_selectivity_bucket: Option<u8>,
    pub architecture: ArchitectureClass,
}
```

Include representation/encoding identity where it materially changes performance. Calibration learned on NEON must not silently drive AVX2/AVX-512 planning.

### Aggregation

Calibration is advisory until a class has a minimum sample count. Start with EWMA mean and EWMA deviation. Do not label an estimate p90 unless using a real quantile sketch such as DDSketch/t-digest or a bounded histogram with documented error.

A frozen `CalibrationSnapshot` is an input to planning and benchmark replay. Online updates produce a new snapshot; they must not mutate the inputs of a plan already being compiled.

### Latency objectives

Only after calibration is trustworthy should `latency_budget_ms` be accepted. Budget allocation subtracts estimated serial work, then assigns the remaining budget across parallel channels while accounting for the critical path and global concurrency limits.

Adaptive `ef_search` must be bounded and must retain an exact or high-ef oracle path for evaluation.

### Acceptance tests

- insufficient observations always use static work comparisons;
- frozen snapshots produce deterministic plans;
- observations are separated by operator/dimension/corpus/architecture class;
- pathological early samples cannot abruptly flip production defaults;
- estimates and actuals appear side by side in EXPLAIN ANALYZE;
- held-out budget violation rate improves versus static defaults.

## Phase 2D: conditional execution

### Objective

Run a cheap initial plan, inspect explicit signals, and execute an additional planned stage only when its predicate fires.

### IR

```rust
pub enum PlanStage {
    Parallel(Vec<PlannedChannel>),
    Fusion(FusionOperator),
    ConditionalEscalation(ConditionalStage),
    Rerank(RerankPlan),
    Context(ContextPlan),
}

pub struct ConditionalStage {
    pub predicate: EscalationPredicate,
    pub additional_channels: Vec<PlannedChannel>,
    pub re_fuse: bool,
}
```

Initial predicates may include:

- insufficient eligible hits;
- low cross-channel agreement;
- small normalized score margin, only for a score family where the threshold is calibrated.

The trace records predicate inputs, threshold, outcome, added work, and re-fusion time. A boolean `escalated` without inputs is insufficient.

### Acceptance tests

- false predicates execute no hidden work;
- true predicates run exactly the additional planned channels;
- stage timings and traces include both predicate evaluation and added work;
- plans serialize through the HTTP API;
- global parallelism limits apply across initial and escalation stages.

## Phase 3B: genuinely query-adaptive policy

The current policy chooses available representations by corpus coverage. The next policy version should use query properties to decide which representations are useful.

### Additional corpus statistics

Add only statistics that power a concrete decision:

- document frequency and vocabulary size for lexical specificity;
- average posting length;
- representation coverage and model/version identity;
- historical per-channel utility from held-out evaluation, not self-reinforcing production choices.

### Candidate query signals

- query length;
- rare-term or IDF mass;
- identifier, code, numeric, and quoted-phrase fractions;
- available query representations;
- filter cardinality;
- multivector token count;
- explicit quality and resource constraints.

### Initial deterministic policy

Examples to validate rather than assume:

- identifier/error-code queries may favor BM25 plus dense;
- semantic natural-language queries with good dense coverage may favor dense;
- long, multifaceted high-quality queries may add late interaction;
- sparse channels should run only when both query representation and matching corpus field exist;
- single-channel plans use native scores; multi-channel plans use a declared fusion method.

Selection reasons should become structured variants. Human EXPLAIN text should be generated from those variants rather than stored as the only representation.

### Candidate budgets

Separate:

```rust
pub struct BudgetPlan {
    pub retrieval_candidates: BTreeMap<ChannelId, usize>,
    pub rerank_depth: usize,
    pub context_limit: usize,
    pub context_token_budget: Option<usize>,
}
```

Do not overload final result limit as candidate depth.

### Learned policy boundary

A learned policy may eventually predict quality and latency for feasible plans:

```text
(query features, corpus features, plan features)
    -> predicted quality, predicted latency
```

Then choose maximum predicted quality subject to a resource budget. Start with interpretable models and frozen offline training/evaluation. Do not add online learning before counterfactual data collection and selection-bias controls exist.

## Phase 4B: context quality

The current context optimizer handles structural limits, exact deduplication, MMR, neighbors, and approximate token packing. Further work should focus on measurable evidence quality:

- tokenizer-specific token estimation;
- semantic near-duplicate detection;
- coverage of distinct source documents and topics;
- retention of conflicting evidence;
- parent/neighbor expansion chosen by policy rather than a fixed request;
- calibrated evidence-sufficiency estimation.

Do not collapse heterogeneous BM25, RRF, cosine, and MaxSim scores into a universal confidence scalar without calibration.

## Phase 5: iterative evidence planner

Implement only after Phases 2–4 have reliable plans, traces, and offline evaluation.

This planner may take ANNex-native actions that do not require language understanding:

- widen candidate depth;
- activate an available representation;
- expand rerank depth;
- expand neighbors;
- increase source diversity;
- stop based on calibrated marginal gain or budget exhaustion.

A separate external agent may perform semantic reformulation, decomposition, collection selection, and reference traversal.

Strategy memory should be keyed by a versioned corpus fingerprint, not generation:

```rust
pub struct CorpusFingerprint {
    pub schema_hash: u64,
    pub config_hash: u64,
    pub representation_versions_hash: u64,
    pub analyzer_version: u32,
    pub document_count_bucket: usize,
}
```

A cached strategy is a prior. It is invalidated when observations diverge materially; it is never declared successful using its own confidence signal.

## Global execution and concurrency

Before increasing automatic channel count, introduce an executor budget shared across queries.

Requirements:

- cap concurrent channel tasks per request;
- cap total retrieval work across requests;
- avoid nested unbounded Rayon parallelism;
- preserve snapshot isolation;
- measure scheduled-arrival tail latency under offered load, not only closed-loop QPS;
- expose queueing, operator time, and total wall time separately.

Possible designs include a dedicated query-level pool plus operator-local serial/parallel thresholds, or a semaphore around channel work. Choose from measurements.

## Offline plan enumerator and regret evaluation

Build an offline tool before sophisticated heuristics. It takes stored queries and executes a declared finite plan set, for example:

```text
dense exact
dense HNSW ef={64,128,256,512}
BM25
sparse
BM25 + dense
BM25 + dense + exact MaxSim rerank depths={20,50,100}
```

For every query and plan, persist:

```text
query ID
plan ID and serialized plan
corpus/config/model fingerprints
quality metrics
wall latency and operator timings
CPU work, memory, disk access where available
failures/timeouts
raw ranked IDs and scores
```

Construct the empirical quality/latency Pareto frontier. For each declared budget, compare the planner's choice with the highest-quality observed feasible plan.

Primary planner metrics:

- quality regret: `oracle_quality - chosen_quality`;
- epsilon-Pareto hit rate;
- budget violation rate;
- p95 budget overrun magnitude;
- planning overhead;
- plan stability under a frozen snapshot.

Do not restore the deleted synthetic regret benchmark. The replacement must run real stored queries, real operators, and a predeclared plan grid.

## Benchmark and launch protocol

`BENCHMARK_POLICY.md` is authoritative. Planner-specific additions:

1. Define development and untouched test queries before tuning.
2. Freeze planner configuration, calibration snapshot, plan grid, metrics, and resource limits before the held-out run.
3. Preserve every plan/query result, including failures and losses.
4. Compare identical encoders, representations, filters, and relevance judgments when making engine claims.
5. Report ingestion, index build/readiness, recovery, memory, disk, CPU, latency, throughput, and quality separately.
6. Evaluate single-query latency and offered-load concurrency.
7. Include mixed reads, updates, deletes, compaction, and ANN maintenance.
8. Publish raw immutable artifacts outside the source tree with hashes and reproduction commands.
9. Require at least one real 1M+ document/chunk workload for production performance claims.
10. Treat small local runs as correctness and development evidence only.

The shared ANN harness lives in:

```text
scripts/run_annex_benchmark.sh
scripts/evaluate_ann_benchmark.py
crates/annex-core/bench/{glove100,lastfm64,mnist784,nyt256,sift1m}/run_all.sh
```

RAG-quality infrastructure lives under `crates/annex-multivector/benchmark/`. Keep ANN kernel/index comparisons separate from end-to-end RAG system comparisons.

## Implementation sequence

Resume in this order:

### Chunk 1: metadata indexes

- add bitmap equality/existence index to immutable state;
- integrate atomic mutations, reopen, and compaction;
- compile supported filters to bitmap plans with residual fallback;
- verify against the scalar predicate oracle;
- measure filter work and end-to-end latency.

### Chunk 2: filtered physical choices

- add exact-eligible execution as an explicit operator;
- add HNSW post-filter overfetch and exact fallback;
- add structured fallback tracing;
- validate adversarial correlations and ordinary held-out filters.

### Chunk 3: calibration

- define operator observations and frozen snapshots;
- collect actual work/timing without changing planning;
- validate stability and architecture stratification;
- enable advisory estimates;
- only then enable latency budgets and adaptive `ef_search`.

### Chunk 4: conditional execution

- add executable conditional stages;
- trace inputs and outcomes;
- add global parallelism limits;
- evaluate under concurrency.

### Chunk 5: adaptive policy and oracle benchmark

- build the real plan enumerator;
- add query/corpus features that improve held-out plan choice;
- report regret and budget adherence;
- consider a learned policy only after deterministic baselines are strong.

### Chunk 6: context and iterative planning

- improve evidence coverage and token accounting;
- calibrate sufficiency signals;
- add ANNex-native iterative actions;
- keep LLM semantic planning outside the storage/query engine boundary.

## Testing requirements

Prefer invariant and state-machine tests over matrices of near-duplicates.

Required categories:

- plan determinism for a frozen snapshot;
- plan/executor/trace operator equality;
- plan HTTP serialization;
- exact oracle equality for filters and fallback paths;
- mutation, crash, reopen, and compaction consistency;
- no stale ANN or metadata-index results after overwrite/delete;
- bounded concurrency and snapshot isolation;
- held-out quality and performance evaluation.

Avoid tests that merely restate an enum mapping or duplicate one implementation with the same algorithm. Integration tests should cover the public HTTP and Python surfaces that unit tests cannot exercise.

## Current validation and known debt

At the time of this handoff:

- PR 14 is mergeable and CI passes on Ubuntu and macOS;
- workspace Rust tests pass;
- multivector strict Clippy passes;
- benchmark compilation passes on Ubuntu and macOS;
- HTTP/Python integration tests pass in CI;
- the branch is net smaller than `master` despite the planner implementation.

Workspace-wide strict Clippy still reports existing warnings in `annex-core`, concentrated in HNSW internals. Address them in focused, behavior-preserving work rather than suppressing warnings or mixing a broad HNSW rewrite into the planner.

Large local `data/`, `logs/`, and `target/` directories are ignored benchmark/build artifacts, not source-tree contents. Do not commit them. Do not delete them without confirming they are no longer needed for local benchmark reproduction.

## Handoff checklist for the next implementer

Before changing code:

```sh
git switch feat/query-planner
git pull --ff-only
cargo test --workspace
gh pr checks 14
```

Then:

1. Read this file and `BENCHMARK_POLICY.md`.
2. Inspect the current `planner.rs`, `policy.rs`, and `retrieval.rs`; code is authoritative when this document and implementation differ.
3. Start with Phase 2A metadata access paths.
4. Keep each commit executable and honestly traced.
5. Do not add future operator variants before their executor exists.
6. Run public HTTP/Python integration tests whenever serialized plan or trace types change.
7. Update this document when a phase becomes implemented or its contract changes.
8. Keep generated benchmark results outside the source tree.

Useful validation commands:

```sh
cargo fmt --all -- --check
cargo test --workspace
cargo clippy -p annex-multivector --all-targets --no-deps -- -D warnings
python3 -m unittest discover -s crates/annex-multivector/benchmark -p 'test_*.py'
bash -n scripts/run_annex_benchmark.sh crates/annex-core/bench/*/run_all.sh
git diff --check
```

The design is successful when ANNex can demonstrate, on held-out real RAG workloads, that its automatically selected plan stays near the empirical quality/latency Pareto frontier while preserving correctness, predictable resource use, and truthful EXPLAIN output.
