# ANNex Five-Phase Query Planner Design

**Date:** 2026-09-30  
**Branch:** feat/query-planner  
**Status:** Revision 2 — awaiting approval

---

## Goal

ANNex compiles an information need into the cheapest retrieval and context plan that satisfies the application's quality requirements. The planner optimises along two independent axes:

- **Estimated execution cost**: CPU work, wall time, memory.
- **Estimated retrieval utility**: expected quality contribution of each operator given corpus and query features.

These must be tracked separately. A planner that conflates them will make wrong decisions when adding a cheap-but-useless channel or dropping an expensive-but-critical one.

---

## Current State (baseline)

- `MultiVectorIndex.retrieve()` executes channels sequentially; binary exact/HNSW decisions via heuristic (`if filtered → exact`).
- Working-tree additions: `LogicalPlan`, `PlannerStats`, `FieldStats`, `RetrievalObjective`, `QualityPreference` in `planner.rs`; `objective` on `RetrieveRequest`.
- `RetrievalTrace` already captures per-channel timing and operator selected.
- Filter path does a full document metadata scan — O(N) — to compute `eligible_documents`.

---

## Key Concept Separation

| Concept | Meaning | Lives in |
|---------|---------|----------|
| `PlannerStats` | Facts about the corpus and index state | Cached on `State` |
| `PlannerConfig` | Tunable thresholds and policy settings | Constant on `MultiVectorIndex` |
| `CalibrationStats` | Observed runtime performance by operator class | Atomic accumulators on `MultiVectorIndex` |
| `QueryFeatures` | Facts derived from the current query | Computed per-request in `policy.rs` |

Do not conflate these. In particular, `PlannerStats` should contain only corpus facts; calibration data belongs in `CalibrationStats`.

---

## Module Structure

```
crates/annex-multivector/src/
  engine.rs         — MultiVectorIndex: storage, ANN, upsert/delete/query
    planner.rs      — Phase 1–2: physical optimizer, cost model, plan IR
    policy.rs       — Phase 3: retrieval-policy optimizer [NEW]
    retrieval.rs    — Phase 4: context optimizer, retrieve(), all request types
    agent.rs        — Phase 5: iterative evidence planner [NEW]
```

---

## Plan IR

Replace the flat `Vec<PlannedChannel>` with a staged plan that can express conditional execution. Used from Phase 1 onward.

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

pub enum EscalationPredicate {
    LowChannelAgreement { threshold: f32 },
    InsufficientEligibleHits { min_count: usize },
    LowTopScore { threshold: f32 },
}

pub struct RetrievalPlan {
    pub logical: LogicalPlan,
    pub stats: PlannerStats,
    pub eligible_documents: usize,
    pub filter: FilterStrategy,
    pub stages: Vec<PlanStage>,
    pub estimate: PlanEstimate,
}

pub struct PlanEstimate {
    pub estimated_wall_ms: f64,
    pub estimated_cpu_work: f64,       // sum of operator work units (not ms)
    pub estimated_peak_memory_bytes: Option<u64>,
}
```

Wall time reflects the parallel critical path (`max` of concurrent channel costs plus serial stages). CPU work reflects aggregate operator effort (useful for understanding resource pressure under concurrent load).

---

## Phase 1 — Physical Optimizer Foundation

### PlannerStats caching

Cache `PlannerStats` on `State` as `stats_cache: Option<PlannerStats>`. Invalidated on every generation bump. Populated lazily on the first `plan()` or `retrieve()` call per generation, then reused.

**Why this matters for the planner:** Without this, computing selectivity estimates requires a full document scan on every `plan()` call. With caching, selectivity estimates are O(1) after the first call per generation.

### Operator taxonomy

```rust
pub enum PhysicalOperator {
    Bm25,
    SparseDot,
    ExactDense,
    HnswDense,
    ExactMaxsim,
    ExactFde,
    HnswFde,
    HnswPostFilter,   // HNSW overfetch + post-filter + exact fallback
    ExactEligibleScan, // filter-first exact scan over eligible set
    FilterBitmap,     // future: bitmap intersection (Phase 2A)
}
```

**`HnswPostFilter` semantics:** Fetch `ceil(k / estimated_selectivity) × safety_factor` ANN results, discard ineligible, fall back to `ExactEligibleScan` if eligible hits < k. This is **not** filter-aware graph traversal (that requires changes to the HNSW library itself and is labelled `FilteredHnsw` for a future phase).

### Per-operator cost functions

Cost is operator-specific. Use these as initial calibration values, overridden by `CalibrationStats` once available:

| Operator | Cost formula |
|----------|-------------|
| `ExactFde` | `F × fde_dimension × 2` (dot product per eligible doc) |
| `HnswFde` | `ef × ceil(log2(N)) × fde_dimension × 3` |
| `HnswPostFilter` | `ceil(k/s) × ef × ceil(log2(N)) × fde_dim × 3` + possible fallback |
| `ExactDense` | `F × dim × 2` |
| `HnswDense` | `ef × ceil(log2(N)) × dim × 3` |
| `ExactMaxsim` | `C × T × dim × 2` (C = candidates, T = avg token count) |
| `Bm25` | `posting_size × 8` (approximate postings traversal) |
| `SparseDot` | `nnz_query × nnz_avg_doc × 4` |
| `FilterScan` | `N × 40` (metadata JSON scan, bytes estimate) |
| `FilterBitmap` | `k × 4` (bitmap intersection, Phase 2A+) |

Wall time estimate: `cost_units / calibrated_throughput_units_per_ms`.

For concurrent stages: `wall_ms = max(channel_wall_ms) + serial_stage_ms`.

### Unfiltered exact vs HNSW selection

```
choose HnswFde/HnswDense when:
  - ANN index is current-generation
  - No filter present
  - cost_hnsw(ef, N) < cost_exact(F)

otherwise: ExactFde / ExactDense / ExactEligibleScan
```

For Phase 1, filtered queries always use `ExactEligibleScan`. Post-filter HNSW is Phase 2B.

### Concurrent channel execution

Dispatch all `PlanStage::Parallel` channels with `rayon::scope`. One scope task per channel; all share the same `Arc<State>` snapshot. All tasks complete before Fusion proceeds.

**Implementation constraint:** Each channel may itself use Rayon internally (e.g., `exact_fde_scores_filtered` uses `par_iter`). Limit total intra-query parallelism to avoid contention under concurrent server load. Use `rayon::ThreadPoolBuilder` to separate the query-parallelism pool from intra-operator pools, or document the interaction.

Behavior specified: independent retrieval branches execute concurrently subject to a bounded query-parallelism budget. The specific scheduling mechanism (Rayon scope, tokio spawn, etc.) is an implementation detail.

### EXPLAIN / EXPLAIN ANALYZE surface

- `plan()` → returns `RetrievalPlan` (dry-run, no execution) — this is EXPLAIN.
- `retrieve()` → returns `RetrievalResponse` containing `RetrievalTrace` with `plan` embedded — this is EXPLAIN ANALYZE.
- Add `per_stage_actual_ms: Vec<f64>` to `RetrievalTrace` aligned with `plan.stages`.
- No new API endpoints needed.

---

## Phase 2 — Advanced Physical Execution

### 2A: Categorical metadata index

Add an `EqualityIndex` struct mapping `(field, value) → BitSet<doc_id>`. Built incrementally on upsert/delete. Supported predicate types: `Eq`, `In`, boolean `And`/`Or`/`Not` over equality predicates.

```rust
struct MetadataIndex {
    equality: HashMap<String, HashMap<Value, RoaringBitmap>>,
    existence: HashMap<String, RoaringBitmap>,
    doc_count: usize,
}
```

Benefits:
1. `eligible_documents` computed via bitmap intersection in O(log N) instead of O(N).
2. Exact cardinality for selectivity estimates without scanning docs.
3. Enables `FilterBitmap` operator (future: replace `FilterScan`).

`PlannerStats` gains:
```rust
pub filter_stats: Option<FilterStats>,

pub struct FilterStats {
    pub indexed_fields: Vec<String>,
    pub estimated_selectivity: f32,   // from bitmap cardinality if indexed
    pub filter_operator: FilterStrategy,
}
```

Range predicates, complex JSON path predicates, and text predicates remain as `FilterScan` (deferred).

### 2B: HnswPostFilter with exact fallback

When filter selectivity `s = F/N`:

| Selectivity | Operator | ef multiplier |
|-------------|----------|--------------|
| `s < 0.05` | `ExactEligibleScan` | — |
| `0.05 ≤ s < 0.25` | `HnswPostFilter` | `min(1/s, 4×)` |
| `s ≥ 0.25` | `HnswPostFilter` | `1.5×` (mild overfetch) |

Use one consistent set of thresholds (not the conflicting values from the earlier draft). Treat these as `PlannerConfig` defaults, not architectural constants — they should be tunable.

`HnswPostFilter` execution:
1. Fetch `ceil(k × ef_multiplier)` ANN results.
2. Filter to eligible set.
3. If `eligible_hits < k` AND ANN was used: fall back to `ExactEligibleScan`.
4. Record fallback in trace (`escalated: bool` on `PlannedChannel`).

Assumption caveat (document in code): `k / selectivity` over-fetch assumes ANN result distribution is independent of filter membership. This can fail when filter-matching vectors occupy a different semantic region. The exact fallback mitigates the worst case.

### 2C: Calibrated ef_search and RuntimeStats

```rust
pub struct CalibrationStats {
    pub by_class: HashMap<OperatorClass, OperatorCalibration>,
}

pub struct OperatorClass {
    pub operator: PhysicalOperator,
    pub dimension_bucket: usize,   // e.g. 128, 256, 512, 1024, 2048+
    pub corpus_bucket: usize,      // e.g. 1K, 10K, 100K, 1M+
}

pub struct OperatorCalibration {
    pub mean_ms: f64,
    pub p90_ms: f64,
    pub observations: u64,
    // EWMA state
    pub ewma_ms: f64,
    pub ewma_alpha: f32,
}
```

Calibration updated after each `retrieve()` via lock-free atomics or a dedicated calibration mutex (separate from the writer lock).

Adaptive ef_search: when `latency_budget_ms` is set:
```
remaining_budget = budget_ms - filter_scan_ms_estimate - fusion_ms_estimate
per_channel_budget = remaining_budget / num_channels
ef_adaptive = clamp(per_channel_budget / calibrated_ns_per_ef_step × 1e6, 16, 65536)
```

Bench notes: the benchmark harness freezes `CalibrationStats` at the start of an evaluation run (`CalibrationSnapshot`) so cost-model comparisons are reproducible.

### 2D: Conditional escalation

Add `ConditionalEscalation(ConditionalStage)` as a plan stage. The executor evaluates the predicate after initial channels complete and, if triggered, runs `additional_channels` as a new parallel stage before re-fusing.

Predicates:
- `LowChannelAgreement`: Jaccard overlap of top-k across channels < threshold.
- `InsufficientEligibleHits`: Filtered result count < k.
- `LowTopScore`: Top-1 score below an absolute threshold.

Escalation channels are chosen by the physical optimizer (typically: switch from HNSW to exact, or add a MaxSim channel). Record triggered escalations in `RetrievalTrace.escalations: Vec<usize>` (stage indices).

---

## Phase 3 — Retrieval-Policy Optimizer

New file: `policy.rs`

### API additions to RetrieveRequest

```rust
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanningMode {
    #[default]
    Manual,
    Auto,
    AutoWithOverrides,
}

pub struct QueryRepresentations {
    pub text: Option<String>,
    pub dense: HashMap<String, Vec<f32>>,
    pub sparse: HashMap<String, SparseVector>,
    pub multivector: HashMap<String, Vec<Vec<f32>>>,
    pub fde: Option<Vec<Vec<f32>>>,  // unnamed late-interaction
}
```

`RetrieveRequest` gains:
```rust
pub planning_mode: PlanningMode,  // default Manual
pub query: Option<QueryRepresentations>,  // required when planning_mode != Manual
```

When `planning_mode == Auto`:
- `query` must be `Some` (validator enforces this).
- `prefetch` must be empty (or is ignored).
- Policy planner intersects available representations in `query` with corpus coverage from `PlannerStats`, selects channels, generates `prefetch`, then passes to `compile_plan`.
- When `planning_mode == AutoWithOverrides`: provided `prefetch` hints are merged with the policy-generated list (explicit channels take precedence, auto-fills the rest).

### QueryFeatures

```rust
pub struct QueryFeatures {
    pub token_count: usize,
    pub rare_term_score: f32,         // fraction of query tokens not in top-10k vocabulary
    pub identifier_fraction: f32,     // fraction of tokens matching identifier pattern
    pub numeric_fraction: f32,
    pub quoted_phrase_count: usize,
    pub lexical_specificity: f32,     // TF-IDF-like score over corpus vocabulary
    pub has_dense: bool,
    pub has_sparse: bool,
    pub has_multivector: bool,
}
```

Computed from `QueryRepresentations` + `PlannerStats.vocabulary` if available. Used by the policy heuristic to route.

### Policy heuristic (v1)

Channel selection considers both corpus coverage AND query features:

| Query signal | Corpus condition | Selected channels |
|---|---|---|
| High lexical specificity / identifiers | BM25 coverage ≥ 50% | BM25-heavy (high limit) |
| Semantic question, no dense available | — | BM25 only |
| Semantic question, dense available | Dense coverage ≥ 50% | Dense primary |
| Mixed signal | Both available | BM25 + Dense, RRF |
| Long/multi-faceted + High quality | FDE coverage ≥ 50% | BM25 + Dense + FDE/MaxSim |
| Sparse representation present | Sparse index non-empty | Add Sparse channel |

Candidate limits by `QualityPreference`:
- `Fast`: `2 × limit`, ef_search 64
- `Balanced`: `5 × limit`, ef_search 256
- `High`: `10 × limit`, ef_search 1024

Fusion: RRF (k=10) for ≥2 channels; native score for single channel.  
Rerank: auto-added when FDE channel present AND `quality == High`.

### Candidate, rerank, and context budgets

Three distinct quantities — do not overload `result_limit`:

```rust
pub struct BudgetPlan {
    pub retrieval_candidates: usize,  // per-channel candidate limit
    pub rerank_depth: usize,          // candidates fed to reranker
    pub context_limit: usize,         // final hits returned
    pub context_token_budget: Option<usize>,
}
```

The policy planner populates `BudgetPlan`; the physical optimizer uses it to set per-channel limits. Context packing uses `context_token_budget` as a stopping guide.

### PolicyPlan (embedded in trace)

```rust
pub struct PolicyPlan {
    pub channels_selected: Vec<LogicalChannelKind>,
    pub selection_reasons: Vec<String>,
    pub query_features: QueryFeatures,
    pub budget: BudgetPlan,
    pub generated_prefetch: Vec<Channel>,
}
```

Embedded in `RetrievalTrace.policy: Option<PolicyPlan>` when `planning_mode != Manual`.

---

## Phase 4 — Context Optimizer

Lives in `retrieval.rs`.

### Token-aware packing

`context_budget_tokens` (on `RetrievalObjective`) is now enforced as a packing constraint, not a stopping-on-first-miss constraint.

Algorithm: iterate candidates in ranked order. For each candidate, estimate its token cost (`word_count × 1.3`). If it fits in remaining budget, include it; otherwise skip it and continue (do not stop). This is bin-packing, not first-fit-decreasing, so large chunks don't block small ones.

Add `estimated_tokens: Option<usize>` to `ContextHit`.

Future: `TokenEstimator::Approximate` (word count) vs `TokenEstimator::ModelTokenizer(...)` as a configurable enum.

### RankingSignals (replaces evidence_confidence)

The previously proposed `evidence_confidence` formula (`1 - median/top1`) is internally contradictory with its description and produces a metric that isn't interpretable across score families (BM25, RRF, cosine, MaxSim have different scales and zero-behaviors).

Replace with descriptive signals:

```rust
pub struct RankingSignals {
    pub top1_margin: f32,        // top1_score - top2_score
    pub topk_score_spread: f32,  // top1_score - topk_score
    pub channel_agreement: f32,  // already in RetrievalTrace
    pub unique_sources: usize,   // distinct parent documents
    pub source_diversity: f32,   // unique_sources / matches.len()
    pub dedup_count: usize,      // candidates dropped by dedup
}
```

Add `signals: RankingSignals` to `RetrievalTrace`. Phase 5's stopping criterion uses `signals` rather than a single calibrated scalar. Calibration (training a `P(sufficient evidence | signals)` model) is future work.

### Dedup instrumentation

The `deduplicate: bool` flag already hashes text. This phase adds `dedup_count` to `RankingSignals` to make it visible. This is **dedup instrumentation**, not semantic deduplication. Near-duplicate detection (embedding similarity threshold) is future work.

---

## Phase 5 — Iterative Evidence Planner

New file: `agent.rs`. Renamed from "Agentic Planner" to reflect that Phase 5 only performs actions ANNex can take without LLM or external model integration. Semantic query reformulation and decomposition requiring NLU are a separate future layer.

### AgentSearch

```rust
pub struct AgentSearch {
    pub objective: RetrievalObjective,
    pub query: QueryRepresentations,
    pub prior_results: Vec<ContextHit>,
    pub iteration: usize,
    pub total_tokens_consumed: usize,
    strategy_memory: HashMap<CorpusFingerprint, CorpusStrategy>,
}

pub struct CorpusFingerprint {
    pub schema_hash: u64,         // hash of field names + types
    pub config_hash: u64,         // hash of IndexConfig
    pub doc_count_bucket: usize,  // e.g. 1K, 10K, 100K, 1M+
}
```

**Why `CorpusFingerprint` instead of generation:** Every write creates a new generation. Keying by generation would expire strategy memory on every document insert. `CorpusFingerprint` is stable unless the schema, index config, or corpus size bucket changes.

**Strategy validity:** A cached `CorpusStrategy` is used as a prior, not as gospel. If `CalibrationStats` shows significant divergence from strategy predictions, the strategy is discarded. Do not mark a strategy "successful" because its own confidence metric was high (self-reinforcing loop). Strategy learning should eventually use externally validated outcomes.

### AgentDecision

```rust
pub enum StopReason {
    TokenBudgetExhausted,
    MarginalGainLow,         // < 10% new candidates vs prior iteration
    MaxIterationsReached,    // default: 5
    HighChannelAgreementAcrossIterations,
}

pub enum AgentDecision {
    Continue(AgentIteration),
    Stop(StopReason),
}

pub struct AgentIteration {
    pub request: RetrieveRequest,
    pub rationale: String,
}
```

### plan_next

```rust
impl AgentSearch {
    pub fn plan_next(
        &mut self,
        stats: &PlannerStats,
        last_trace: Option<&RetrievalTrace>,
    ) -> AgentDecision
}
```

Actions available (no LLM required):
- Increase candidate depth (widen retrieval)
- Activate an additional representation channel
- Widen numeric filter range
- Run MaxSim rerank on a larger candidate pool
- Expand context neighbors
- Increase source diversity target

Stopping uses `RankingSignals` from the last trace, not a single calibrated scalar (which Phase 5 doesn't have yet).

---

## API Summary — What Changes

### New/changed fields on existing types

| Type | Change | Phase |
|------|--------|-------|
| `RetrieveRequest` | `+ planning_mode: PlanningMode` | 3 |
| `RetrieveRequest` | `+ query: Option<QueryRepresentations>` | 3 |
| `RetrievalPlan` | `stages: Vec<PlanStage>` replaces `channels`/`fusion`/`rerank`/`context` | 1 |
| `RetrievalPlan` | `+ estimate: PlanEstimate` | 1 |
| `RetrievalTrace` | `+ per_stage_actual_ms: Vec<f64>` | 1 |
| `RetrievalTrace` | `+ escalations: Vec<usize>` | 2D |
| `RetrievalTrace` | `+ policy: Option<PolicyPlan>` | 3 |
| `RetrievalTrace` | `+ signals: RankingSignals` | 4 |
| `ContextHit` | `+ estimated_tokens: Option<usize>` | 4 |
| `PlannerStats` | `+ filter_stats: Option<FilterStats>` | 2A |
| `MultiVectorIndex` | `+ calibration: CalibrationStats` (atomic accumulators) | 2C |

### New types (re-exported from lib.rs)

`PlanningMode`, `QueryRepresentations`, `QueryFeatures`, `PolicyPlan`, `BudgetPlan`, `PlanEstimate`, `PlanStage`, `ConditionalStage`, `EscalationPredicate`, `RankingSignals`, `CalibrationStats`, `AgentSearch`, `AgentDecision`, `StopReason`

---

## Backward Compatibility

**Wire-format compatibility** (JSON serialization): preserved. All new fields on serialized types use `#[serde(default)]`. Existing serialized requests remain valid.

**Rust source compatibility**: may break callers constructing public structs directly (e.g. `RetrieveRequest { prefetch, filter, ... }` without `..Default::default()`). Adding public fields to non-`#[non_exhaustive]` structs is a compile-time breaking change for direct struct literal construction. If downstream Rust callers exist, now is the right time to move toward a builder pattern or mark public request types `#[non_exhaustive]`.

---

## Planner Regret Benchmark

Enumerate feasible plans for a query (operator subset × ef ∈ {64, 256, 1024}), measure quality/latency for each, build empirical Pareto frontier.

**Metrics:**

- **Quality regret** = `oracle_nDCG − chosen_nDCG` (positive when ANNex underperforms oracle)
- **ε-Pareto hit rate**: fraction of queries where chosen plan is within ε_q=1% quality and ε_l=5% latency of the empirical frontier (more stable than exact membership due to benchmark noise)
- **Budget violation rate** = fraction of queries where `actual_latency > latency_budget`
- **P95 budget overrun** = p95(`max(0, actual_latency / budget − 1)`)

The evaluation harness freezes `PlannerConfig` and `CalibrationSnapshot` at run start for reproducibility.

---

## Revised Implementation Sequence

**1A:** `PlannerStats` caching + `Plan IR` (`PlanStage`, `PlanEstimate`) + EXPLAIN surface.

**1B:** Concurrent channel execution + correct wall/work accounting.

**1C:** Cost-based exact vs HNSW for **unfiltered** queries.

**2A:** Categorical metadata bitmaps + `FilterStats` cardinality.

**2B:** `HnswPostFilter` operator with overfetch + exact fallback.

**2C:** `CalibrationStats` + adaptive `ef_search`.

**2D:** `ConditionalEscalation` plan stage.

**3A:** `QueryRepresentations` + `QueryFeatures` types + validator.

**3B:** `PlanningMode::Auto` policy routing + `PolicyPlan` trace.

**4:** Token-aware packing (skip-and-continue) + `RankingSignals`.

**5:** `AgentSearch::plan_next` + `CorpusFingerprint` strategy memory.

---

## Out of Scope

- LLM-based query decomposition and semantic reformulation (Phase 5 uses only actions ANNex can take natively)
- Genuinely filter-aware HNSW graph traversal (labelled `FilteredHnsw`, requires HNSW library changes)
- Range and text metadata indexes (categorical equality bitmaps cover the 80/20 case)
- Plan cache (compile_plan is cheap relative to execution)
- Persistent calibration stats (reset on process restart in v1)
- Near-duplicate semantic dedup (text-hash dedup instrumented; embedding-similarity dedup deferred)
