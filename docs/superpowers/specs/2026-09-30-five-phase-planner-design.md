# ANNex Five-Phase Query Planner Design

**Date:** 2026-09-30  
**Branch:** feat/query-planner  
**Status:** Approved for implementation planning

---

## Goal

ANNex compiles an information need into the cheapest retrieval and context plan that satisfies the application's quality requirements. The planner is the engine that makes that happen — progressively more autonomous across five phases.

---

## Current State (baseline)

- `MultiVectorIndex.retrieve()` executes channels sequentially, makes binary exact/HNSW decisions via heuristic (`if filtered → exact`).
- Working-tree additions (not yet committed): `LogicalPlan`, `PlannerStats`, `FieldStats`, `RetrievalObjective`, `QualityPreference` types in `planner.rs`; `objective` field on `RetrieveRequest`.
- `RetrievalTrace` already captures per-channel timing and operator selected.
- Filter path does a full document metadata scan to compute `eligible_documents`.

---

## Module Structure

```
crates/annex-multivector/src/
  engine.rs         — MultiVectorIndex core: storage, ANN, upsert/delete/query
    planner.rs      — Phase 1–2: physical optimizer, cost model, PlannerStats
    policy.rs       — Phase 3: retrieval-policy optimizer [NEW]
    retrieval.rs    — Phase 4: context optimizer, retrieve(), all request types
    agent.rs        — Phase 5: agentic planner, iterative search [NEW]
```

`engine.rs` uses `#[path = "..."]` includes for all sub-modules.

---

## Phase 1 — Physical Optimizer Foundation

### PlannerStats caching

`PlannerStats` is currently computed on every `plan()` and `retrieve()` call by iterating all documents. Cache it on `State` as `stats_cache: Option<PlannerStats>`. Invalidated (set to `None`) on every write that bumps generation. Populated lazily on first `plan()` or `retrieve()` call within a generation, then reused for the life of that `State` snapshot.

### Cost model

Two competing operators for each vector channel:

```
cost_exact(F, D)         = F × D                     // F eligible docs, D = fde_dimension
cost_hnsw(ef, N, D)      = ef × ceil(log2(N)) × D    // N = corpus, ef = ef_search
```

Choose HNSW when:
- `cost_hnsw < cost_exact` AND
- ANN index is current-generation AND
- Not a filtered query OR selectivity `F/N ≥ 0.1`

For filtered queries with `F/N ≥ 0.1`, use HNSW with over-fetch: `ef = max(ef_search, ceil(k / selectivity))` capped at `4 × ef_search`.

Add `estimated_cost: f64` to `PlannedChannel`. Add `actual_ms: f64` and `candidates_returned: usize` to per-channel trace entries (already exist but need cost field).

### Concurrent channel execution

Replace the sequential loop in `retrieve()` with `rayon::scope`. Each channel scores in its own scope task. The `Arc<State>` snapshot is cloned into each task (cheap — Arc clone). All tasks share the same generation. Results collected into a `Vec<Vec<(String, f32)>>` in channel order.

### EXPLAIN endpoint surface

`plan()` already returns `RetrievalPlan`. `retrieve()` already returns `RetrievalTrace` with per-channel elapsed and operator. No new endpoints needed — the existing `plan()` is the dry-run EXPLAIN, and `RetrievalTrace` inside `RetrievalResponse` is the EXPLAIN ANALYZE.

Add `plan_cost: f64` (sum of `estimated_cost` across channels) to `RetrievalPlan`. Add `actual_cost_ms: f64` (sum of per-channel `actual_ms`) to `RetrievalTrace`.

---

## Phase 2 — Advanced Physical Execution

### Selectivity thresholds

Three zones:
- `F/N < 0.05` → always exact (filter is too selective; HNSW over-fetch cost exceeds exact scan)
- `0.05 ≤ F/N < 0.3` → HNSW with amplified ef: `ef = min(ef_search / selectivity, 4 × ef_search)`
- `F/N ≥ 0.3` → HNSW with original `ef_search` (selectivity is mild; standard over-fetch is fine)

Selectivity thresholds stored as constants, exposed in `PlannerStats` (so they can be tuned).

### Adaptive ef_search

When `latency_budget_ms` is set, estimate available per-channel budget:

```
per_channel_budget = (latency_budget_ms - filter_scan_ms_estimate) / num_channels
ef_adaptive = clamp(per_channel_budget / ns_per_ef_step_estimate × 1e6, 16, 65536)
```

`ns_per_ef_step_estimate` comes from `RuntimeStats` in `PlannerStats` (Phase 2 addition). Default estimate: 1000 ns/step (calibrated over time). When adaptive ef is computed, `PlannedChannel.ef_search` reflects the adapted value and a new `PlanReason::AdaptiveLatencyBudget` is recorded.

### RuntimeStats

New struct on `PlannerStats`:

```rust
pub struct RuntimeStats {
    pub exact_ns_per_fde_vector: f64,   // moving average from retrieve traces
    pub hnsw_ns_per_ef_step: f64,       // moving average
    pub filter_scan_ns_per_doc: f64,    // moving average
}
```

Updated after each `retrieve()` via an `AtomicU64` pair (sum, count) on `MultiVectorIndex` — not on `State` (shared mutable, doesn't need generation). Stats exposed in `PlannerStats` so cost model uses calibrated values.

### Conditional escalation

If `rerank` is configured with `adaptive`, and channel agreement falls below the adaptive threshold, the plan already triggers extended rerank. Phase 2 adds **full escalation**: if post-rerank score improvement exceeds a threshold (top-1 MaxSim score / top-1 FDE score > 1.5), mark the channel as `escalated: true` in the trace and note it in `RetrievalTrace.escalations: Vec<usize>` (channel indices that escalated).

---

## Phase 3 — Retrieval-Policy Optimizer

New file: `policy.rs`

### API change

```rust
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanningMode {
    #[default]
    Manual,    // use prefetch as-is (current behavior)
    Auto,      // ignore prefetch, generate from corpus stats + objective
    AutoWithOverrides,  // generate policy plan, but merge with provided prefetch hints
}
```

`RetrieveRequest` gains `planning_mode: PlanningMode` (default `Manual`, so all existing callers are unaffected).

When `planning_mode == Auto`, `prefetch` may be empty; `policy.rs` generates the channel list before `compile_plan` is called.

### Policy planner heuristic (v1)

Coverage thresholds (all relative to `PlannerStats.documents`):

| Representation | Include condition |
|----------------|------------------|
| BM25 | `token_documents ≥ 0.5 × documents` |
| Dense field | `field_stats[field].documents ≥ 0.5 × documents` |
| Sparse field | sparse index non-empty |
| Multivector (FDE) | `token_documents ≥ 0.5 × documents` |

Candidate limits from `QualityPreference`:
- `Fast`: `2 × result_limit`, ef_search 64
- `Balanced`: `5 × result_limit`, ef_search 256
- `High`: `10 × result_limit`, ef_search 1024

Fusion: RRF (k=10) for ≥2 channels; native score for single channel.

Rerank: added automatically when multivector (FDE) channel is included AND `quality == High`.

Context budget: if `context_budget_tokens` is set, `result_limit` is adjusted upward by 1.5× to give the context optimizer room to pack.

### PolicyPlan

```rust
pub struct PolicyPlan {
    pub channels_selected: Vec<LogicalChannelKind>,
    pub reason: Vec<String>,    // human-readable selection rationale
    pub generated_prefetch: Vec<Channel>,
}
```

Embedded in `RetrievalTrace` when `planning_mode != Manual`.

---

## Phase 4 — Context Optimizer

Lives in `retrieval.rs` (extends existing context assembly).

### Token-aware packing

`context_budget_tokens` is already on `RetrievalObjective` but not enforced as a stopping condition in context assembly. Add enforcement: when assembling `ContextHit` list, track cumulative estimated tokens. Estimate: `text.split_whitespace().count() × 1.3` (word count × overhead factor). Stop adding hits when cumulative estimate exceeds `context_budget_tokens`.

Add `estimated_tokens: Option<usize>` to `ContextHit`.

### Evidence confidence

Add to `RetrievalTrace`:

```rust
pub evidence_confidence: f32,    // 0.0–1.0
pub source_diversity: f32,       // unique parents / total hits, 0.0–1.0
```

`evidence_confidence = 1.0 − (median_score / top1_score)` clipped to [0, 1]. High confidence: scores cluster near the top. Low confidence: top-1 vastly outscores the rest.

`source_diversity = unique_parent_count / matches.len()` (1.0 if all hits from different parents).

### Semantic dedup (strengthen existing)

The `deduplicate: bool` flag already hashes text. Add `dedup_count: usize` to `RetrievalTrace` — number of candidates dropped by dedup. Helps callers understand if dedup is actually firing.

---

## Phase 5 — Agentic Planner

New file: `agent.rs`

### AgentSearch

```rust
pub struct AgentSearch {
    pub objective: RetrievalObjective,
    pub prior_results: Vec<ContextHit>,
    pub iteration: usize,
    pub total_tokens_consumed: usize,
    strategy_memory: HashMap<u64, CorpusStrategy>,  // keyed by PlannerStats.generation
}

pub struct AgentIteration {
    pub request: RetrieveRequest,
    pub rationale: AgentRationale,
}

pub enum StopReason {
    ConfidenceThresholdMet,
    ContextBudgetExhausted,
    DiminishingReturns,     // < 10% new candidates vs prior iteration
    MaxIterationsReached,
}

pub enum AgentDecision {
    Continue(AgentIteration),
    Stop(StopReason),
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

Decision logic:
1. If iteration 0: emit auto-policy base request.
2. If `total_tokens_consumed ≥ context_budget_tokens`: stop `ContextBudgetExhausted`.
3. If `last_trace.evidence_confidence ≥ 0.85`: stop `ConfidenceThresholdMet`.
4. If new candidates from last iteration < 10% of prior: stop `DiminishingReturns`.
5. If `iteration ≥ 5`: stop `MaxIterationsReached`.
6. Otherwise: reformulate (widen/narrow query based on confidence, add adjacent sub-queries).

### CorpusStrategy memory

After a successful agentic search, record the effective strategy (channels used, average iterations, final confidence) keyed by `stats.generation`. On future calls with the same generation, use the recorded strategy as the prior for plan generation rather than the baseline heuristic.

---

## API Summary — What Changes

### New fields on existing types

| Type | Field | Phase |
|------|-------|-------|
| `RetrieveRequest` | `planning_mode: PlanningMode` | 3 |
| `PlannedChannel` | `estimated_cost: f64` | 1 |
| `RetrievalTrace` | `plan_cost: f64`, `actual_cost_ms: f64` | 1 |
| `RetrievalTrace` | `escalations: Vec<usize>` | 2 |
| `RetrievalTrace` | `policy: Option<PolicyPlan>` | 3 |
| `RetrievalTrace` | `evidence_confidence: f32`, `source_diversity: f32`, `dedup_count: usize` | 4 |
| `ContextHit` | `estimated_tokens: Option<usize>` | 4 |
| `PlannerStats` | `runtime: RuntimeStats` | 2 |

### New types

`PlanningMode`, `PolicyPlan`, `RuntimeStats`, `AgentSearch`, `AgentDecision`, `AgentIteration`, `StopReason`, `CorpusStrategy`

### New re-exports from `lib.rs`

`PlanningMode`, `PolicyPlan`, `AgentSearch`, `AgentDecision`, `StopReason`

### New public methods on `MultiVectorIndex`

None required beyond what exists. `AgentSearch` is a standalone struct callers construct and drive.

---

## Backward Compatibility

All new fields on serialized types use `#[serde(default)]`. `planning_mode` defaults to `Manual`. All existing `retrieve()` and `plan()` callers are unaffected. `prefetch` remains required (non-optional) when `planning_mode == Manual`; the validator rejects empty `prefetch` in Manual mode as it does today.

---

## Planner Regret Benchmark

Separate from implementation, but designed-for:

1. Enumerate feasible plans for a query (exact vs HNSW, each subset of channels, ef in {64, 256, 1024}).
2. Score each plan: (nDCG@10, latency_ms).
3. Build Pareto frontier.
4. Compare ANNex's chosen plan to the Pareto frontier.

Metrics: **Pareto hit rate** (% of queries where chosen plan is on the frontier), **median nDCG regret** (chosen nDCG − oracle nDCG), **P95 latency violation** (% of queries where chosen plan violates latency budget), **budget violation rate**.

---

## Out of Scope

- LLM-based query reformulation (Phase 5 uses text splitting only)
- Persistent runtime stats (stats reset on process restart in v1)
- Metadata indexes (bitmap/posting-list filter acceleration) — deferred; current filter scan is acceptable for corpora < 1M documents
- Plan cache (deferred; compile_plan is cheap)
