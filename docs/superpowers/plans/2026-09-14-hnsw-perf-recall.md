# HNSW Performance & Recall Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement five orthogonal optimizations — parameter calibration, triangle-inequality neighbor skip (PEOs), RCM graph reordering, SQ8 quantization + rerank, and LID bulk-insert ordering — targeting 2–3× throughput and +5–10pp recall on the NYTimes-256 benchmark.

**Architecture:** Tasks 1–5 are independent; 2–5 can be worked in parallel branches. All changes live in `src/vector/hnsw/`. The warmed frontier benchmark (`tests/nytimes_frontier.rs`) measures each improvement. Snapshots retain backward compatibility — edge distances and quantized codes are recomputed on load, not persisted.

**Tech Stack:** Rust stable, `parking_lot` RwLock (already in Cargo.toml), AVX2/NEON SIMD via existing `std::arch` patterns in `core.rs`, `bincode` for snapshots.

**Spec:** this document

## Global Constraints

- No new crate dependencies; match existing `dot_avx2_fma` / `dot_neon` style for any new SIMD
- Snapshot format stays at version 2; no new persisted fields (recompute on load)
- Every new behavior gated behind a `SearchRuntimeOptions` bool or env var in `config.rs` — defaults unchanged until benchmarks confirm a win
- After every commit: `cargo test` (all non-ignored), `cargo fmt --all -- --check`, `cargo clippy --all-targets` must all pass
- Benchmark evidence required before changing any default: collect ef=32,64,128,256 × 3 rounds on the saved NYT-256 index

---

## Task 1: Calibrate early-exit patience, scan patience, and adaptive EF

These three features are already implemented but either have no default (adaptive EF) or unvalidated defaults. This task runs systematic sweeps, picks the best values, and wires them as documented env-var recommendations.

**Files:**
- Modify: `tests/nytimes_frontier.rs` — add calibration sweep mode
- Modify: `src/vector/hnsw/config.rs` — update any default that benchmarks justify

**Interfaces:**
- Produces: calibrated default values for `early_exit_patience`, `neighbor_scan_patience`, `adaptive_ef_high` + `adaptive_ef_score_threshold`

- [ ] **Step 1: Extend the frontier benchmark with calibration sweep**

Add this function and test to `tests/nytimes_frontier.rs` (after the existing `nytimes_warmed_frontier` test):

```rust
fn load_index(path: &str) -> HNSWIndex {
    use annex::segment::Segment;
    let segment = Segment::load_from_path(path).unwrap();
    let index = HNSWIndex::from_snapshot(segment.hnsw().to_snapshot());
    drop(segment);
    index
}

#[test]
#[ignore]
fn nytimes_calibrate_patience() {
    let path = std::env::var("VECTORDB_NYT_PERSIST_PATH")
        .unwrap_or_else(|_| "data/nytimes-256-angular/index_m16_m0_32_stored_128_efc300.bin".into());
    let index = load_index(&path);
    let queries: ndarray::Array2<f32> =
        ndarray_npy::read_npy("data/nytimes-256-angular/queries.npy").unwrap();
    let truth: Vec<Vec<u64>> = serde_json::from_slice(
        &std::fs::read("data/nytimes-256-angular/ground_truth.json").unwrap(),
    )
    .unwrap();
    let count = 1000usize;
    let queries: Vec<annex::utils::types::Vector> = queries
        .rows()
        .into_iter()
        .take(count)
        .map(|r| r.to_vec())
        .collect();

    for early_patience in [0usize, 1, 2, 3, 5, 8] {
        for scan_patience in [0usize, 2, 4, 8] {
            for ef in [32usize, 64, 128, 256] {
                let opts = SearchRuntimeOptions {
                    ef_search: Some(ef),
                    early_exit_patience: Some(early_patience),
                    neighbor_scan_patience: Some(scan_patience),
                    ..Default::default()
                };
                let mut hits = 0usize;
                for (i, q) in queries.iter().enumerate() {
                    let results = index.search_with_options(q, 20, &opts).unwrap();
                    hits += results
                        .iter()
                        .filter(|r| truth[i][..20].contains(&r.id))
                        .count();
                }
                let recall = hits as f64 / (count * 20) as f64;
                let start = std::time::Instant::now();
                for q in &queries {
                    std::hint::black_box(
                        index.search_with_options(std::hint::black_box(q), 20, &opts).unwrap()
                    );
                }
                let qps = count as f64 / start.elapsed().as_secs_f64();
                println!(
                    "{}",
                    serde_json::json!({
                        "experiment": "calibrate_patience",
                        "ef": ef,
                        "early_patience": early_patience,
                        "scan_patience": scan_patience,
                        "recall": recall,
                        "qps": qps
                    })
                );
            }
        }
    }
}

#[test]
#[ignore]
fn nytimes_calibrate_adaptive_ef() {
    let path = std::env::var("VECTORDB_NYT_PERSIST_PATH")
        .unwrap_or_else(|_| "data/nytimes-256-angular/index_m16_m0_32_stored_128_efc300.bin".into());
    let index = load_index(&path);
    let queries: ndarray::Array2<f32> =
        ndarray_npy::read_npy("data/nytimes-256-angular/queries.npy").unwrap();
    let truth: Vec<Vec<u64>> = serde_json::from_slice(
        &std::fs::read("data/nytimes-256-angular/ground_truth.json").unwrap(),
    )
    .unwrap();
    let count = 1000usize;
    let queries: Vec<annex::utils::types::Vector> = queries
        .rows()
        .into_iter()
        .take(count)
        .map(|r| r.to_vec())
        .collect();

    // Sweep thresholds: distance values where we retry with higher ef.
    // For cosine, sort_key is 1 - similarity; 0.05 = query has similarity < 0.95 to top result.
    for base_ef in [64usize, 128] {
        for high_ef in [base_ef * 2, base_ef * 4] {
            for threshold in [0.03f32, 0.05, 0.08, 0.12] {
                let opts = SearchRuntimeOptions {
                    ef_search: Some(base_ef),
                    adaptive_ef_high: Some(high_ef),
                    adaptive_ef_score_threshold: Some(threshold),
                    ..Default::default()
                };
                let mut hits = 0usize;
                let start = std::time::Instant::now();
                for (i, q) in queries.iter().enumerate() {
                    let results = index.search_with_options(q, 20, &opts).unwrap();
                    hits += results
                        .iter()
                        .filter(|r| truth[i][..20].contains(&r.id))
                        .count();
                }
                let qps = count as f64 / start.elapsed().as_secs_f64();
                let recall = hits as f64 / (count * 20) as f64;
                println!(
                    "{}",
                    serde_json::json!({
                        "experiment": "calibrate_adaptive_ef",
                        "base_ef": base_ef,
                        "high_ef": high_ef,
                        "threshold": threshold,
                        "recall": recall,
                        "qps": qps
                    })
                );
            }
        }
    }
}
```

- [ ] **Step 2: Confirm calibration tests compile and run**

```bash
cargo test --test nytimes_frontier -- --list 2>&1 | grep calibrate
```
Expected: two `nytimes_calibrate_*` tests listed as ignored.

- [ ] **Step 3: Run patience calibration (requires NYT index)**

```bash
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_m16_m0_32_stored_128_efc300.bin \
  cargo test --test nytimes_frontier nytimes_calibrate_patience -- --ignored --nocapture 2>/dev/null \
  | grep '"experiment"' | python3 -c "
import sys, json
rows = [json.loads(l) for l in sys.stdin]
# find best recall-vs-qps per ef
for ef in sorted(set(r['ef'] for r in rows)):
    subset = [r for r in rows if r['ef'] == ef]
    subset.sort(key=lambda r: (-r['recall'], -r['qps']))
    best = subset[0]
    print(f'ef={ef} best: recall={best[\"recall\"]:.4f} qps={best[\"qps\"]:.0f} early={best[\"early_patience\"]} scan={best[\"scan_patience\"]}')
"
```

- [ ] **Step 4: Run adaptive EF calibration**

```bash
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_m16_m0_32_stored_128_efc300.bin \
  cargo test --test nytimes_frontier nytimes_calibrate_adaptive_ef -- --ignored --nocapture 2>/dev/null \
  | grep '"experiment"'
```

- [ ] **Step 5: If sweep reveals a clearly better patience default (same recall, ≥5% more QPS at ef=128), update it in `config.rs`**

Current line in `src/vector/hnsw/config.rs:92`:
```rust
pub fn early_exit_patience() -> usize {
    *EARLY_EXIT_PATIENCE.get_or_init(|| env_usize("VECTORDB_EARLY_EXIT_PATIENCE").unwrap_or(2))
}
```
Change the `unwrap_or(2)` value if the sweep justifies it. If no clear win, leave as-is and document in the commit message.

- [ ] **Step 6: Run full test suite**

```bash
cargo test
```
Expected: all non-ignored tests pass.

- [ ] **Step 7: Commit**

```bash
git add tests/nytimes_frontier.rs src/vector/hnsw/config.rs
git commit -m "Add calibration benchmarks for patience and adaptive EF; tune defaults"
```

---

## Task 2: Triangle-inequality neighbor skip (PEOs) at L0

**Key idea:** When expanding candidate `c` (distance `d(q,c)` known), for each neighbor `n` with stored edge distance `d(c,n)`, the triangle inequality gives `d(q,n) ≥ d(q,c) − d(c,n)`. If that lower bound already exceeds the current worst result, skip the expensive `fast_score` call. For cosine on pre-normalized vectors this is an approximation (cosine isn't exactly a metric); empirically it holds well for close neighbors in a high-quality HNSW graph.

**Files:**
- Modify: `src/vector/hnsw/core.rs` — add `edge_dists_l0` field, extend in node alloc
- Modify: `src/vector/hnsw/insert.rs` — write edge distances alongside neighbor lists
- Modify: `src/vector/hnsw/search.rs` — apply TI skip in hot path
- Modify: `src/vector/hnsw/types.rs` — add `use_ti_skip` to `SearchRuntimeOptions`
- Modify: `src/vector/hnsw/config.rs` — add `ti_skip_enabled()` env var
- Test: `tests/hnsw.rs`

**Interfaces:**
- Consumes: `HNSWIndex::fast_score`, `layers[0][idx].read()` read lock
- Produces: `HNSWIndex::edge_dists_l0: Vec<parking_lot::RwLock<Vec<f32>>>`, `SearchRuntimeOptions::use_ti_skip: Option<bool>`

- [ ] **Step 1: Add `use_ti_skip` to `SearchRuntimeOptions` in `src/vector/hnsw/types.rs`**

Add inside the `SearchRuntimeOptions` struct (after the existing `adaptive_ef_score_threshold` field):

```rust
/// Enable triangle-inequality neighbor skip at L0. When true and the index has
/// edge distances stored, neighbors whose TI lower bound exceeds `worst_score` are
/// skipped before computing `fast_score`. Not valid for Dot metric (no bounded metric).
/// Overrides `VECTORDB_TI_SKIP`.
pub use_ti_skip: Option<bool>,
```

- [ ] **Step 2: Add `ti_skip_enabled()` to `src/vector/hnsw/config.rs`**

Add after the `adaptive_ef_score_threshold_default` function:

```rust
static TI_SKIP: OnceLock<bool> = OnceLock::new();

pub fn ti_skip_enabled() -> bool {
    *TI_SKIP.get_or_init(|| env_bool("VECTORDB_TI_SKIP").unwrap_or(false))
}
```

Also add `ti_skip_enabled` to the import list in `search.rs` and wherever used.

- [ ] **Step 3: Add `edge_dists_l0` field to `HNSWIndex` in `src/vector/hnsw/core.rs`**

After the `deleted_count` field:

```rust
/// Parallel to layers[0]: edge_dists_l0[idx] holds the distance from node idx to
/// each of its L0 neighbors, co-indexed with layers[0][idx].
/// Protected by the same logical lock as layers[0][idx]: always write under
/// layers[0][idx].write() and read under layers[0][idx].read().
pub(crate) edge_dists_l0: Vec<parking_lot::RwLock<Vec<f32>>>,
```

In `HNSWIndex::new(...)` initializer block (after the `deleted_count: 0` line):

```rust
edge_dists_l0: Vec::new(),
```

- [ ] **Step 4: Extend `edge_dists_l0` in `register_node` and `extend_layers_for_new_node`**

In `register_node` (after `self.deleted.push(false);`):

```rust
self.edge_dists_l0.push(parking_lot::RwLock::new(Vec::new()));
```

In `extend_layers_for_new_node` (after the existing `for level` loop, unconditionally):

```rust
while self.edge_dists_l0.len() < nodes_len {
    self.edge_dists_l0.push(parking_lot::RwLock::new(Vec::new()));
}
```

In `from_snapshot`, after `deleted_count:` line and before closing `Self {`:

```rust
let mut edge_dists_l0: Vec<parking_lot::RwLock<Vec<f32>>> =
    (0..ids.len()).map(|_| parking_lot::RwLock::new(Vec::new())).collect();
// Backfill edge distances for L0 from stored vectors.
if let Some(l0) = layers.first() {
    for (idx, nb_lock) in l0.iter().enumerate() {
        let nb = nb_lock.read();
        if nb.is_empty() { continue; }
        let src = &vectors[idx * snapshot.dim..(idx + 1) * snapshot.dim];
        let dists: Vec<f32> = nb.iter().map(|&n| {
            let dst = &vectors[n * snapshot.dim..(n + 1) * snapshot.dim];
            // For cosine, vectors are pre-normalized; distance = 1 - dot(src, dst)
            let dot: f32 = src.iter().zip(dst.iter()).map(|(a, b)| a * b).sum();
            (1.0 - dot).max(0.0)
        }).collect();
        *edge_dists_l0[idx].write() = dists;
    }
}
```

Add `edge_dists_l0,` to the `Self { ... }` block in `from_snapshot`.

- [ ] **Step 5: Write edge distances in `insert.rs` when the L0 neighbor list is written**

In the block that writes the new node's own neighbor list (around line 141–148 of `insert.rs`), after `*self.layers[l][idx].write() = linked;`, add:

```rust
if l == 0 {
    let src = self.vector_slice(idx).to_vec();
    let dists: Vec<f32> = linked.iter().map(|&n| {
        if n == idx { return 0.0; }
        let dst_slice = self.vector_slice(n);
        let dot: f32 = src.iter().zip(dst_slice.iter()).map(|(a, b)| a * b).sum();
        (1.0 - dot).max(0.0)
    }).collect();
    *self.edge_dists_l0[idx].write() = dists;
}
```

In the block that writes back-edges (around line 162–173 of `insert.rs`), after `nb_list.insert(pos, idx);`, add (still inside `if l == 0` guard):

```rust
if l == 0 {
    let n_vec = n_vec.clone(); // already have n_vec from above
    let dists: Vec<f32> = nb_list.iter().map(|&nb| {
        if nb == n { return 0.0; }
        let nb_slice = self.vector_slice(nb);
        let dot: f32 = n_vec.iter().zip(nb_slice.iter()).map(|(a, b)| a * b).sum();
        (1.0 - dot).max(0.0)
    }).collect();
    *self.edge_dists_l0[n].write() = dists;
}
```

Note: this recalculates all distances for node `n` each time a back-edge is added. This is O(degree × dim) per back-edge insertion — acceptable for insert paths which are already much more expensive than this.

- [ ] **Step 6: Apply TI skip in `search_layer_unfiltered` hot path**

In `src/vector/hnsw/search.rs`, add to the imports at the top:

```rust
use super::config::{/* existing imports */, ti_skip_enabled};
```

After computing `worst_score` (around line 270), add:

```rust
let use_ti = level == 0
    && self.metric != DistanceMetric::Dot
    && (opts.use_ti_skip.unwrap_or_else(ti_skip_enabled))
    && !self.edge_dists_l0.is_empty();
```

Inside the `use_simple_scan` branch, change the neighbor iteration loop to:

```rust
for (position, &neighbor) in neighbors.iter().enumerate() {
    // TI lower-bound skip: if result set is full and d(q,n) ≥ d(q,c) - d(c,n) > worst,
    // n cannot improve the result set.
    if use_ti && scratch.result_set.len() >= ef {
        if let Some(ed_lock) = self.edge_dists_l0.get(current.idx) {
            let ed = ed_lock.read();
            if let Some(&edge_d) = ed.get(position) {
                if current.sort_key - edge_d > worst_score {
                    continue;
                }
            }
        }
    }
    if let Some(&next) = neighbors.get(position + 2)
    // ... rest of existing prefetch + visited check + scoring
```

Apply the same TI check in the complex-scan `'neighbor_scan` loop (same pattern, around line 437).

- [ ] **Step 7: Write the failing test**

In `tests/hnsw.rs`, add:

```rust
#[test]
fn ti_skip_matches_baseline_recall() {
    use annex::vector::hnsw::{HNSWIndex, SearchRuntimeOptions};
    use annex::utils::types::DistanceMetric;
    let mut index = HNSWIndex::new(DistanceMetric::Cosine, 16, 200, 4, 8);
    let vecs: Vec<Vec<f32>> = (0..200u64).map(|i| {
        let mut v = vec![0.0f32; 8];
        v[i as usize % 8] = 1.0;
        v[(i as usize + 1) % 8] = 0.5;
        v
    }).collect();
    for (i, v) in vecs.iter().enumerate() {
        index.insert(i as u64, v.clone()).unwrap();
    }
    let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let baseline_opts = SearchRuntimeOptions {
        ef_search: Some(50),
        use_ti_skip: Some(false),
        ..Default::default()
    };
    let ti_opts = SearchRuntimeOptions {
        ef_search: Some(50),
        use_ti_skip: Some(true),
        ..Default::default()
    };
    let baseline = index.search_with_options(&query, 10, &baseline_opts).unwrap();
    let ti_result = index.search_with_options(&query, 10, &ti_opts).unwrap();
    // TI skip is an approximation; top-1 must match exactly.
    assert_eq!(baseline[0].id, ti_result[0].id, "top-1 must match");
    // At least 80% recall for the top-10 set.
    let baseline_ids: std::collections::HashSet<_> = baseline.iter().map(|r| r.id).collect();
    let overlap = ti_result.iter().filter(|r| baseline_ids.contains(&r.id)).count();
    assert!(overlap >= 8, "TI skip recall vs baseline: {}/10", overlap);
}
```

- [ ] **Step 8: Run test to verify it fails**

```bash
cargo test ti_skip_matches_baseline_recall -- --nocapture
```
Expected: compile error or test failure (field/option not yet wired).

- [ ] **Step 9: Implement all changes from Steps 1–6**

Complete all the edits described above.

- [ ] **Step 10: Run tests**

```bash
cargo test
```
Expected: all non-ignored tests pass including `ti_skip_matches_baseline_recall`.

- [ ] **Step 11: Benchmark TI skip on NYT-256 (requires saved index)**

```bash
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_m16_m0_32_stored_128_efc300.bin \
VECTORDB_TI_SKIP=true \
VECTORDB_EF_SEARCH_LIST=32,64,128,256 \
  cargo test --test nytimes_frontier nytimes_warmed_frontier -- --ignored --nocapture 2>/dev/null \
  | grep '"qps"'
```

Compare QPS and recall against the baseline run (same command without `VECTORDB_TI_SKIP=true`). If QPS is higher with equal or better recall at every ef, set the default to `true` in `config.rs`.

- [ ] **Step 12: Commit**

```bash
git add src/vector/hnsw/core.rs src/vector/hnsw/insert.rs src/vector/hnsw/search.rs \
        src/vector/hnsw/types.rs src/vector/hnsw/config.rs tests/hnsw.rs
git commit -m "Add triangle-inequality neighbor skip at L0 (PEOs); store edge distances"
```

---

## Task 3: RCM graph node reordering

**Key idea:** After building the HNSW graph, permute node indices using Reverse Cuthill-McKee so that graph-adjacent nodes become memory-adjacent. This dramatically reduces L1/L2/L3 cache miss rates during BFS traversal with no algorithm changes.

**Files:**
- Modify: `src/vector/hnsw/core.rs` — add `reorder_rcm()` method
- Test: `tests/hnsw.rs`

**Interfaces:**
- Consumes: `self.layers[0]` (L0 adjacency), `self.vectors`, `self.idx_to_point`, `self.deleted`, `self.point_to_idx`, `self.edge_dists_l0`
- Produces: `pub fn reorder_rcm(&mut self)` — permutes all node-indexed arrays in place

- [ ] **Step 1: Write the failing test**

In `tests/hnsw.rs`, add:

```rust
#[test]
fn reorder_rcm_preserves_search_results() {
    use annex::vector::hnsw::HNSWIndex;
    use annex::utils::types::DistanceMetric;
    let mut index = HNSWIndex::new(DistanceMetric::Cosine, 8, 50, 4, 16);
    let mut rng_state = 12345u64;
    let mut lcg = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 33) as f32 / u32::MAX as f32
    };
    for i in 0..100u64 {
        let v: Vec<f32> = (0..16).map(|_| lcg()).collect();
        index.insert(i, v).unwrap();
    }
    let query: Vec<f32> = (0..16).map(|j| if j == 0 { 1.0 } else { 0.0 }).collect();
    let before = index.search(&query, 10).unwrap();
    index.reorder_rcm();
    let after = index.search(&query, 10).unwrap();
    let before_ids: Vec<u64> = before.iter().map(|r| r.id).collect();
    let after_ids: Vec<u64> = after.iter().map(|r| r.id).collect();
    assert_eq!(before_ids, after_ids, "reorder must not change search results");
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test reorder_rcm_preserves_search_results
```
Expected: compile error — method `reorder_rcm` not found.

- [ ] **Step 3: Implement `reorder_rcm` in `src/vector/hnsw/core.rs`**

Add as a new `impl HNSWIndex` block after the existing `allocate_entry_point` block:

```rust
impl HNSWIndex {
    /// Permute node indices using Reverse Cuthill-McKee so that graph-adjacent
    /// nodes at L0 become memory-adjacent. Reduces cache miss rate during BFS.
    /// All node-indexed arrays (vectors, layers, idx_to_point, deleted,
    /// point_to_idx, edge_dists_l0) are permuted consistently.
    pub fn reorder_rcm(&mut self) {
        let n = self.len();
        if n == 0 {
            return;
        }

        // Build undirected adjacency list from L0.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        if let Some(l0) = self.layers.first() {
            for (u, nb_lock) in l0.iter().enumerate() {
                for &v in nb_lock.read().iter() {
                    if v != u {
                        if !adj[u].contains(&v) { adj[u].push(v); }
                        if !adj[v].contains(&u) { adj[v].push(u); }
                    }
                }
            }
        }

        // RCM: BFS starting from the node with minimum degree.
        let start = (0..n).min_by_key(|&i| adj[i].len()).unwrap_or(0);
        let mut perm = Vec::with_capacity(n);
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        while let Some(u) = queue.pop_front() {
            perm.push(u);
            let mut nbrs: Vec<usize> = adj[u].iter().copied()
                .filter(|&v| !visited[v])
                .collect();
            nbrs.sort_by_key(|&v| adj[v].len()); // ascending degree
            for v in nbrs {
                if !visited[v] {
                    visited[v] = true;
                    queue.push_back(v);
                }
            }
        }
        // Handle disconnected nodes.
        for i in 0..n {
            if !visited[i] {
                perm.push(i);
            }
        }
        perm.reverse(); // Reverse Cuthill-McKee

        // Build inverse permutation: inv_perm[old_idx] = new_idx
        let mut inv_perm = vec![0usize; n];
        for (new_idx, &old_idx) in perm.iter().enumerate() {
            inv_perm[old_idx] = new_idx;
        }

        // Permute vectors (flat slice: each node occupies `dim` consecutive f32s).
        let dim = self.dim;
        let mut new_vectors = vec![0.0f32; n * dim];
        for (new_idx, &old_idx) in perm.iter().enumerate() {
            let src = &self.vectors[old_idx * dim..(old_idx + 1) * dim];
            new_vectors[new_idx * dim..(new_idx + 1) * dim].copy_from_slice(src);
        }
        self.vectors = new_vectors;

        // Permute idx_to_point and deleted.
        let old_itp = std::mem::take(&mut self.idx_to_point);
        let old_del = std::mem::take(&mut self.deleted);
        let old_lvl = std::mem::take(&mut self.levels);
        self.idx_to_point = perm.iter().map(|&o| old_itp[o]).collect();
        self.deleted = perm.iter().map(|&o| old_del[o]).collect();
        self.levels = perm.iter().map(|&o| old_lvl[o]).collect();

        // Rebuild point_to_idx from new idx_to_point.
        self.point_to_idx.clear();
        for (new_idx, &id) in self.idx_to_point.iter().enumerate() {
            self.point_to_idx.insert(id, new_idx);
        }

        // Update entry point.
        if let Some(ep) = self.entry_point {
            self.entry_point = Some(inv_perm[ep]);
        }

        // Permute all layers: remap neighbor indices through inv_perm.
        for layer in self.layers.iter_mut() {
            let mut new_layer: Vec<parking_lot::RwLock<Vec<usize>>> =
                (0..n).map(|_| parking_lot::RwLock::new(Vec::new())).collect();
            for (old_idx, nb_lock) in layer.iter().enumerate() {
                let new_nbs: Vec<usize> = nb_lock.read().iter()
                    .map(|&nb| inv_perm[nb])
                    .collect();
                *new_layer[inv_perm[old_idx]].write() = new_nbs;
            }
            *layer = new_layer;
        }

        // Permute edge_dists_l0 if present.
        if !self.edge_dists_l0.is_empty() {
            let mut new_ed: Vec<parking_lot::RwLock<Vec<f32>>> =
                (0..n).map(|_| parking_lot::RwLock::new(Vec::new())).collect();
            for (old_idx, ed_lock) in self.edge_dists_l0.iter().enumerate() {
                let dists = ed_lock.read().clone();
                *new_ed[inv_perm[old_idx]].write() = dists;
            }
            self.edge_dists_l0 = new_ed;
        }
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test
```
Expected: all non-ignored tests pass including `reorder_rcm_preserves_search_results`.

- [ ] **Step 5: Benchmark RCM on NYT-256**

```bash
# Build a test index and call reorder_rcm on it, then run the frontier benchmark.
# The simplest approach: add a one-shot binary or extend the frontier test.
```

Add a temporary `#[ignore]` test to `tests/nytimes_frontier.rs`:

```rust
#[test]
#[ignore]
fn nytimes_rcm_benchmark() {
    use annex::segment::Segment;
    use annex::vector::hnsw::{HNSWIndex, SearchRuntimeOptions};
    use std::time::Instant;
    let path = std::env::var("VECTORDB_NYT_PERSIST_PATH")
        .unwrap_or_else(|_| "data/nytimes-256-angular/index_m16_m0_32_stored_128_efc300.bin".into());
    let segment = Segment::load_from_path(&path).unwrap();
    let mut index = HNSWIndex::from_snapshot(segment.hnsw().to_snapshot());
    drop(segment);
    let queries: ndarray::Array2<f32> =
        ndarray_npy::read_npy("data/nytimes-256-angular/queries.npy").unwrap();
    let truth: Vec<Vec<u64>> = serde_json::from_slice(
        &std::fs::read("data/nytimes-256-angular/ground_truth.json").unwrap(),
    )
    .unwrap();
    let count = 1000usize;
    let qs: Vec<annex::utils::types::Vector> = queries.rows().into_iter().take(count)
        .map(|r| r.to_vec()).collect();

    for label in ["before", "after"] {
        for ef in [64usize, 128, 256] {
            let opts = SearchRuntimeOptions { ef_search: Some(ef), ..Default::default() };
            let mut hits = 0;
            for (i, q) in qs.iter().enumerate() {
                let r = index.search_with_options(q, 20, &opts).unwrap();
                hits += r.iter().filter(|x| truth[i][..20].contains(&x.id)).count();
            }
            let start = Instant::now();
            for q in &qs {
                std::hint::black_box(index.search_with_options(q, 20, &opts).unwrap());
            }
            let qps = count as f64 / start.elapsed().as_secs_f64();
            println!("{}", serde_json::json!({"phase": label, "ef": ef,
                "recall": hits as f64 / (count * 20) as f64, "qps": qps}));
        }
        if label == "before" {
            let t = Instant::now();
            index.reorder_rcm();
            eprintln!("RCM took {:?}", t.elapsed());
        }
    }
}
```

Run with:
```bash
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_m16_m0_32_stored_128_efc300.bin \
  cargo test --release --test nytimes_frontier nytimes_rcm_benchmark -- --ignored --nocapture
```

Note: must use `--release` for meaningful latency numbers.

- [ ] **Step 6: Commit**

```bash
git add src/vector/hnsw/core.rs tests/hnsw.rs tests/nytimes_frontier.rs
git commit -m "Add RCM graph node reordering; permutes all node arrays for cache locality"
```

---

## Task 4: SQ8 scalar quantization + f32 rerank

**Key idea:** Store an 8-bit scalar quantization of each L0 vector. During candidate traversal, score with SQ8 (4× wider SIMD lane count vs f32). Collect top `ef_search × rerank_factor` candidates, then rerank with full f32 precision before returning. Memory footprint for traversal drops to 25% of the f32 footprint; SIMD throughput increases proportionally.

**Files:**
- Modify: `src/vector/hnsw/core.rs` — add `quantized: Vec<u8>`, `quant_min: Vec<f32>`, `quant_scale: Vec<f32>` fields; add SQ8 score functions
- Modify: `src/vector/hnsw/search.rs` — add SQ8 traversal path with f32 rerank
- Modify: `src/vector/hnsw/types.rs` — add `sq8_rerank_factor` to `SearchRuntimeOptions`
- Modify: `src/vector/hnsw/config.rs` — add `sq8_rerank_factor_default()`
- Test: `tests/hnsw.rs`

**Interfaces:**
- Consumes: `self.vectors` (f32 flat storage), `self.dim`, `self.metric`
- Produces:
  - `HNSWIndex::quantize_all()` — builds the quantization tables
  - `HNSWIndex::sq8_score(query_q: &[i16], vec_idx: usize) -> f32` — approx score using SQ8
  - `SearchRuntimeOptions::sq8_rerank_factor: Option<usize>`

- [ ] **Step 1: Add `sq8_rerank_factor` to `SearchRuntimeOptions` in `types.rs`**

```rust
/// When > 0, use SQ8 quantized scoring during L0 traversal and collect
/// `ef_search * sq8_rerank_factor` candidates, then rerank with full f32.
/// Set to 0 to disable SQ8 traversal. Overrides `VECTORDB_SQ8_RERANK_FACTOR`.
pub sq8_rerank_factor: Option<usize>,
```

- [ ] **Step 2: Add `sq8_rerank_factor_default()` to `config.rs`**

```rust
static SQ8_RERANK_FACTOR: OnceLock<Option<usize>> = OnceLock::new();

pub fn sq8_rerank_factor_default() -> Option<usize> {
    *SQ8_RERANK_FACTOR.get_or_init(|| env_usize_nonzero("VECTORDB_SQ8_RERANK_FACTOR"))
}
```

- [ ] **Step 3: Add quantization fields to `HNSWIndex` in `core.rs`**

After `edge_dists_l0`:

```rust
/// SQ8 quantized vectors: quantized[idx * dim + d] = u8 encoding of dimension d.
/// Empty until `quantize_all()` is called.
pub(crate) quantized: Vec<u8>,
/// Per-dimension minimum value used for SQ8 quantization.
pub(crate) quant_min: Vec<f32>,
/// Per-dimension scale: (max - min) / 255.0. Zero means that dimension is constant.
pub(crate) quant_scale: Vec<f32>,
```

In `HNSWIndex::new()`:

```rust
quantized: Vec::new(),
quant_min: Vec::new(),
quant_scale: Vec::new(),
```

In `from_snapshot`, after building `edge_dists_l0`:

```rust
quantized: Vec::new(),
quant_min: Vec::new(),
quant_scale: Vec::new(),
```

(Quantization is built lazily via `quantize_all()` — it's not persisted in snapshots.)

- [ ] **Step 4: Implement `quantize_all` and SQ8 scoring in `core.rs`**

Add as a new impl block:

```rust
impl HNSWIndex {
    /// Build per-dimension min/max from all stored (non-deleted) vectors,
    /// then encode each vector as SQ8. Call after bulk insert or snapshot load.
    pub fn quantize_all(&mut self) {
        let n = self.len();
        let dim = self.dim;
        if n == 0 || dim == 0 {
            return;
        }
        let mut min_d = vec![f32::MAX; dim];
        let mut max_d = vec![f32::MIN; dim];
        for idx in 0..n {
            if self.deleted.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let v = self.vector_slice(idx);
            for d in 0..dim {
                min_d[d] = min_d[d].min(v[d]);
                max_d[d] = max_d[d].max(v[d]);
            }
        }
        let mut scale = vec![0.0f32; dim];
        for d in 0..dim {
            let range = max_d[d] - min_d[d];
            scale[d] = if range > 0.0 { range / 255.0 } else { 1.0 };
        }
        let mut quantized = vec![0u8; n * dim];
        for idx in 0..n {
            let v = self.vector_slice(idx);
            for d in 0..dim {
                let q = ((v[d] - min_d[d]) / scale[d]).clamp(0.0, 255.0).round() as u8;
                quantized[idx * dim + d] = q;
            }
        }
        self.quant_min = min_d;
        self.quant_scale = scale;
        self.quantized = quantized;
    }

    /// Quantize a query vector into signed i16 for dot-product scoring.
    /// Returns a Vec<i16> of length dim.
    pub(crate) fn quantize_query(&self, query: &[f32]) -> Vec<i16> {
        query.iter().enumerate().map(|(d, &v)| {
            let q = ((v - self.quant_min[d]) / self.quant_scale[d])
                .clamp(0.0, 255.0)
                .round() as i16;
            q
        }).collect()
    }

    /// Approximate SQ8 cosine score: dot product of quantized query and stored u8 vector.
    /// Result is an i32 that is monotonically related to the true cosine (higher = closer).
    /// Suitable for candidate ranking but NOT for final score reporting.
    #[inline]
    pub(crate) fn sq8_approx_dot(&self, query_q: &[i16], idx: usize) -> i32 {
        let v = &self.quantized[idx * self.dim..(idx + 1) * self.dim];
        v.iter().zip(query_q.iter()).map(|(&b, &q)| (b as i32) * (q as i32)).sum()
    }
}
```

- [ ] **Step 5: Write the failing test**

In `tests/hnsw.rs`, add:

```rust
#[test]
fn sq8_rerank_top1_matches_f32() {
    use annex::vector::hnsw::{HNSWIndex, SearchRuntimeOptions};
    use annex::utils::types::DistanceMetric;
    let mut index = HNSWIndex::new(DistanceMetric::Cosine, 8, 50, 4, 16);
    let mut rng = 99u64;
    let lcg = |r: &mut u64| -> f32 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*r >> 33) as f32 / u32::MAX as f32
    };
    for i in 0..150u64 {
        let v: Vec<f32> = (0..16).map(|_| lcg(&mut rng)).collect();
        index.insert(i, v).unwrap();
    }
    index.quantize_all();
    let query: Vec<f32> = (0..16).map(|j| if j < 4 { 0.7 } else { 0.0 }).collect();
    let f32_opts = SearchRuntimeOptions {
        ef_search: Some(50),
        sq8_rerank_factor: Some(0),
        ..Default::default()
    };
    let sq8_opts = SearchRuntimeOptions {
        ef_search: Some(50),
        sq8_rerank_factor: Some(4),
        ..Default::default()
    };
    let f32_res = index.search_with_options(&query, 5, &f32_opts).unwrap();
    let sq8_res = index.search_with_options(&query, 5, &sq8_opts).unwrap();
    // Top-1 must agree; at least 3/5 recall for top-5.
    assert_eq!(f32_res[0].id, sq8_res[0].id, "top-1 must match");
    let f32_ids: std::collections::HashSet<_> = f32_res.iter().map(|r| r.id).collect();
    let overlap = sq8_res.iter().filter(|r| f32_ids.contains(&r.id)).count();
    assert!(overlap >= 3, "SQ8 recall vs f32: {}/5", overlap);
}
```

- [ ] **Step 6: Run test to verify it fails**

```bash
cargo test sq8_rerank_top1_matches_f32
```
Expected: compile error — field/method not found.

- [ ] **Step 7: Implement SQ8 traversal path in `search.rs`**

In `run_l0_search`, before the call to `search_layer_unfiltered`, add:

```rust
use super::config::sq8_rerank_factor_default;

let rerank_factor = opts.sq8_rerank_factor
    .or_else(sq8_rerank_factor_default)
    .unwrap_or(0);
let use_sq8 = rerank_factor > 0
    && !self.quantized.is_empty()
    && self.metric != DistanceMetric::Dot;
```

If `use_sq8`, run a modified traversal:

```rust
if use_sq8 {
    let query_q = self.quantize_query(query);
    let extended_ef = ef.saturating_mul(rerank_factor).max(ef);
    // Run the BFS with SQ8 scoring to get a larger candidate pool.
    let (sq8_candidates, _) = self.search_layer_unfiltered_sq8(
        query, &query_q, entries, 0, extended_ef, opts, normalize, stats, trace
    )?;
    // Rerank with full f32.
    let mut reranked: Vec<NodeCandidate> = sq8_candidates.iter().map(|c| {
        let raw = self.fast_score(query, self.vector_slice(c.idx));
        let sort_key = if normalize { self.normalize_score(raw) } else { raw };
        NodeCandidate { idx: c.idx, raw_score: raw, sort_key }
    }).collect();
    reranked.sort_by(|a, b| a.sort_key.partial_cmp(&b.sort_key).unwrap());
    reranked.truncate(ef);
    return Ok(reranked);
}
```

Implement `search_layer_unfiltered_sq8` as a thin wrapper over `search_layer_unfiltered` that uses `sq8_approx_dot` instead of `fast_score` for scoring. The simplest correct implementation: duplicate the logic of `search_layer_unfiltered` but replace `self.fast_score(query, ...)` with `-(self.sq8_approx_dot(query_q, idx) as f32)` as the sort key (negated because higher dot = closer = lower sort key in the result heap). The raw_score is left as 0.0 (it gets overwritten in the rerank pass).

Since duplicating the full BFS loop is verbose, implement it by calling `search_layer_unfiltered` with a temporary lambda override is not possible in Rust; write a new private method `search_layer_unfiltered_sq8` with the same structure as `search_layer_unfiltered` but replace the two `self.fast_score(...)` lines with `-(self.sq8_approx_dot(query_q, idx) as f32)` and the `normalize` flag set to false (the sq8 score is already a sort key).

The full method signature:
```rust
fn search_layer_unfiltered_sq8(
    &self,
    query: &[f32],          // original f32 query (for prefetch)
    query_q: &[i16],        // quantized query
    entries: &[usize],
    level: usize,
    ef: usize,
    opts: &SearchRuntimeOptions,
    normalize: bool,
    stats: Option<&mut SearchLayerStats>,
    trace: Option<&mut SearchTraceCtx>,
) -> Result<(Vec<NodeCandidate>, SearchCounters), DBError>
```

Body: copy `search_layer_unfiltered`, replace:
- `self.fast_score(query, self.vector_slice(idx))` → `-(self.sq8_approx_dot(query_q, idx) as f32)`
- Remove `normalize_score` calls (sq8 sort keys are already correct direction)
- Set `raw_score: 0.0` (placeholder; reranked with f32 afterwards)

- [ ] **Step 8: Run tests**

```bash
cargo test
```
Expected: all non-ignored tests pass including `sq8_rerank_top1_matches_f32`.

- [ ] **Step 9: Benchmark SQ8 on NYT-256**

Add a `#[test] #[ignore] fn nytimes_sq8_benchmark()` test in `tests/nytimes_frontier.rs` following the same pattern as `nytimes_rcm_benchmark` in Task 3: load index, call `quantize_all()`, then sweep ef=64,128,256 with `sq8_rerank_factor=0` (baseline) and `sq8_rerank_factor=3`.

Run: `cargo test --release --test nytimes_frontier nytimes_sq8_benchmark -- --ignored --nocapture`

- [ ] **Step 10: Commit**

```bash
git add src/vector/hnsw/core.rs src/vector/hnsw/search.rs \
        src/vector/hnsw/types.rs src/vector/hnsw/config.rs tests/hnsw.rs \
        tests/nytimes_frontier.rs
git commit -m "Add SQ8 quantization + f32 rerank; quantize_all() builds per-dim tables"
```

---

## Task 5: LID-ordered bulk insertion

**Key idea:** Before parallel bulk insert, sort vectors by descending estimated Local Intrinsic Dimensionality (LID). High-LID vectors (hubs/outliers) become entry-point candidates for upper layers, improving long-range routing and recall without changing query-time behavior.

**Files:**
- Modify: `src/vector/hnsw/insert.rs` — add LID estimation and sort before `par_insert_batch` Phase 1
- Modify: `src/vector/hnsw/config.rs` — add `lid_sort_enabled()` env var
- Test: `tests/hnsw.rs`

**Interfaces:**
- Consumes: `entries: &[(PointId, Vector)]` (input to `par_insert_batch`)
- Produces: sorted order applied before Phase 1 allocation; query path unchanged

- [ ] **Step 1: Add `lid_sort_enabled()` to `config.rs`**

```rust
static LID_SORT: OnceLock<bool> = OnceLock::new();

pub fn lid_sort_enabled() -> bool {
    *LID_SORT.get_or_init(|| env_bool("VECTORDB_LID_SORT").unwrap_or(false))
}
```

- [ ] **Step 2: Write the failing test**

LID ordering is a build-time quality change; its effect is visible in recall at low ef. Test: build a 500-vector index without LID sort and with LID sort, run recall@10 at ef=20 over 50 random queries, confirm LID sort achieves at least as good recall (or statistically equivalent — LID's effect is probabilistic).

In `tests/hnsw.rs`:

```rust
#[test]
fn lid_sort_does_not_regress_recall() {
    use annex::vector::hnsw::{HNSWIndex, SearchRuntimeOptions};
    use annex::utils::types::DistanceMetric;

    fn build(use_lid: bool) -> HNSWIndex {
        std::env::set_var("VECTORDB_LID_SORT", if use_lid { "true" } else { "false" });
        // Reset the OnceLock by bypassing (can't easily; use a function that reads env directly).
        // Instead, create a helper that accepts the lid flag directly.
        let mut index = HNSWIndex::new(DistanceMetric::Cosine, 8, 100, 4, 32);
        let mut rng = 42u64;
        let lcg = |r: &mut u64| -> f32 {
            *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*r >> 33) as f32 / u32::MAX as f32
        };
        let mut entries: Vec<(u64, Vec<f32>)> = (0..500u64).map(|i| {
            (i, (0..32).map(|_| lcg(&mut rng)).collect())
        }).collect();
        if use_lid {
            annex::vector::hnsw::HNSWIndex::sort_by_lid(&mut entries);
        }
        for (id, v) in entries {
            index.insert(id, v).unwrap();
        }
        index
    }

    fn recall(index: &HNSWIndex, queries: &[(u64, Vec<f32>)], truth_k: usize) -> f64 {
        // Use very high ef as near-exact ground truth (index is only 500 vectors).
        let truth_opts = SearchRuntimeOptions { ef_search: Some(450), ..Default::default() };
        let eval_opts  = SearchRuntimeOptions { ef_search: Some(20),  ..Default::default() };
        let total = queries.len() * truth_k;
        let mut hits = 0usize;
        for (_, q) in queries {
            let truth = index.search_with_options(q, truth_k, &truth_opts).unwrap();
            let res   = index.search_with_options(q, truth_k, &eval_opts).unwrap();
            let truth_ids: std::collections::HashSet<_> = truth.iter().map(|r| r.id).collect();
            hits += res.iter().filter(|r| truth_ids.contains(&r.id)).count();
        }
        hits as f64 / total as f64
    }

    let mut rng = 999u64;
    let lcg = |r: &mut u64| -> f32 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*r >> 33) as f32 / u32::MAX as f32
    };
    let queries: Vec<(u64, Vec<f32>)> = (0..50u64).map(|i| {
        (i, (0..32).map(|_| lcg(&mut rng)).collect())
    }).collect();

    let baseline = build(false);
    let lid = build(true);
    let r_base = recall(&baseline, &queries, 10);
    let r_lid = recall(&lid, &queries, 10);
    // LID sort should not regress recall by more than 2pp (probabilistic build).
    assert!(r_lid >= r_base - 0.02, "LID recall={:.3} < baseline={:.3} - 0.02", r_lid, r_base);
}
```

Note: ground truth uses `search_with_options` with a very high ef (≥ index size), which is exact for small test indices. `sort_by_lid` is the function we'll add.

- [ ] **Step 3: Run test to verify it fails**

```bash
cargo test lid_sort_does_not_regress_recall
```
Expected: compile error — `sort_by_lid` not found.

- [ ] **Step 4: Implement LID estimation and sort in `insert.rs`**

Add a public function (associated function on HNSWIndex):

```rust
impl HNSWIndex {
    /// Sort a batch of (id, vector) pairs by descending estimated Local Intrinsic
    /// Dimensionality. High-LID (hub/outlier) points are inserted first so they
    /// propagate to upper layers and improve routing quality.
    ///
    /// LID estimation: for each vector, sample min(32, n-1) other vectors from the
    /// batch, compute cosine distances, sort, and use the Hill estimator:
    ///   LID ≈ −k / Σ_{i=1..k} log(d_i / d_k)
    /// where d_1 ≤ d_2 ≤ … ≤ d_k are the k-nearest distances from the sample.
    pub fn sort_by_lid(entries: &mut Vec<(u64, Vec<f32>)>) {
        let n = entries.len();
        if n < 4 {
            return;
        }
        let sample_k = 16usize.min(n - 1);
        let sample_n = 32usize.min(n - 1);

        // Pre-normalize for cosine.
        let normed: Vec<Vec<f32>> = entries.iter().map(|(_, v)| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 { v.iter().map(|x| x / norm).collect() } else { v.clone() }
        }).collect();

        let lids: Vec<f32> = (0..n).map(|i| {
            // Sample indices (stride through the batch deterministically).
            let stride = n / sample_n + 1;
            let mut dists: Vec<f32> = (0..sample_n).map(|s| {
                let j = (i + 1 + s * stride) % n;
                let dot: f32 = normed[i].iter().zip(normed[j].iter()).map(|(a, b)| a * b).sum();
                (1.0 - dot).max(0.0)
            }).collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            dists.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
            let k = dists.len().min(sample_k);
            if k < 2 {
                return 0.0;
            }
            let d_k = dists[k - 1];
            if d_k < 1e-9 {
                return 0.0;
            }
            let sum_log: f32 = dists[..k].iter()
                .map(|&d| (d / d_k).max(1e-9).ln())
                .sum();
            if sum_log.abs() < 1e-9 { 0.0 } else { -(k as f32) / sum_log }
        }).collect();

        // Sort: descending LID (highest LID first).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| lids[b].partial_cmp(&lids[a]).unwrap());
        let mut sorted = Vec::with_capacity(n);
        for i in order {
            sorted.push(entries[i].clone());
        }
        *entries = sorted;
    }
}
```

- [ ] **Step 5: Wire into `par_insert_batch`**

In `par_insert_batch` in `insert.rs`, add after the `if n_link == 0 { return Ok(0); }` check (around line 268, before Phase 1):

```rust
use super::config::lid_sort_enabled;
if lid_sort_enabled() {
    Self::sort_by_lid(&mut entries_owned); // entries_owned is the mutable Vec
}
```

`par_insert_batch` takes `entries: &[(PointId, Vector)]` (a slice). Do NOT change the signature. Inside `par_insert_batch`, add an owned indirection vector at the top of the function body, before Phase 1:

```rust
use super::config::lid_sort_enabled;
// LID sort: build a permuted view of entries without copying vectors.
let sort_order: Vec<usize> = if lid_sort_enabled() && entries.len() > 3 {
    let mut owned: Vec<(u64, Vec<f32>)> = entries.iter()
        .map(|(id, v)| (*id, v.clone()))
        .collect();
    Self::sort_by_lid(&mut owned);
    // Map sorted IDs back to original slice positions.
    let id_to_orig: std::collections::HashMap<u64, usize> = entries.iter()
        .enumerate()
        .map(|(i, (id, _))| (*id, i))
        .collect();
    owned.iter().map(|(id, _)| id_to_orig[id]).collect()
} else {
    (0..entries.len()).collect()
};
```

Then wherever Phase 1 iterates `entries` to allocate nodes (the sequential loop over `entries`), replace `entries[i]` with `entries[sort_order[i]]`. Phase 2 (parallel search + link) already uses the allocated node indices, so it is unaffected — only the allocation order changes.

- [ ] **Step 6: Run tests**

```bash
cargo test
```
Expected: all non-ignored tests pass including `lid_sort_does_not_regress_recall`.

- [ ] **Step 7: Benchmark LID sort on NYT-256 (build-time recall test)**

LID sort only helps fresh builds. Benchmark: build a fresh index from NYT data with and without LID sort, compare recall at ef=64,128.

Add a `#[test] #[ignore] fn nytimes_lid_build_recall()` in `tests/nytimes_frontier.rs` that:
1. Reads the raw NYT vectors from `data/nytimes-256-angular/train.npy`
2. Builds two fresh `HNSWIndex` (with/without `VECTORDB_LID_SORT`)
3. Runs recall@20 at ef=64,128 against ground truth

```bash
VECTORDB_LID_SORT=true cargo test --release --test nytimes_frontier nytimes_lid_build_recall -- --ignored --nocapture
```

- [ ] **Step 8: Commit**

```bash
git add src/vector/hnsw/insert.rs src/vector/hnsw/config.rs tests/hnsw.rs \
        tests/nytimes_frontier.rs
git commit -m "Add LID-ordered bulk insertion; high-LID vectors inserted first for better routing"
```

---

## Execution order and branching

Tasks 2–5 are independent. Recommended parallel branch strategy:
- `perf/ti-skip` — Task 2
- `perf/rcm-reorder` — Task 3
- `perf/sq8-rerank` — Task 4
- `perf/lid-insert` — Task 5
- Task 1 runs on `perf/recall-frontier` (current branch) since it's benchmark-only

After each PR is benchmarked and confirmed, merge to master. The frontier benchmark accumulates comparison JSON in `docs/recall-investigation-results.json`.
