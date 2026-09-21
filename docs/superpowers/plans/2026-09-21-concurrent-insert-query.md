# Concurrent Insert + Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `HNSWIndex::insert` and `search` safe to call concurrently from multiple threads by replacing `&mut self` inserts with interior-mutability inserts backed by stable-address storage.

**Architecture:** Five sequential tasks introduce primitives bottom-up: (1) `VectorArena` with snapshot views eliminates the vector reallocation hazard; (2) `ChunkedArray<T>` generalises the same stable-slot pattern to all node-indexed data; (3) `HNSWIndex` internals are swapped to the new primitives while keeping `&mut self` and all existing tests green; (4) the outer layer containers get the same treatment; (5) `AllocState`, `NodeState` publication ordering, and the `&self` insert flip complete the concurrent API. Each task compiles and all tests pass before the next begins.

**Tech Stack:** Rust stable 1.98, `parking_lot` (already in Cargo.toml), `std::sync::atomic`

**Spec:** This document plus the architectural discussion summarised at
`/Users/rohansharma/.claude/projects/-Users-rohansharma-Desktop-Code-vectordb/memory/project_strategy.md`

## Global Constraints

- No new crate dependencies — all synchronisation primitives from `std` or the existing `parking_lot` crate
- Snapshot version stays at 2; no new persisted fields
- Every task ends with `cargo test`, `cargo fmt --all -- --check`, `cargo clippy --all-targets` passing
- `vector_slice(&self, idx) -> &[f32]` **must not be removed in Task 3** — it is called in 35+ sites; a migration helper `vector_slice_legacy` will bridge until Task 6
- `insert(&mut self)` stays `&mut self` through Task 4; only Task 5 flips it
- All atomics on the hot query path (entry_point, max_level, node_state reads) use `Ordering::Acquire`; corresponding writes use `Ordering::Release`
- Entry point and max level are always observed as a consistent pair — packed into one `AtomicU64`: upper 32 bits = max_level, lower 32 bits = entry_idx (u32::MAX = None)

---

## Task 1: VectorArena — stable-address chunk store with per-search snapshot views

**Files:**
- Create: `src/vector/hnsw/arena.rs`
- Modify: `src/vector/hnsw/mod.rs` (add `pub(crate) mod arena`)
- Test: `tests/hnsw.rs` (add `vector_arena_*` tests)

**Interfaces:**
- Produces:
  - `pub struct VectorArena` — the arena; accepts push, returns views
  - `pub struct VectorArenaView` — snapshot, `&self` lifetime; `get(idx) -> &[f32]` is zero-cost after construction
  - `VectorArena::new(dim: usize, chunk_capacity: usize) -> Self`
  - `VectorArena::push(&self, vec: &[f32]) -> usize` — returns the new node's idx; acquires `writer` Mutex briefly
  - `VectorArena::view(&self) -> VectorArenaView` — acquires RwLock read briefly, clones `Arc<[Arc<VectorChunk>]>`, returns it; holds no lock afterwards
  - `VectorArena::len(&self) -> usize`
  - `VectorArenaView::get(&self, idx: usize) -> &[f32]` — no locks, no Arc operations; lifetime tied to view

**Key invariant:** `VectorChunk::data` is a `Box<[f32]>` of exactly `chunk_capacity * dim` f32 values, pre-allocated at chunk creation and **never resized**. Addresses inside it are stable for the chunk's lifetime. Chunks are kept alive by `Arc<VectorChunk>`.

- [ ] **Step 1: Write failing tests in `tests/hnsw.rs`**

```rust
#[test]
fn vector_arena_push_and_get() {
    use annex::vector::hnsw::arena::{VectorArena, VectorArenaView};
    let arena = VectorArena::new(4, 8); // dim=4, 8 vectors per chunk
    let v0 = vec![1.0f32, 2.0, 3.0, 4.0];
    let v1 = vec![5.0f32, 6.0, 7.0, 8.0];
    let idx0 = arena.push(&v0);
    let idx1 = arena.push(&v1);
    assert_eq!(idx0, 0);
    assert_eq!(idx1, 1);
    let view = arena.view();
    assert_eq!(view.get(0), v0.as_slice());
    assert_eq!(view.get(1), v1.as_slice());
    assert_eq!(arena.len(), 2);
}

#[test]
fn vector_arena_crosses_chunk_boundary() {
    use annex::vector::hnsw::arena::VectorArena;
    let arena = VectorArena::new(2, 4); // 4 vectors per chunk
    for i in 0..10u64 {
        let v = vec![i as f32, i as f32 * 2.0];
        arena.push(&v);
    }
    let view = arena.view();
    for i in 0..10u64 {
        let got = view.get(i as usize);
        assert_eq!(got[0], i as f32);
        assert_eq!(got[1], i as f32 * 2.0);
    }
}

#[test]
fn vector_arena_view_survives_concurrent_push() {
    // A view snapshot keeps previously-pushed vectors accessible even
    // while new vectors are being pushed concurrently.
    use annex::vector::hnsw::arena::VectorArena;
    use std::sync::Arc;
    let arena = Arc::new(VectorArena::new(2, 4));
    arena.push(&[1.0, 2.0]);
    arena.push(&[3.0, 4.0]);
    let view = arena.view();
    // Snapshot taken — now push more in a separate thread
    let arena2 = arena.clone();
    let handle = std::thread::spawn(move || {
        for i in 0..20 {
            arena2.push(&[i as f32, i as f32]);
        }
    });
    // Original view still valid, no crash, correct values
    assert_eq!(view.get(0), &[1.0f32, 2.0]);
    assert_eq!(view.get(1), &[3.0f32, 4.0]);
    handle.join().unwrap();
}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cargo test vector_arena 2>&1 | tail -5
```
Expected: compile error — module `arena` not found.

- [ ] **Step 3: Create `src/vector/hnsw/arena.rs`**

```rust
use std::sync::Arc;
use parking_lot::{Mutex, RwLock};

/// Fixed-capacity chunk of pre-allocated f32 storage.
/// `data` is allocated once and never resized — addresses are permanently stable.
pub(crate) struct VectorChunk {
    data: Box<[f32]>,
}

impl VectorChunk {
    fn new(chunk_capacity: usize, dim: usize) -> Arc<Self> {
        Arc::new(Self {
            data: vec![0.0f32; chunk_capacity * dim].into_boxed_slice(),
        })
    }

    /// Return the slice for the `local_idx`-th vector in this chunk.
    /// SAFETY: caller must ensure local_idx < chunk_capacity and vector was initialized.
    #[inline]
    fn get(&self, local_idx: usize, dim: usize) -> &[f32] {
        &self.data[local_idx * dim..(local_idx + 1) * dim]
    }

    fn write(&self, local_idx: usize, dim: usize, vec: &[f32]) {
        let start = local_idx * dim;
        // SAFETY: we hold the writer mutex, so no other thread writes this slot.
        // Readers may observe the old (zeroed) values until the push returns,
        // but node_state = Reserved prevents any reader from accessing this slot.
        let dest = &self.data[start..start + dim] as *const [f32] as *mut [f32];
        unsafe { (*dest).copy_from_slice(vec) };
    }
}

struct WriterState {
    total: usize,
    chunks: Vec<Arc<VectorChunk>>,
}

/// Stable-address vector storage with per-search snapshot views.
///
/// Chunks are never resized after creation. Growing the arena appends a new chunk.
/// Readers acquire one `Arc<[Arc<VectorChunk>]>` snapshot per search, then access
/// vectors with zero locks or refcount operations.
pub struct VectorArena {
    /// Immutable snapshot of current chunk list. Updated (swapped) atomically when
    /// a new chunk is added. Readers take a short read-lock to clone the Arc.
    chunks: RwLock<Arc<[Arc<VectorChunk>]>>,
    dim: usize,
    chunk_capacity: usize,
    writer: Mutex<WriterState>,
}

impl VectorArena {
    pub fn new(dim: usize, chunk_capacity: usize) -> Self {
        Self {
            chunks: RwLock::new(Arc::from(Vec::<Arc<VectorChunk>>::new())),
            dim,
            chunk_capacity,
            writer: Mutex::new(WriterState { total: 0, chunks: Vec::new() }),
        }
    }

    /// Append one vector, return its stable index.
    /// Acquires writer mutex briefly; may briefly acquire chunks write-lock when adding a chunk.
    pub fn push(&self, vec: &[f32]) -> usize {
        debug_assert_eq!(vec.len(), self.dim);
        let mut w = self.writer.lock();
        let idx = w.total;
        let local_idx = idx % self.chunk_capacity;

        if local_idx == 0 {
            // Need a new chunk
            let chunk = VectorChunk::new(self.chunk_capacity, self.dim);
            chunk.write(0, self.dim, vec);
            w.chunks.push(chunk.clone());
            // Publish the updated chunk list atomically
            let new_snapshot: Arc<[Arc<VectorChunk>]> = w.chunks.clone().into();
            *self.chunks.write() = new_snapshot;
        } else {
            let chunk = &w.chunks[idx / self.chunk_capacity];
            chunk.write(local_idx, self.dim, vec);
        }
        w.total += 1;
        idx
    }

    /// Return a snapshot view. Holds one Arc clone for the duration of the search.
    /// No locks held after this call returns.
    pub fn view(&self) -> VectorArenaView {
        VectorArenaView {
            chunks: self.chunks.read().clone(),
            dim: self.dim,
            chunk_capacity: self.chunk_capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.writer.lock().total
    }

    /// Direct indexed access — use only in single-threaded contexts (snapshot, reorder).
    /// Panics if idx >= len.
    pub fn get_direct(&self, idx: usize) -> &[f32] {
        let chunks = self.chunks.read();
        let chunk = &chunks[idx / self.chunk_capacity];
        // SAFETY: chunk is Arc-kept alive; we hold the read lock on chunks.
        // This is used only during serialised operations (build, reorder, snapshot).
        let ptr = chunk.get(idx % self.chunk_capacity, self.dim).as_ptr();
        unsafe { std::slice::from_raw_parts(ptr, self.dim) }
    }
}

/// Snapshot of the arena's chunk list at a point in time.
/// Accessing previously-pushed vectors via `get` is zero-cost (no locks, no refcounts).
/// Vectors pushed AFTER `view()` was called are NOT visible through this snapshot.
pub struct VectorArenaView {
    chunks: Arc<[Arc<VectorChunk>]>,
    dim: usize,
    chunk_capacity: usize,
}

impl VectorArenaView {
    /// Return the vector at `idx`. Panics if idx was not pushed before this view was taken.
    #[inline]
    pub fn get(&self, idx: usize) -> &[f32] {
        self.chunks[idx / self.chunk_capacity].get(idx % self.chunk_capacity, self.dim)
    }
}
```

- [ ] **Step 4: Export `VectorArena` and `VectorArenaView` from `arena.rs`**

Add to `src/vector/hnsw/mod.rs`:
```rust
pub(crate) mod arena;
```

For tests to access `VectorArena`, add a `pub use` in `src/lib.rs`:
```rust
pub use vector::hnsw::arena::{VectorArena, VectorArenaView};
```

- [ ] **Step 5: Run tests**

```bash
cargo test vector_arena 2>&1 | tail -5
```
Expected: all 3 tests pass.

- [ ] **Step 6: Run full suite, fmt, clippy**

```bash
cargo test && cargo fmt --all -- --check && cargo clippy --all-targets 2>&1 | grep "^error" | head -5
```
Expected: 0 failures, 0 errors.

- [ ] **Step 7: Commit**

```bash
git add src/vector/hnsw/arena.rs src/vector/hnsw/mod.rs src/lib.rs tests/hnsw.rs
git commit -m "Add VectorArena: stable-address chunk store with per-search snapshot views"
```

---

## Task 2: ChunkedArray<T> — stable-slot append-only array for all other node-indexed data

**Files:**
- Modify: `src/vector/hnsw/arena.rs`
- Test: `tests/hnsw.rs` (add `chunked_array_*` tests)

**Interfaces:**
- Consumes: (no Task 1 types used here)
- Produces:
  - `pub(crate) struct ChunkedArray<T: Default + Send + Sync>`
  - `ChunkedArray::new(chunk_capacity: usize) -> Self`
  - `ChunkedArray::push_default(&self) -> usize` — allocates one slot at its default value; returns idx
  - `ChunkedArray::get(&self, idx: usize) -> &T` — stable reference, valid while `ChunkedArray` lives
  - `ChunkedArray::len(&self) -> usize`
  - `ChunkedArray::view(&self) -> ChunkedArrayView<T>` — snapshot for zero-lock hot-path access
  - `ChunkedArrayView::get(&self, idx: usize) -> &T`

**Key constraint:** `T::default()` is called exactly once per slot at allocation time. Callers mutate through interior mutability (`AtomicBool`, `AtomicU8`, `RwLock<Vec<usize>>`). `Box<[T]>` per chunk — T is not moved after init.

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn chunked_array_atomic_bool() {
    use annex::vector::hnsw::arena::ChunkedArray;
    use std::sync::atomic::{AtomicBool, Ordering};
    let arr: ChunkedArray<AtomicBool> = ChunkedArray::new(4);
    let i0 = arr.push_default();
    let i1 = arr.push_default();
    assert_eq!(i0, 0); assert_eq!(i1, 1);
    assert_eq!(arr.get(0).load(Ordering::Relaxed), false);
    arr.get(0).store(true, Ordering::Relaxed);
    assert_eq!(arr.get(0).load(Ordering::Relaxed), true);
    assert_eq!(arr.get(1).load(Ordering::Relaxed), false);
}

#[test]
fn chunked_array_view_stable_after_push() {
    use annex::vector::hnsw::arena::ChunkedArray;
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::Arc;
    let arr: Arc<ChunkedArray<AtomicU8>> = Arc::new(ChunkedArray::new(4));
    arr.push_default(); arr.push_default();
    arr.get(0).store(7, Ordering::Relaxed);
    let view = arr.view();
    // Push more concurrently
    let arr2 = arr.clone();
    let h = std::thread::spawn(move || {
        for _ in 0..20 { arr2.push_default(); }
    });
    // View still valid
    assert_eq!(view.get(0).load(Ordering::Relaxed), 7);
    h.join().unwrap();
}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cargo test chunked_array 2>&1 | tail -5
```
Expected: compile error.

- [ ] **Step 3: Implement `ChunkedArray<T>` in `src/vector/hnsw/arena.rs`**

Add after the `VectorArenaView` impl:

```rust
use std::mem::MaybeUninit;

struct ChunkedArrayChunk<T> {
    data: Box<[MaybeUninit<T>]>,
    capacity: usize,
}

impl<T: Default> ChunkedArrayChunk<T> {
    fn new(capacity: usize) -> Arc<Self> {
        let mut v = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            v.push(MaybeUninit::new(T::default()));
        }
        Arc::new(Self { data: v.into_boxed_slice(), capacity })
    }

    /// SAFETY: idx < capacity and slot was initialized by new().
    #[inline]
    unsafe fn get(&self, idx: usize) -> &T {
        self.data[idx].assume_init_ref()
    }
}

struct ChunkedArrayWriter {
    total: usize,
    chunks: Vec<Arc<dyn std::any::Any + Send + Sync>>,
}

pub(crate) struct ChunkedArray<T: Default + Send + Sync + 'static> {
    chunks: RwLock<Arc<[Arc<ChunkedArrayChunk<T>>]>>,
    chunk_capacity: usize,
    writer: Mutex<(usize, Vec<Arc<ChunkedArrayChunk<T>>>)>, // (total, chunks)
}

impl<T: Default + Send + Sync + 'static> ChunkedArray<T> {
    pub(crate) fn new(chunk_capacity: usize) -> Self {
        Self {
            chunks: RwLock::new(Arc::from(Vec::<Arc<ChunkedArrayChunk<T>>>::new())),
            chunk_capacity,
            writer: Mutex::new((0, Vec::new())),
        }
    }

    pub(crate) fn push_default(&self) -> usize {
        let mut w = self.writer.lock();
        let idx = w.0;
        let local_idx = idx % self.chunk_capacity;
        if local_idx == 0 {
            let chunk = ChunkedArrayChunk::<T>::new(self.chunk_capacity);
            w.1.push(chunk);
            let new_snap: Arc<[Arc<ChunkedArrayChunk<T>>]> = w.1.clone().into();
            *self.chunks.write() = new_snap;
        }
        w.0 += 1;
        idx
    }

    pub(crate) fn get(&self, idx: usize) -> &T {
        let chunks = self.chunks.read();
        let chunk = chunks[idx / self.chunk_capacity].clone();
        // SAFETY: slot was initialized in push_default via T::default()
        let ptr = unsafe { chunk.get(idx % self.chunk_capacity) as *const T };
        // SAFETY: chunk kept alive by Arc stored in self.chunks (held by RwLock read guard
        // which is alive as long as caller holds the return value... but we drop the guard here).
        // This is sound only while ChunkedArray lives — document that constraint.
        drop(chunks);
        unsafe { &*ptr }
    }

    pub(crate) fn view(&self) -> ChunkedArrayView<T> {
        ChunkedArrayView {
            chunks: self.chunks.read().clone(),
            chunk_capacity: self.chunk_capacity,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.writer.lock().0
    }
}

pub(crate) struct ChunkedArrayView<T: Send + Sync + 'static> {
    chunks: Arc<[Arc<ChunkedArrayChunk<T>>]>,
    chunk_capacity: usize,
}

impl<T: Default + Send + Sync + 'static> ChunkedArrayView<T> {
    #[inline]
    pub(crate) fn get(&self, idx: usize) -> &T {
        // SAFETY: idx was pushed before view was taken; Arc keeps chunk alive.
        unsafe { self.chunks[idx / self.chunk_capacity].get(idx % self.chunk_capacity) }
    }
}
```

Note: `ChunkedArray::get` drops the read guard before returning a raw pointer — this is safe because the `Arc<ChunkedArrayChunk<T>>` array is only ever extended (never mutated), and Arc keeps chunks alive as long as any reference (in `self.chunks` or a view) exists. The backing `Box<[MaybeUninit<T>]>` never moves. Add a `// SAFETY` doc comment explaining this invariant.

Export `ChunkedArray` and `ChunkedArrayView` alongside `VectorArena` in `src/lib.rs`.

- [ ] **Step 4: Run tests and verify**

```bash
cargo test chunked_array && cargo test vector_arena 2>&1 | tail -5
```
Expected: all pass.

- [ ] **Step 5: Full suite, fmt, clippy, commit**

```bash
cargo test && cargo fmt --all -- --check && cargo clippy --all-targets 2>&1 | grep "^error"
git add src/vector/hnsw/arena.rs src/lib.rs tests/hnsw.rs
git commit -m "Add ChunkedArray<T>: stable-slot append-only array for concurrent node-indexed data"
```

---

## Task 3: Swap HNSWIndex vector, deleted, level, and entry-point fields

**Files:**
- Modify: `src/vector/hnsw/core.rs`
- Modify: `src/vector/hnsw/snapshot.rs` (update `from_snapshot` and `to_snapshot`)
- Modify: `src/vector/hnsw/insert.rs` (update `register_node` and related helpers)
- Test: `tests/hnsw.rs`, `tests/persist.rs` — existing tests must still pass

**Interfaces:**
- Consumes: `VectorArena`, `VectorArenaView`, `ChunkedArray<AtomicBool>`, `ChunkedArray<AtomicU8>` from Task 2
- Produces (new `HNSWIndex` fields replacing the old ones):
  - `vectors: VectorArena` — replaces `vectors: Vec<f32>`
  - `deleted: ChunkedArray<AtomicBool>` — replaces `deleted: Vec<bool>`, `deleted_count: usize`
  - `levels: ChunkedArray<AtomicU8>` — replaces `levels: Vec<usize>` (levels fit in u8; max_level_cap ≤ 255)
  - `entry_level_ep: AtomicU64` — replaces `entry_point: Option<usize>` + `current_max_level: usize`; encoding: upper 32 bits = max_level, lower 32 bits = entry_idx; `u32::MAX` in lower bits = no entry point
  - Remove: `alloc_lock: RwLock<()>` (dead field)
  - Keep: `point_to_idx: HashMap<PointId, usize>`, `idx_to_point: Vec<PointId>`, `deleted_count: usize` temporarily (will move to AllocState in Task 5)
- `vector_slice(&self, idx: usize) -> &[f32]` **must be preserved** but now reads from `VectorArena` via `get_direct()` (single-threaded contexts only); hot-path users in search.rs are updated in Task 6

**Entry-level-ep encoding helpers:**

```rust
const NO_EP: u32 = u32::MAX;

fn pack_ep(entry_idx: usize, max_level: usize) -> u64 {
    ((max_level as u64) << 32) | (entry_idx as u64)
}

fn unpack_ep(v: u64) -> (Option<usize>, usize) {
    let idx = (v & 0xffff_ffff) as u32;
    let level = (v >> 32) as usize;
    if idx == NO_EP { (None, level) } else { (Some(idx as usize), level) }
}
```

**`register_node` changes** (in `insert.rs`):
- `self.vectors.push(vector)` — returns idx
- `self.deleted.push_default()` — AtomicBool::default() = false
- `self.levels.push_default()` — then `self.levels.get(idx).store(level as u8, Ordering::Relaxed)`
- Still takes `&mut self` in Task 3 (flip happens in Task 5)

**`mark_deleted` changes**:
- `self.deleted.get(idx).store(true, Ordering::Release)` — Release so search threads see the deletion
- `deleted_count` still maintained as before (cleaned up in Task 5)

**`find_highest_level_entry_point` changes**: iterate levels via `ChunkedArray::get(i)` instead of `self.levels.get(i)`

**Snapshot round-trip**:
- `to_snapshot`: read vectors via `self.vectors.get_direct(idx)`, read deleted via `self.deleted.get(idx).load(Ordering::Relaxed)`, read levels via `self.levels.get(idx).load(Ordering::Relaxed)`
- `from_snapshot`: call `self.vectors.push(vec)`, `self.deleted.push_default()` (and store if deleted), `self.levels.push_default()` (and store level)

- [ ] **Step 1: Write failing test that will detect the field change**

```rust
#[test]
fn fields_changed_sanity() {
    // This test just verifies the refactored fields work correctly.
    // It tests deleted and levels through the existing public API.
    use annex::vector::hnsw::HNSWIndex;
    use annex::utils::types::DistanceMetric;
    let mut idx = HNSWIndex::new(DistanceMetric::Cosine, 4, 20, 4, 4);
    for i in 0u64..10 {
        idx.insert(i, vec![i as f32, 0.0, 0.0, 0.0]).unwrap();
    }
    assert_eq!(idx.len(), 10);
    idx.mark_deleted(3);
    assert_eq!(idx.deleted_count(), 1);
    // Snapshot round-trip must be lossless
    let snap = idx.to_snapshot();
    let restored = HNSWIndex::from_snapshot(snap);
    assert_eq!(restored.len(), 10);
    assert_eq!(restored.deleted_count(), 1);
    assert!(restored.search(&vec![3.0, 0.0, 0.0, 0.0], 5).unwrap()
        .iter().all(|r| r.id != 3));
}
```

- [ ] **Step 2: Run test to verify it passes before changes** (baseline)

```bash
cargo test fields_changed_sanity 2>&1 | tail -3
```
Expected: PASS (it uses public API only; should pass on current code).

- [ ] **Step 3: Replace struct fields in `src/vector/hnsw/core.rs`**

Remove these fields from `HNSWIndex`:
```rust
pub(crate) vectors: Vec<f32>,
pub(crate) levels: Vec<usize>,
pub(crate) deleted: Vec<bool>,
pub(crate) deleted_count: usize,
pub(crate) entry_point: Option<usize>,
pub(crate) current_max_level: usize,
pub(crate) alloc_lock: RwLock<()>,
```

Add these:
```rust
pub(crate) vectors: crate::vector::hnsw::arena::VectorArena,
pub(crate) deleted: crate::vector::hnsw::arena::ChunkedArray<std::sync::atomic::AtomicBool>,
pub(crate) levels: crate::vector::hnsw::arena::ChunkedArray<std::sync::atomic::AtomicU8>,
pub(crate) entry_level_ep: std::sync::atomic::AtomicU64,
// Temporary: move to AllocState in Task 5
pub(crate) deleted_count: std::sync::atomic::AtomicUsize,
pub(crate) point_to_idx: HashMap<PointId, usize>,
pub(crate) idx_to_point: Vec<PointId>,
```

Add the helpers `pack_ep` / `unpack_ep` and constants `NO_EP` as free functions in `core.rs`.

- [ ] **Step 4: Update `HNSWIndex::new()` initialiser**

```rust
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize};
use crate::vector::hnsw::arena::{ChunkedArray, VectorArena};

const CHUNK_CAP: usize = 4096;

Self {
    layers: Vec::new(),
    vectors: VectorArena::new(dim, CHUNK_CAP),
    levels: ChunkedArray::new(CHUNK_CAP),
    deleted: ChunkedArray::new(CHUNK_CAP),
    entry_level_ep: AtomicU64::new(pack_ep(NO_EP as usize, 0)),
    deleted_count: AtomicUsize::new(0),
    edge_dists_l0: Vec::new(),
    quantized: Vec::new(),
    quant_min: Vec::new(),
    quant_scale: Vec::new(),
    point_to_idx: HashMap::new(),
    idx_to_point: Vec::new(),
    exact_fallback_enabled: ...,
    exact_fallback_threshold: ...,
}
```

- [ ] **Step 5: Update every read/write of the replaced fields throughout `core.rs`, `insert.rs`, `snapshot.rs`**

For each occurrence:

| Old expression | New expression |
|---|---|
| `self.entry_point` | `{ let (ep, _) = unpack_ep(self.entry_level_ep.load(Ordering::Acquire)); ep }` |
| `self.current_max_level` | `{ let (_, ml) = unpack_ep(self.entry_level_ep.load(Ordering::Acquire)); ml }` |
| `self.entry_point = Some(idx); self.current_max_level = level` | `self.entry_level_ep.store(pack_ep(idx, level), Ordering::Release)` |
| `self.vectors[idx*dim..(idx+1)*dim]` | `self.vectors.get_direct(idx)` |
| `self.vectors.extend_from_slice(&vector)` | `self.vectors.push(&vector)` (in `register_node`) |
| `self.levels.push(level)` | `self.levels.push_default(); self.levels.get(idx).store(level as u8, Ordering::Relaxed)` |
| `self.levels.get(idx).copied()` | `Some(self.levels.get(idx).load(Ordering::Relaxed) as usize)` |
| `self.deleted.push(false)` | `self.deleted.push_default()` |
| `self.deleted.get(idx).copied().unwrap_or(false)` | `self.deleted.get(idx).load(Ordering::Acquire)` |
| `self.deleted.get_mut(idx)` then `*flag = true` | `self.deleted.get(idx).store(true, Ordering::Release)` |
| `self.deleted_count += 1` | `self.deleted_count.fetch_add(1, Ordering::Relaxed)` |
| `self.deleted_count` (read) | `self.deleted_count.load(Ordering::Relaxed)` |
| `self.idx_to_point.len()` | unchanged (len() still returns this) |
| `alloc_lock` | remove all references |

`vector_slice` stub for legacy callers (will be removed in Task 6):
```rust
#[inline]
pub(crate) fn vector_slice(&self, idx: usize) -> &[f32] {
    self.vectors.get_direct(idx)
}
```

- [ ] **Step 6: Run full test suite**

```bash
cargo test 2>&1 | grep -E "^test result:|FAILED"
```
Expected: all pass.

- [ ] **Step 7: fmt, clippy, commit**

```bash
cargo fmt --all && cargo clippy --all-targets 2>&1 | grep "^error"
git add src/vector/hnsw/core.rs src/vector/hnsw/insert.rs src/vector/hnsw/snapshot.rs
git commit -m "Replace Vec fields with VectorArena and ChunkedArray; entry_point → AtomicU64"
```

---

## Task 4: Stable-slot outer containers for layers and edge_dists_l0

**Files:**
- Modify: `src/vector/hnsw/core.rs` (struct field types, helpers)
- Modify: `src/vector/hnsw/insert.rs` (`extend_layers_for_new_node`, `register_node`)
- Modify: `src/vector/hnsw/snapshot.rs`
- Test: `tests/hnsw.rs`, `tests/persist.rs` — all existing tests must pass

**Problem:** `layers: Vec<Vec<RwLock<Vec<usize>>>>` — `layers[level]` is a `Vec` indexed by node idx. When a new node is added, `layers[level].push(RwLock::new(...))` can reallocate that Vec, invalidating any concurrent reader holding `&layers[level][old_idx]`.

**Fix:** Replace each `layers[level]` with `ChunkedArray<parking_lot::RwLock<Vec<usize>>>`. The level dimension is small and fixed (pre-allocated at `max_level_cap + 1` entries at construction time). Each level's `ChunkedArray` grows by node-idx as nodes are added.

**New type:**
```rust
// Each level of the graph is a stable-slot array of neighbor lists.
pub(crate) layers: Vec<ChunkedArray<parking_lot::RwLock<Vec<usize>>>>,
// Edge distances at L0 — same pattern.
pub(crate) edge_dists_l0: ChunkedArray<parking_lot::RwLock<Vec<f32>>>,
```

The outer `Vec<ChunkedArray<...>>` for `layers` is pre-allocated to `max_level_cap + 1` at construction and **never grown** — level indices are bounded by `max_level_cap`. `ensure_level_capacity` becomes a no-op (or an assertion) since all level slots exist at construction.

`RwLock<Vec<usize>>` implements `Default` (unlocked lock, empty Vec) — so `ChunkedArray<RwLock<Vec<usize>>>` works with `push_default()`.

- [ ] **Step 1: Verify existing tests pass (baseline)**

```bash
cargo test --test hnsw --test persist 2>&1 | grep "^test result:"
```

- [ ] **Step 2: Update struct definition in `core.rs`**

```rust
pub(crate) layers: Vec<crate::vector::hnsw::arena::ChunkedArray<parking_lot::RwLock<Vec<usize>>>>,
pub(crate) edge_dists_l0: crate::vector::hnsw::arena::ChunkedArray<parking_lot::RwLock<Vec<f32>>>,
```

- [ ] **Step 3: Update `HNSWIndex::new()`**

Pre-allocate all level slots:
```rust
let layers = (0..=max_level_cap)
    .map(|_| ChunkedArray::new(CHUNK_CAP))
    .collect();
let edge_dists_l0 = ChunkedArray::new(CHUNK_CAP);
```

- [ ] **Step 4: Update `register_node` in `insert.rs`**

Replace `self.layers[level].push(RwLock::new(...))` pattern:
```rust
// Allocate a slot in each level's ChunkedArray
for l in 0..=self.layers.len() {
    self.layers[l].push_default(); // RwLock<Vec<usize>> default = unlocked empty list
}
self.edge_dists_l0.push_default();
```

The returned idx from `VectorArena::push` equals the idx used here since both grow in lockstep.

- [ ] **Step 5: Update every `self.layers[level][node_idx]` access**

Old: `self.layers[level][node_idx].read()` / `.write()`
New: `self.layers[level].get(node_idx).read()` / `.write()`

Old: `self.edge_dists_l0[node_idx]`
New: `self.edge_dists_l0.get(node_idx)`

Remove `ensure_level_capacity` and `extend_layers_for_new_node` (now no-ops since all level slots are pre-allocated and each node alloc adds exactly one slot to each level).

- [ ] **Step 6: Update snapshot round-trip**

`to_snapshot`: iterate `for level in 0..=self.current_max_level()` and for each level iterate `for node_idx in 0..self.len()`, calling `self.layers[level].get(node_idx).read()` to read neighbor lists.

`from_snapshot`: call `self.layers[level].push_default()` per node per level, then write the list under `.write()`.

- [ ] **Step 7: Run full test suite**

```bash
cargo test 2>&1 | grep -E "^test result:|FAILED"
```
Expected: all pass.

- [ ] **Step 8: fmt, clippy, commit**

```bash
cargo fmt --all && cargo clippy --all-targets 2>&1 | grep "^error"
git add src/vector/hnsw/core.rs src/vector/hnsw/insert.rs src/vector/hnsw/snapshot.rs
git commit -m "Replace layers/edge_dists_l0 Vec growth with ChunkedArray stable slots"
```

---

## Task 5: AllocState, NodeState, and flip insert to `&self`

**Files:**
- Modify: `src/vector/hnsw/core.rs` (add `AllocState`, `NodeState`, update struct)
- Modify: `src/vector/hnsw/insert.rs` (flip `&mut self` → `&self`, implement ALLOC/LINK/PUBLISH)
- Test: `tests/hnsw.rs` (add concurrent insert+query test)

**Interfaces:**
- Consumes: all arena primitives from Tasks 1–4
- Produces:
  - `insert(&self, ...) -> Result<(), DBError>` — concurrent-safe
  - `par_insert_batch(&self, ...) -> Result<usize, DBError>` — concurrent-safe
  - `delete(&self, point_id: PointId)` (becomes `&self` too)
  - Internal: `struct AllocState { point_to_idx: HashMap<PointId, usize>, idx_to_point: Vec<PointId>, node_count: usize }`
  - Internal: `node_state: ChunkedArray<AtomicU8>` with constants `NODE_RESERVED=0u8, NODE_LIVE=1u8, NODE_DELETED=2u8`

**Publication protocol for insert:**

```
ALLOC (under Mutex<AllocState>):
  - check duplicate
  - compute level
  - normalize vector
  - idx = vectors.push(normalized)
  - deleted.push_default()  (false)
  - levels.push_default(); levels.get(idx).store(level, Relaxed)
  - node_state.push_default(); // NODE_RESERVED
  - all layers[l].push_default() for l in 0..=layers.len()
  - edge_dists_l0.push_default()
  - update alloc.point_to_idx, alloc.idx_to_point, alloc.node_count
  release Mutex

LINK (no global lock — per-node RwLocks only):
  - greedy descent through upper layers (reads only live nodes)
  - search_layer_unfiltered at insertion level (reads live nodes)
  - select_diverse_neighbors
  - write own neighbor list: layers[l].get(idx).write()
  - write back-edges: layers[l].get(n).write() for each neighbor n
  - write edge_dists_l0 if enabled

PUBLISH:
  - node_state.get(idx).store(NODE_LIVE, Ordering::Release)
  - update entry_level_ep if new max level (compare-and-swap loop)
```

**Reader filter:** all BFS hot paths check `node_state.get(neighbor).load(Ordering::Acquire) == NODE_LIVE` before considering a neighbor. NODE_RESERVED nodes are invisible to queries — callers see them as deleted. This is the key invariant: a node becomes reachable only after full initialization.

**entry_level_ep update (CAS loop):**
```rust
loop {
    let old = self.entry_level_ep.load(Ordering::Acquire);
    let (_, old_max) = unpack_ep(old);
    if level <= old_max { break; }
    let new_val = pack_ep(idx, level);
    if self.entry_level_ep.compare_exchange_weak(old, new_val,
        Ordering::Release, Ordering::Relaxed).is_ok() { break; }
}
```

- [ ] **Step 1: Write concurrent insert+query test**

```rust
#[test]
fn concurrent_insert_and_query() {
    use annex::vector::hnsw::HNSWIndex;
    use annex::utils::types::DistanceMetric;
    use std::sync::Arc;
    // Build initial index
    let index = Arc::new(HNSWIndex::new(DistanceMetric::Cosine, 8, 50, 4, 16));
    // Pre-populate
    for i in 0u64..100 {
        let v: Vec<f32> = (0..16).map(|d| if d == (i%16) as usize { 1.0 } else { 0.0 }).collect();
        index.insert(i, v).unwrap();
    }
    // Concurrent: inserts on one thread, queries on another
    let index_w = index.clone();
    let insert_thread = std::thread::spawn(move || {
        for i in 100u64..200 {
            let v: Vec<f32> = (0..16).map(|d| if d == (i%16) as usize { 1.0 } else { 0.0 }).collect();
            index_w.insert(i, v).unwrap();
        }
    });
    let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for _ in 0..50 {
        let results = index.search(&query, 5).unwrap();
        // Results must be non-empty and all be live nodes (no corruption)
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.raw_score >= 0.0));
    }
    insert_thread.join().unwrap();
    assert!(index.len() >= 100);
}
```

- [ ] **Step 2: Run to confirm it fails (insert still `&mut self` so Arc::new fails to compile)**

```bash
cargo test concurrent_insert_and_query 2>&1 | head -10
```
Expected: compile error about `&mut self` not being callable on `&Arc`.

- [ ] **Step 3: Add `AllocState` and `node_state` to `HNSWIndex`**

```rust
pub(crate) node_state: crate::vector::hnsw::arena::ChunkedArray<std::sync::atomic::AtomicU8>,
pub(crate) alloc: std::sync::Mutex<AllocState>,
```

Where:
```rust
pub(crate) struct AllocState {
    pub(crate) point_to_idx: HashMap<PointId, usize>,
    pub(crate) idx_to_point: Vec<PointId>,
    pub(crate) node_count: usize,
}
```

Remove `point_to_idx: HashMap<PointId, usize>` and `idx_to_point: Vec<PointId>` from the top-level struct (move into `AllocState`).

Move `deleted_count: AtomicUsize` into `AllocState` too (it's only written at insert/delete time, never on the hot query path).

- [ ] **Step 4: Rewrite `insert()` in `insert.rs` with `&self` and ALLOC/LINK/PUBLISH**

Change signature:
```rust
pub fn insert(&self, point_id: PointId, vector: Vector) -> Result<(), DBError>
```

Implement the three phases as documented above. The `alloc` Mutex is held only during the ALLOC phase. LINK uses existing BFS logic (now reading `node_state` to skip RESERVED nodes). PUBLISH stores NODE_LIVE with Release ordering.

- [ ] **Step 5: Rewrite `par_insert_batch()` with `&self`**

Phase 1 becomes a loop of ALLOC operations under the Mutex, collecting `(idx, level)` pairs. Phase 2 (parallel link) is unchanged but passes `&self` naturally since threads hold `Arc`. Phase 3 (entry point update) uses the CAS loop.

- [ ] **Step 6: Update BFS to skip NODE_RESERVED nodes**

In `search_layer_unfiltered` (`search.rs`), change the deleted check:
```rust
// Old:
if self.deleted.get(neighbor).copied().unwrap_or(false) || !scratch.mark_visited(neighbor)
// New:
let state = self.node_state.get(neighbor).load(Ordering::Acquire);
if state != NODE_LIVE || !scratch.mark_visited(neighbor)
```

This single change makes queries skip both RESERVED (inserting) and DELETED nodes.

- [ ] **Step 7: Run full test suite including concurrent test**

```bash
cargo test 2>&1 | grep -E "^test result:|FAILED"
```
Expected: all pass including `concurrent_insert_and_query`.

- [ ] **Step 8: fmt, clippy, commit**

```bash
cargo fmt --all && cargo clippy --all-targets 2>&1 | grep "^error"
git add src/vector/hnsw/core.rs src/vector/hnsw/insert.rs src/vector/hnsw/search.rs
git commit -m "Flip insert to &self: AllocState mutex, NodeState publication ordering, concurrent BFS"
```

---

## Task 6: Per-search VectorArenaView; retire vector_slice legacy bridge

**Files:**
- Modify: `src/vector/hnsw/search.rs` (acquire view once per BFS; replace all `vector_slice` calls)
- Modify: `src/vector/hnsw/core.rs` (remove `vector_slice` or make it `#[deprecated]`)
- Test: `tests/hnsw.rs`, benchmarks — verify no regression

**Goal:** Remove the ~13 `vector_slice` call sites in `search.rs` that each implicitly go through `get_direct` (a single-threaded API). Replace with `view.get(idx)` where the view is acquired once at the start of the BFS call.

**Change pattern in `search_layer_unfiltered` and `run_l0_search`:**

```rust
// At the start of the BFS outer function (before the SEARCH_SCRATCH.with closure):
let vec_view = self.vectors.view();  // one Arc clone, no lock held afterwards

// Inside the BFS — replace every:
self.fast_score(query, self.vector_slice(idx))
// with:
self.fast_score(query, vec_view.get(idx))

// Replace every:
self.prefetch_vector(neighbor)  // internally calls vector_slice
// with an updated prefetch that takes a view reference:
prefetch_from_view(&vec_view, neighbor)
```

Also update `quantize_all` and `reorder_rcm` — these are single-threaded operations that can continue to use `get_direct`.

After this task, `vector_slice` can be marked `#[cfg(test)]` or removed if no call sites remain.

- [ ] **Step 1: Acquire view at the top of `search_layer_unfiltered`**

In `src/vector/hnsw/search.rs`, at the start of `search_layer_unfiltered` (before the `SEARCH_SCRATCH.with` closure), add:
```rust
let vec_view = self.vectors.view();
```

- [ ] **Step 2: Update all `self.vector_slice(idx)` calls inside the BFS to `vec_view.get(idx)`**

There are ~13 sites in `search.rs`. Change each one. The view is captured by the closure passed to `SEARCH_SCRATCH.with` via a shared reference.

- [ ] **Step 3: Update `prefetch_vector` to take a view**

```rust
#[inline(always)]
fn prefetch_from_view(view: &crate::vector::hnsw::arena::VectorArenaView, idx: usize) {
    let vector = view.get(idx);
    for offset in [0, 16, 32, 48] {
        if let Some(value) = vector.get(offset) {
            prefetch_read(value);
        }
    }
}
```

Replace all `self.prefetch_vector(neighbor)` calls in the BFS with `prefetch_from_view(&vec_view, neighbor)`.

- [ ] **Step 4: Run benchmarks to confirm no regression**

```bash
cargo build --release --test nytimes_frontier
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_m16_m0_32_stored_128_efc300.bin \
  cargo test --release --test nytimes_frontier nytimes_feature_isolation -- --ignored --nocapture 2>/dev/null \
  | python3 -c "
import sys, json
rows = [json.loads(l) for l in sys.stdin if l.strip().startswith('{')]
for r in rows:
    if r['label'] == 'baseline' and r['ef'] in (64, 128):
        print(f\"ef={r['ef']} p50={r['p50_ms']:.3f}ms recall={r['recall']:.4f}\")
"
```
Expect p50 within 5% of pre-Task-6 baseline.

- [ ] **Step 5: Remove or deprecate `vector_slice`**

If no call sites remain outside tests, remove `vector_slice` from `core.rs`. If test call sites remain, mark `#[cfg(test)]`.

- [ ] **Step 6: Run full suite, fmt, clippy, commit**

```bash
cargo test && cargo fmt --all -- --check && cargo clippy --all-targets 2>&1 | grep "^error"
git add src/vector/hnsw/core.rs src/vector/hnsw/search.rs
git commit -m "Acquire VectorArenaView once per search; remove vector_slice hot-path usage"
```
