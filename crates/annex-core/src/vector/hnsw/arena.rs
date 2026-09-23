//! Stable-address storage primitives for concurrent HNSW operation.
//!
//! # Memory model
//!
//! The fundamental problem with a plain `Vec<f32>` for vector storage is
//! reallocation: when the Vec grows, it may move all existing data to a new
//! allocation, invalidating any `&[f32]` slices held by concurrent readers.
//!
//! Both types here solve this with the same strategy:
//! - Backing storage is split into fixed-size **chunks**.
//! - Each chunk is a `Box<[T]>` / `Box<[f32]>` allocated once and **never resized**.
//!   Addresses inside a chunk are permanently stable.
//! - Growing the arena appends a new chunk rather than moving existing data.
//! - Readers acquire a **snapshot view** (`VectorArenaView` / `ChunkedArrayView`)
//!   that holds an `Arc<[Arc<Chunk>]>`. The outer slice is immutable; a new one
//!   is published atomically when a chunk is added. Readers who already hold a
//!   snapshot are unaffected.
//!
//! # Hot-path usage
//!
//! ```text
//! // Once per search — one Arc clone, short read-lock acquisition:
//! let view = index.vectors.view();
//!
//! // ~2825 times inside BFS — zero locks, zero refcount operations:
//! let vec: &[f32] = view.get(idx);
//! ```

use std::cell::UnsafeCell;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

// ── VectorArena ──────────────────────────────────────────────────────────────

/// One fixed-size chunk of f32 vector data.
///
/// Each f32 element is wrapped in its own `UnsafeCell` so Miri can verify that
/// writes to slot N do not alias reads from slot M (different `UnsafeCell` instances).
/// This is the correct granularity for Stacked Borrows / Tree Borrows aliasing models.
///
/// `data` is preallocated to `chunk_capacity * dim` cells and **never resized**.
/// Addresses of individual cells are permanently stable.
pub(crate) struct VectorChunk {
    data: Box<[UnsafeCell<f32>]>,
}

// SAFETY: VectorChunk is shared across threads. Each cell is written exactly once
// (by the thread holding VectorArena::writer mutex) and read-only afterwards.
// UnsafeCell disables Rust's read-only aliasing guarantees; callers are responsible
// for ensuring write-before-read ordering via the NodeState Release/Acquire protocol.
unsafe impl Sync for VectorChunk {}
unsafe impl Send for VectorChunk {}

impl VectorChunk {
    fn new(chunk_capacity: usize, dim: usize) -> Arc<Self> {
        let data: Vec<UnsafeCell<f32>> = (0..chunk_capacity * dim)
            .map(|_| UnsafeCell::new(0.0f32))
            .collect();
        Arc::new(Self {
            data: data.into_boxed_slice(),
        })
    }

    /// Write `vec` into slot `local_idx`.
    ///
    /// # Safety
    /// Caller must hold the `VectorArena::writer` mutex (no concurrent writes).
    /// Readers cannot observe this slot until `NodeState` is published as LIVE.
    fn write(&self, local_idx: usize, dim: usize, vec: &[f32]) {
        let start = local_idx * dim;
        for i in 0..dim {
            // SAFETY: writer mutex ensures exclusive write access to this slot.
            // Each UnsafeCell<f32> is an independent aliasing unit.
            unsafe { *self.data[start + i].get() = vec[i] };
        }
    }

    /// Return a shared slice for slot `local_idx`.
    ///
    /// # Safety
    /// Caller must ensure the slot has been written and published (NodeState = LIVE
    /// with Release/Acquire fence). Each UnsafeCell<f32> may only be read after
    /// its corresponding write has been published.
    #[inline]
    unsafe fn slice(&self, local_idx: usize, dim: usize) -> &[f32] {
        let start = local_idx * dim;
        // SAFETY: UnsafeCell<f32> has the same layout as f32. We reinterpret a
        // slice of UnsafeCell<f32> as &[f32]. The slice is valid until the chunk
        // is dropped (kept alive by Arc). No writes to these cells happen after
        // publication (write-once protocol).
        let ptr = self.data[start].get() as *const f32;
        std::slice::from_raw_parts(ptr, dim)
    }
}

struct VectorArenaWriter {
    total: usize,
    chunks: Vec<Arc<VectorChunk>>,
}

/// Stable-address chunk store for f32 vectors.
///
/// Chunks are pre-allocated to `chunk_capacity` vectors × `dim` f32s and never
/// resized. Growing appends a new chunk.  Concurrent readers acquire a
/// [`VectorArenaView`] snapshot (one `Arc` clone per search) and access vectors
/// at zero per-vector cost.
pub struct VectorArena {
    /// Atomic snapshot of the current chunk list.
    /// Readers take a brief read-lock to clone the `Arc<[Arc<VectorChunk>]>`.
    /// Writers take a brief write-lock only when adding a new chunk.
    chunks: RwLock<Arc<[Arc<VectorChunk>]>>,
    dim: usize,
    chunk_capacity: usize,
    writer: Mutex<VectorArenaWriter>,
}

impl VectorArena {
    pub fn new(dim: usize, chunk_capacity: usize) -> Self {
        assert!(dim > 0 && chunk_capacity > 0);
        Self {
            chunks: RwLock::new(Arc::from(Vec::<Arc<VectorChunk>>::new())),
            dim,
            chunk_capacity,
            writer: Mutex::new(VectorArenaWriter {
                total: 0,
                chunks: Vec::new(),
            }),
        }
    }

    /// Append one vector; return its stable node index.
    ///
    /// Holds the writer `Mutex` for the duration of the write.
    /// Holds the chunks `RwLock` write-lock only when adding a new chunk (~every
    /// `chunk_capacity` pushes).
    pub fn push(&self, vec: &[f32]) -> usize {
        debug_assert_eq!(vec.len(), self.dim, "vector dimension mismatch");
        let mut w = self.writer.lock();
        let idx = w.total;
        let local_idx = idx % self.chunk_capacity;

        if local_idx == 0 {
            // Need a new chunk.
            let chunk = VectorChunk::new(self.chunk_capacity, self.dim);
            chunk.write(0, self.dim, vec);
            w.chunks.push(chunk);
            // Publish the updated chunk list atomically.
            // Any reader who already holds a snapshot of the OLD list is unaffected;
            // the old Arc<[Arc<VectorChunk>]> remains alive via their view.
            let new_snapshot: Arc<[Arc<VectorChunk>]> = w.chunks.clone().into();
            *self.chunks.write() = new_snapshot;
        } else {
            w.chunks[idx / self.chunk_capacity].write(local_idx, self.dim, vec);
        }
        w.total += 1;
        idx
    }

    /// Acquire a snapshot view.
    ///
    /// Holds the chunks `RwLock` read-lock only for the duration of the `Arc` clone.
    /// The returned view holds **no lock** and allows zero-cost `get(idx)` access.
    #[inline]
    pub fn view(&self) -> VectorArenaView {
        VectorArenaView {
            chunks: self.chunks.read().clone(),
            dim: self.dim,
            chunk_capacity: self.chunk_capacity,
        }
    }

    /// Total number of vectors pushed.
    pub fn len(&self) -> usize {
        self.writer.lock().total
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Direct indexed access for single-threaded contexts (snapshot serialisation,
    /// reorder, quantisation).  Do **not** use in the BFS hot path — use `view()`.
    ///
    /// # Panics
    /// Panics if `idx >= self.len()`.
    pub(crate) fn get_direct(&self, idx: usize) -> &[f32] {
        let chunks = self.chunks.read();
        let chunk_idx = idx / self.chunk_capacity;
        let local_idx = idx % self.chunk_capacity;
        let chunk = chunks[chunk_idx].clone();
        // SAFETY: slot was written in `push()` before `total` was incremented.
        // The Arc<VectorChunk> we cloned keeps the backing Box alive. The read guard
        // is dropped here, but `chunk` (the Arc) keeps the allocation alive for the
        // duration of the `&[f32]` borrow — which is bounded by `&self` lifetime.
        //
        // Formally: we extend the lifetime from the Arc's local scope to `&self`.
        // This is sound because:
        //   (a) VectorChunk data is never mutated after push(),
        //   (b) The Arc is one of many held by `self.chunks`, so the allocation
        //       lives as long as `self` does.
        drop(chunks);
        unsafe { std::slice::from_raw_parts(chunk.slice(local_idx, self.dim).as_ptr(), self.dim) }
    }

    /// Iterate all pushed vectors in index order.  Single-threaded only.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &[f32])> {
        let n = self.len();
        (0..n).map(move |idx| (idx, self.get_direct(idx)))
    }
}

/// Snapshot of [`VectorArena`]'s chunk list at a point in time.
///
/// Holds one `Arc<[Arc<VectorChunk>]>` — no lock is held after construction.
/// `get(idx)` is a pure indexed slice with no synchronisation.
///
/// Vectors pushed **after** this view was acquired are **not** visible through it.
/// That is intentional: the BFS loop only traverses nodes that were `LIVE` when
/// the search began.
pub struct VectorArenaView {
    chunks: Arc<[Arc<VectorChunk>]>,
    dim: usize,
    chunk_capacity: usize,
}

impl VectorArenaView {
    /// Return the vector slice at `idx`.
    ///
    /// # Panics
    /// Panics if `idx` was pushed after this view was taken (chunk not in snapshot).
    #[inline]
    pub fn get(&self, idx: usize) -> &[f32] {
        let chunk_idx = idx / self.chunk_capacity;
        let local_idx = idx % self.chunk_capacity;
        // SAFETY: (a) slot was written in push() before total was incremented;
        // (b) this view holds an Arc to the chunk, keeping its Box<[f32]> alive.
        unsafe { self.chunks[chunk_idx].slice(local_idx, self.dim) }
    }
}

// ── ChunkedArray<T> ──────────────────────────────────────────────────────────

/// One fixed-size chunk of elements of type `T`.
///
/// Elements are default-initialised at chunk creation and never moved.
struct ArrayChunk<T> {
    data: Box<[T]>,
}

impl<T: Default> ArrayChunk<T> {
    /// Allocate `capacity` elements, each initialised with `T::default()`.
    fn new(capacity: usize) -> Arc<Self> {
        let data: Vec<T> = (0..capacity).map(|_| T::default()).collect();
        Arc::new(Self {
            data: data.into_boxed_slice(),
        })
    }

    /// Return a reference to element at `local_idx`.
    ///
    /// # Safety
    /// `local_idx` must be < chunk capacity.
    #[inline]
    unsafe fn get(&self, local_idx: usize) -> &T {
        self.data.get_unchecked(local_idx)
    }
}

struct ArrayWriter<T> {
    total: usize,
    chunks: Vec<Arc<ArrayChunk<T>>>,
}

/// Stable-slot append-only array of `T`.
///
/// Identical growth strategy to [`VectorArena`]: chunks are `Box<[T]>` slabs
/// allocated once and never moved. The element at slot `idx` has a permanent
/// address for the lifetime of the `ChunkedArray`.
///
/// # Access pattern
///
/// **Do not** call `get` directly on `ChunkedArray` — there is no safe `get`
/// method on the container itself (returning `&T` would require keeping a chunk
/// Arc alive, which cannot be expressed in the return type without `Arc` cloning).
///
/// Instead:
/// - For multi-access hot paths: acquire a `ChunkedArrayView` once with `view()`.
/// - For single-access mutation: use `with(idx, |slot| ...)`.
pub struct ChunkedArray<T: Default + Send + Sync + 'static> {
    chunks: RwLock<Arc<[Arc<ArrayChunk<T>>]>>,
    chunk_capacity: usize,
    writer: Mutex<ArrayWriter<T>>,
}

impl<T: Default + Send + Sync + 'static> ChunkedArray<T> {
    pub fn new(chunk_capacity: usize) -> Self {
        assert!(chunk_capacity > 0);
        Self {
            chunks: RwLock::new(Arc::from(Vec::<Arc<ArrayChunk<T>>>::new())),
            chunk_capacity,
            writer: Mutex::new(ArrayWriter {
                total: 0,
                chunks: Vec::new(),
            }),
        }
    }

    /// Allocate one slot initialised to `T::default()`; return its stable index.
    pub fn push_default(&self) -> usize {
        let mut w = self.writer.lock();
        let idx = w.total;
        let local_idx = idx % self.chunk_capacity;

        if local_idx == 0 {
            let chunk = ArrayChunk::<T>::new(self.chunk_capacity);
            w.chunks.push(chunk);
            let new_snapshot: Arc<[Arc<ArrayChunk<T>>]> = w.chunks.clone().into();
            *self.chunks.write() = new_snapshot;
        }
        w.total += 1;
        idx
    }

    /// Acquire a snapshot view.
    ///
    /// Captures both the chunk snapshot and the logical length under the writer
    /// lock so `view.len()` and `view.get(idx)` are always consistent — any
    /// `idx < view.len()` is guaranteed to reside in a published chunk.
    pub fn view(&self) -> ChunkedArrayView<T> {
        let w = self.writer.lock();
        let chunks = self.chunks.read().clone();
        let len = w.total;
        drop(w);
        ChunkedArrayView {
            chunks,
            chunk_capacity: self.chunk_capacity,
            len,
        }
    }

    /// Access slot `idx` for a single operation, without keeping a view alive.
    ///
    /// Acquires a snapshot view briefly, calls `f`, drops the view.
    /// Use this for write-path mutations (e.g. `store` on an `AtomicBool`).
    #[inline]
    pub fn with<F, R>(&self, idx: usize, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        let view = self.view();
        f(view.get(idx))
    }

    pub fn len(&self) -> usize {
        self.writer.lock().total
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Snapshot of a [`ChunkedArray`]'s chunk list.
///
/// Holds one `Arc<[Arc<ArrayChunk<T>>]>` — no lock is held after construction.
/// `get(idx)` returns a `&T` with lifetime tied to this view.
pub struct ChunkedArrayView<T: Send + Sync + 'static> {
    chunks: Arc<[Arc<ArrayChunk<T>>]>,
    chunk_capacity: usize,
    len: usize,
}

impl<T: Default + Send + Sync + 'static> ChunkedArrayView<T> {
    /// Return a reference to slot `idx`.
    ///
    /// # Panics
    /// Panics if `idx` was pushed after this view was taken.
    #[inline]
    pub fn get(&self, idx: usize) -> &T {
        // SAFETY: (a) slot was initialised by push_default() → ArrayChunk::new()
        //             which fills every element with T::default();
        //         (b) this view holds an Arc to the chunk, keeping its Box alive;
        //         (c) T is never moved after initialisation.
        unsafe { self.chunks[idx / self.chunk_capacity].get(idx % self.chunk_capacity) }
    }

    /// Logical length of the array captured at view time.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return a reference to slot `idx` if `idx < self.len()`.
    #[inline]
    pub fn get_opt(&self, idx: usize) -> Option<&T> {
        if idx < self.len {
            Some(self.get(idx))
        } else {
            None
        }
    }
}

// ── Send/Sync impls ──────────────────────────────────────────────────────────
//
// VectorArena and ChunkedArray<T> are Send+Sync provided T: Send+Sync.
// The interior mutability (RwLock, Mutex) already makes them Sync.
// VectorArenaView and ChunkedArrayView hold only Arc<[Arc<Chunk>]> which is
// Send+Sync when the chunk data is.

unsafe impl Send for VectorArena {}
unsafe impl Sync for VectorArena {}
unsafe impl Send for VectorArenaView {}
unsafe impl Sync for VectorArenaView {}
