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

use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

// ── VectorArena ──────────────────────────────────────────────────────────────

/// One fixed-size chunk of f32 vector data.
///
/// `data` is preallocated to `chunk_capacity * dim` f32 values and **never resized**.
/// Addresses inside it are permanently stable for the lifetime of this `Arc<VectorChunk>`.
pub(crate) struct VectorChunk {
    data: Box<[f32]>,
}

impl VectorChunk {
    fn new(chunk_capacity: usize, dim: usize) -> Arc<Self> {
        Arc::new(Self {
            data: vec![0.0f32; chunk_capacity * dim].into_boxed_slice(),
        })
    }

    /// Write `vec` into slot `local_idx`.
    ///
    /// # Safety
    /// Caller must hold the `VectorArena::writer` mutex, ensuring no other thread
    /// writes this slot concurrently. Readers cannot observe this slot until after
    /// the node's `NodeState` is set to `LIVE` (see `arena` module docs).
    fn write(&self, local_idx: usize, dim: usize, vec: &[f32]) {
        let start = local_idx * dim;
        // SAFETY: we have exclusive write access via the writer mutex.
        // The Box allocation is stable — this pointer is valid for the chunk's lifetime.
        // We cast the Box's data pointer to *mut f32 to write the target slice.
        let base = self.data.as_ptr() as *mut f32;
        unsafe {
            let dest = std::slice::from_raw_parts_mut(base.add(start), dim);
            dest.copy_from_slice(vec);
        }
    }

    /// Return the slice for slot `local_idx`.
    ///
    /// # Safety
    /// Caller must ensure `local_idx < chunk_capacity` and the slot has been written.
    #[inline]
    unsafe fn slice(&self, local_idx: usize, dim: usize) -> &[f32] {
        &self.data[local_idx * dim..(local_idx + 1) * dim]
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
pub(crate) struct ChunkedArray<T: Default + Send + Sync + 'static> {
    chunks: RwLock<Arc<[Arc<ArrayChunk<T>>]>>,
    chunk_capacity: usize,
    writer: Mutex<ArrayWriter<T>>,
}

impl<T: Default + Send + Sync + 'static> ChunkedArray<T> {
    pub(crate) fn new(chunk_capacity: usize) -> Self {
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
    pub(crate) fn push_default(&self) -> usize {
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

    /// Acquire a snapshot view. One `Arc` clone; no lock held afterwards.
    pub(crate) fn view(&self) -> ChunkedArrayView<T> {
        ChunkedArrayView {
            chunks: self.chunks.read().clone(),
            chunk_capacity: self.chunk_capacity,
        }
    }

    /// Access slot `idx` for a single operation, without keeping a view alive.
    ///
    /// Acquires a snapshot view briefly, calls `f`, drops the view.
    /// Use this for write-path mutations (e.g. `store` on an `AtomicBool`).
    #[inline]
    pub(crate) fn with<F, R>(&self, idx: usize, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        let view = self.view();
        f(view.get(idx))
    }

    pub(crate) fn len(&self) -> usize {
        self.writer.lock().total
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Snapshot of a [`ChunkedArray`]'s chunk list.
///
/// Holds one `Arc<[Arc<ArrayChunk<T>>]>` — no lock is held after construction.
/// `get(idx)` returns a `&T` with lifetime tied to this view.
pub(crate) struct ChunkedArrayView<T: Send + Sync + 'static> {
    chunks: Arc<[Arc<ArrayChunk<T>>]>,
    chunk_capacity: usize,
}

impl<T: Default + Send + Sync + 'static> ChunkedArrayView<T> {
    /// Return a reference to slot `idx`.
    ///
    /// # Panics
    /// Panics if `idx` was pushed after this view was taken.
    #[inline]
    pub(crate) fn get(&self, idx: usize) -> &T {
        // SAFETY: (a) slot was initialised by push_default() → ArrayChunk::new()
        //             which fills every element with T::default();
        //         (b) this view holds an Arc to the chunk, keeping its Box alive;
        //         (c) T is never moved after initialisation.
        unsafe { self.chunks[idx / self.chunk_capacity].get(idx % self.chunk_capacity) }
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
