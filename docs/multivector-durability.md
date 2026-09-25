# Multivector mutation and recovery contract

`MultiVectorIndex::open` uses **fsync durability**. The server exposes
`--durability fsync|buffered`; library callers can use
`open_with_durability(path, config, Durability::Buffered)` for disposable
benchmarks. This setting is per open, not part of the stored index configuration.

## Atomic mutations

An upsert batch, training operation, or successful delete commits one generation.
Concurrent queries see one whole generation for candidate selection and rescoring.
Duplicate IDs in a batch are applied in order (last wins). Empty batches and deletes
of missing IDs are no-ops, including for ANN validity. Generations never wrap.

A mutation builds a private copy of metadata and postings under the writer lock.
Pre-publication errors leave documents, codebooks, postings, generation, and any
built FDE ANN unchanged. Upserts and deletes stage an ANN overlay alongside the
documents: updated/new IDs enter an exact-scanned delta, and overwritten/deleted
base points become tombstones. Both publish in the same generation, including
when a post-rename error reports `CommitUncertain`. The shared base graph is
immutable; no fallible graph mutation follows manifest publication. Appended bytes
from failed writes may remain unreachable; they are never query candidates.

The exclusive OS lock in `index.lock` permits one open index handle per directory.
Share that handle across threads with `Arc`; opening a second handle/process fails.
The file may remain after exit, but the OS releases the lock even after a crash.
Files must not be modified by processes that bypass the lock.

## Commit protocol

1. Append compressed objects and FDE vectors.
2. Sync both segment files (fsync mode).
3. Serialize generation N+1, including committed segment lengths and per-record
   BLAKE3 digests, into a checksummed manifest envelope.
4. Write `manifest.pending`, then sync it (fsync mode).
5. Atomically rename it to `manifest.json` in the same directory.
6. Sync the containing directory (fsync mode).
7. Publish the staged in-memory state, release the writer lock, and acknowledge.

The rename is the publication point. Manifest bytes and their digest travel in
**one file**, so a crash cannot pair generation N metadata with generation N+1's
checksum. The checksum algorithm is BLAKE3; the old `.sha256` filename was a
misnomer and is no longer written.

If an I/O failure occurs *after* rename, the live handle retains the newly
published generation and returns `IndexError::CommitUncertain`. The caller must
not interpret that error as rollback: inspect/reopen or retry idempotently. A
successful fsync-mode call acknowledges only after all required syncs succeed.
Durability assumes a filesystem/device that honors file and directory sync and
same-directory atomic rename. Filesystems or platforms lacking these operations
return an error; the implementation does not silently downgrade durability.

Buffered mode follows the same atomic publication protocol but omits syncs. It
makes **no power-loss guarantee**; a power loss may lose recently acknowledged
writes or leave an unrecoverable manifest/segment mismatch. There is no background
flush promise and no memory-only mode. Benchmarks must disclose the chosen mode.

## Reopen and compatibility

Open validates manifest version/checksum, codebook shapes, referenced record
checksums and shapes, finite vectors, centroid IDs, and committed segment bounds
before publishing any state. Truncated or corrupt committed data is rejected;
there is no silent fallback to an older generation. Referenced records must be
wholly inside committed boundaries. Open then discards only uncommitted segment
tails; pending manifests are ignored. A missing manifest with nonempty segments
is rejected without truncation.

New manifests use format 2. Format 1 (including manifests predating an explicit
version) remains readable. Its optional, historically named `manifest.sha256`
sidecar is checked if present; legacy records receive structural validation and
are assigned digests in memory. The next successful mutation upgrades the
manifest. Legacy data without digests cannot retroactively provide integrity
proof. A format-2 manifest is self-contained and ignores stale legacy sidecars.
Older binaries cannot open newly committed format-2 manifests; keep a backup if
you need to roll back binaries. This is independent of annex-core's HNSW format.

## Mutable ANN and encoding versions

The HTTP query default (`auto`) uses the built base plus its current overlay, or
an exact FDE scan when no graph is available. Build/rebuild explicitly with
`POST /v1/fde/index`; reopening starts with exact fallback. A rebuild folds live
records into a new base and clears the overlay. If the document generation moves
before publication, the build fails and preserves the current base/overlay.
`fde_ann_base_nodes`, `fde_ann_delta_documents`, and `fde_ann_tombstones` expose
maintenance costs; `fde_ann_nodes` counts live indexed documents. There is no
automatic rebuild threshold. Delta scan and tombstone overfetch costs grow until
the next rebuild.

FDE uses raw inner-product scoring. Normalizing each stored FDE would change the
objective because document FDE norms differ. Cosine-only SQ8 screening is therefore
disabled; FDE storage and base vectors remain FP32. Dot HNSW uses a bounded candidate
reservoir and ordinary best-first termination, so its candidate results are
approximate; pin `candidate_backend: "muvera"` for the exhaustive FDE oracle.

Manifests now independently record `fde_encoding_version`. New indexes use v2,
which fixes double-division when filling an empty bucket from an already-averaged
bucket. It still fills from a bucket average, not a selected nearest token.
Unversioned manifests default to v1 and keep v1 encoding for future writes, so
existing and newly inserted documents remain comparable. Unsupported versions
are rejected. This does not change the manifest envelope version.

The intermediate commit `e9ffe8c` wrote v2 encodings without a version marker.
Indexes ingested with that revision cannot be distinguished from older indexes
by their manifest. Reingest those indexes from original token embeddings into a
new directory; reopening or rebuilding HNSW cannot repair mixed encodings.

## Verification and remaining costs

The ordinary test suite includes deterministic randomized mutation/restart
sequences checked against an independent scalar compressed-MaxSim model,
concurrent whole-batch readers, deletion during queries, version-1 migration,
malformed counts/dimensions, record/manifest corruption, and injected I/O failure.
Subprocess tests exit without destructors at 11 write/commit boundaries, including
partial records and partial manifests, and verify one whole recovered generation.
These are process-interruption tests, **not hardware power-cut testing**.
The scheduled correctness workflow expands the model test to 128 seeds × 512
operations; normal CI runs 8 × 160. Seed/step are included in model assertions.

This correctness-first implementation still clones metadata/postings and rewrites
the full manifest per mutation, holds a write lock during ingest, and scans live
record checksums on open. Superseded records inside committed boundaries require
future compaction. Immutable query generations, WAL/checkpointing, persistent
mmaps, automatic ANN maintenance, and execution pools remain subsequent systems work.
