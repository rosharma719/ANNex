# ANNexDB: Python bindings for ANNex

[ANNex website](https://annexsearch.dev)

`annexdb.Index` builds, saves, loads and searches an ANNex dense vector index from NumPy
arrays, with native HNSW search for single queries and threaded batches. Python 3.9+ and NumPy
are required. Embeddings are supplied by the caller.

These bindings cover dense vector search. Hybrid retrieval (BM25 + dense + sparse), payload
filters and multivector search live in the Rust API and in the
[`annex-multivector` HTTP server](../../crates/annex-multivector/README.md).

## Install

```sh
python -m pip install ANNexDB
```

To build from source, run this from the workspace root in an activated virtual environment:

```sh
python -m pip install 'maturin>=1.8,<2' numpy
maturin develop --release --manifest-path python/annex-py/Cargo.toml
```

## Quickstart

```python
import annexdb
import numpy as np

# One row per item: your embeddings, float32 and C-contiguous.
vectors = np.array(
    [
        [0.9, 0.1, 0.0, 0.0],
        [0.8, 0.2, 0.1, 0.0],
        [0.0, 0.1, 0.9, 0.2],
        [0.1, 0.0, 0.8, 0.3],
    ],
    dtype=np.float32,
)

index = annexdb.Index.build(vectors, metric="cosine")  # row i gets id i
query = np.array([0.85, 0.15, 0.05, 0.0], dtype=np.float32)
ids, scores = index.search(query, k=3)

index.save("segment.bin")                  # a snapshot Rust can load too
index = annexdb.Index("segment.bin")       # reload it later
ids, scores = index.search_batch(query[None, :], k=3, threads=4)
```

`Index.build(vectors, *, ids=None, metric="cosine", m=16, ef_construct=200, level_cap=16, quantize=False)`
takes a float32 array of shape `[n, dim]`; everything after `vectors` is keyword-only. `ids` is
an optional uint64 array of `n` unique ids; by default row `i` gets id `i`. `metric` is
`"cosine"`, `"dot"` or `"euclidean"` (aliases `"angular"`, `"ip"` and `"l2"`). `m` and
`ef_construct` are the usual HNSW build parameters, spelled as in the Rust API and the HTTP
server. `quantize=True` builds the SQ8 codes used by `sq8_screen` searches; they live in memory
and are not stored in the snapshot, so pass `quantize=True` again when loading. `index.save(path)`
writes a snapshot that `Index(path)` and Rust's `Segment::load_from_path` both load, and
`index.metric()` reports the metric. Snapshots written by the Rust library load the same way.

## Search

Inputs must be C-contiguous float32 NumPy arrays with the index's dimension and finite values
(use `array.astype(np.float32)` for float64 data). Inputs are copied before releasing the GIL;
batch workers are capped to available hardware threads. Single searches return up to `k`
results. Batches have shape `(queries, k)` and pad missing matches with `uint64`'s maximum
value and `NaN`. Empty batches and `k=0` preserve these shapes. Combined batch output is limited
to 1 GiB; oversized shapes raise `OverflowError` before native allocation. Errors propagate to
Python rather than becoming empty result rows.

### Scores

Scores follow the index metric, and which direction is closer depends on it:

| Metric | Score | Closer is |
| --- | --- | --- |
| `cosine` | `1 - cosine similarity` | lower |
| `dot` | dot product | higher |
| `euclidean` | squared Euclidean distance | lower |

Results are returned best first. SQ8 screening requires loading with `quantize=True` and
querying with `sq8_screen=True`. Runtime options `scan_cap` and `patience` keep the core search
meanings; results remain approximate (use `ef` at least the index size for exact results on
small indexes).

## Tests

CI builds and installs the wheel on Linux and macOS, generates a deterministic Rust snapshot,
and checks the Python results against an independent NumPy oracle. The build tests need no
fixture; the snapshot-loading tests use one built by Rust:

```sh
cargo run -p annex-py --example make_test_fixture -- /tmp/annex-fixture.bin
ANNEX_TEST_SNAPSHOT=/tmp/annex-fixture.bin \
  python -m unittest discover -s python/annex-py/tests -v
```
