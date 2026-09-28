# Python snapshot search

`annex_py.Index` loads an ANNex core `Segment` snapshot and provides native
single-query and batch HNSW search. Python 3.9+ and NumPy are required.
The [multivector HTTP API](../../crates/annex-multivector/README.md) owns hybrid
queries and collections; this binding currently exposes snapshot search only.

Build/install from the workspace root in an activated virtual environment:

```sh
python -m pip install 'maturin>=1.8,<2' numpy
maturin develop --release --manifest-path python/annex-py/Cargo.toml
```

```python
import annex_py
import numpy as np

index = annex_py.Index("segment.bin", quantize=False)
query = np.zeros(index.dim(), dtype=np.float32)  # replace with your embedding
ids, scores = index.search(query, k=10, ef=128)
ids, scores = index.search_batch(query[None, :], k=10, ef=128, threads=4)
```

Inputs must be C-contiguous float32 NumPy arrays with the index's dimension and
finite values. Inputs are copied before releasing the GIL; batch workers are
capped to available hardware threads. Single searches return up to `k` results.
Batches have shape `(queries, k)` and pad missing matches with `uint64`'s maximum
value and `NaN`. Empty batches and `k=0` preserve these shapes. Combined batch
output is limited to 1 GiB; oversized shapes raise `OverflowError` before native
allocation. Errors propagate to Python rather than becoming empty result rows.

Scores retain the snapshot metric: squared Euclidean distance is lower-is-better;
cosine and dot-product similarities are higher-is-better. SQ8 screening requires
loading with `quantize=True` and querying with `sq8_screen=True`. Runtime options
`scan_cap` and `patience` keep the core search meanings; results remain approximate.

CI builds and installs the wheel on Linux and macOS, generates a deterministic
Rust snapshot, and checks the Python results against an independent NumPy oracle:

```sh
cargo run -p annex-py --example make_test_fixture -- /tmp/annex-fixture.bin
ANNEX_TEST_SNAPSHOT=/tmp/annex-fixture.bin \
  python -m unittest discover -s python/annex-py/tests -v
```
