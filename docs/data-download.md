# Dataset Downloads

All benchmark datasets are managed through `scripts/fetch_dataset.py`.
The registry at `data/registry.json` is the committed source of truth for
URLs, expected dimensions, and verified checksums.

## Quick start

```bash
# Download one or more datasets
python3 scripts/fetch_dataset.py nytimes-256-angular
python3 scripts/fetch_dataset.py sift-128-euclidean glove-100-angular
python3 scripts/fetch_dataset.py --all   # all registered datasets

# List what's registered and what you have locally
python3 scripts/fetch_dataset.py

# Verify integrity of already-downloaded data (no network)
python3 scripts/fetch_dataset.py --verify-only
```

Requires: `numpy`, `h5py`, `curl` (all present in `/tmp/annbench_env`).

## Registered datasets

| ID | Name | Dims | Metric | Base vectors |
|----|------|-----:|--------|-------------:|
| `nytimes-256-angular` | NYT-256-Angular | 256 | cosine | 290,000 |
| `sift-128-euclidean` | SIFT-1M | 128 | euclidean | 1,000,000 |
| `glove-100-angular` | GloVe-100-Angular | 100 | cosine | 1,183,514 |

All sourced from [ann-benchmarks.com](https://ann-benchmarks.com).

## Adding a new dataset

1. Add an entry to `data/registry.json` with the URL and `null` checksums.
2. Run `python3 scripts/fetch_dataset.py <id> --record` to download,
   convert, and populate the checksums.
3. Commit `data/registry.json` with the new checksums.

## On-disk format (all datasets)

```
data/<id>/
  base.npy            float32 array  (n_base, dims)
  queries.npy         float32 array  (n_queries, dims)
  ground_truth.json   list[list[int]], 100 true neighbors per query
  dataset.json        local record: fetched_at, shapes, checksums (git-ignored)
  [index cache files] competitor indexes (git-ignored)
```

## H&M (2048-D cosine, filtered)

H&M uses a different source and format and is not yet in the registry.
Until it is migrated, use the manual instructions below.

```bash
mkdir -p data/hnm
curl -L https://storage.googleapis.com/ann-filtered-benchmark/datasets/hnm.tgz \
  | tar -xz -C data/hnm --strip-components=1
# Produces: vectors.npy, payloads.jsonl, tests.jsonl, filters.json
```
