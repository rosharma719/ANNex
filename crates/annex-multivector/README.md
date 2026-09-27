# ANNex multivector

Persistent late-interaction retrieval over caller-supplied token embeddings.
MUVERA fixed-dimensional encodings select candidates; packed residual vectors
support compressed MaxSim reranking. Text encoding runs outside the database.

## Quickstart

From the workspace root:

```sh
cargo run --release -p annex-multivector --bin annex-multivector -- \
  --dimension 2 --centroids 2 --path ./data/multivector --listen 127.0.0.1:8080

curl localhost:8080/v1/train -H 'content-type: application/json' \
  -d '{"vectors":[[1,0],[0,1]],"iterations":20}'
curl localhost:8080/v1/vectors/upsert -H 'content-type: application/json' \
  -d '{"documents":[{"id":"doc-1","vectors":[[1,0],[0,1]],"metadata":{"source":"manual"}}]}'
curl localhost:8080/v1/query -H 'content-type: application/json' \
  -d '{"vectors":[[1,0]],"top_k":10,"candidates":80,"explain":true}'
```

Train once before ingestion using representative document tokens from the same
model. Reopening requires the same index configuration.

## HTTP contract

| Route | Purpose |
| --- | --- |
| `GET /healthz` | Process liveness and version |
| `GET /v1/stats` | Document/token counts, generation and ANN overlay sizes |
| `POST /v1/train` | `vectors`, optional `iterations` |
| `POST /v1/vectors/upsert` | Atomic `documents` batch; each has `id`, `vectors`, optional `metadata` |
| `POST /v1/vectors/delete` | Delete one `id`; returns `deleted: false` when already absent |
| `POST /v1/fde/index` | Build/rebuild ANN with optional `m`, `ef_construct` |
| `POST /v1/query` | Token `vectors`, `top_k`, candidate controls below |
| `POST /v1/debug/candidates` | Candidate IDs/scores; `vectors`, `count`, optional backend |
| `POST /v1/debug/score` | Score `query` against exactly one of raw `document` or stored `id` |

Unknown fields are rejected, including unsupported filters and tenant constraints.
Metadata is stored and returned; it does not currently constrain retrieval.
Invalid parameters return 4xx. Storage failures return 5xx; uncertain commits
require the recovery procedure in the durability contract.

Request bounds are enforced in [main.rs](src/main.rs): token/batch sizes, training
work parameters, query budgets and ID lengths. ANN construction bounds also apply
to Rust callers. The HTTP body limit is 256 MiB. These bounds do not provide
admission control or guarantee latency under overload.

## Candidate selection

- Omitted backend or `auto`: HNSW when available, exact FDE otherwise.
- `hnsw`: require a built graph; accepts `ef_search`.
- `muvera`: exact FDE candidate scan. Optional `probes` selects the centroid
  path; optional `rerank_candidates` enables centroid pruning. These two
  options are mutually exclusive.
- Explicit HNSW also accepts `rerank_candidates`, but rejects `probes`.
  Exact MUVERA rejects `ef_search`; explicit auto rejects pruning/probe knobs.
- Legacy omitted-backend requests with `probes` or `rerank_candidates` retain
  their MUVERA/centroid behavior. Debug candidates default to exact FDE.

Build ANN using `POST /v1/fde/index` with `{"m":16,"ef_construct":256}`.
Writes preserve the graph through exact deltas and tombstones. Rebuild folds the
overlay into a new base. Reopening currently requires rebuilding ANN.

`explain: true` reports the requested and executed backend separately. FDE
candidate scoring uses raw inner products. Centroid-only results omit FDE scores.
Current MUVERA empty-bucket fill uses a bucket average, not nearest-token fill.

See [durability and compatibility](../../docs/multivector-durability.md),
[benchmarking](benchmark/README.md) and the [remaining work](../../docs/launch-verification-plan.md).
