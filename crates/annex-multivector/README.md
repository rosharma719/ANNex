# ANNex multivector

Persistent lexical, dense, sparse and late-interaction retrieval over
caller-supplied text and embeddings. Encoding runs outside the database.

## Quickstart

From the workspace root:

```sh
cargo run --release -p annex-multivector --bin annex-multivector -- \
  --dimension 2 --centroids 2 --path ./data/multivector --listen 127.0.0.1:8080

curl localhost:8080/v1/vectors/upsert -H 'content-type: application/json' \
  -d '{"documents":[{"id":"doc-1","text":"E123 recovery procedure","metadata":{"team":"support"},"representations":{"semantic":{"kind":"dense","vector":[1,0]}}}]}'
curl localhost:8080/v1/retrieve -H 'content-type: application/json' \
  -d '{"prefetch":[{"kind":"bm25","text":"E123 recovery"},{"kind":"dense","field":"semantic","vector":[1,0]}],"filter":{"op":"eq","field":"team","value":"support"},"limit":10}'
```

Named representations and text do not require training. The legacy top-level
`vectors` field uses compressed MaxSim and MUVERA candidates; train first with
`POST /v1/train`, using representative document tokens from the same model.
Reopening the default index requires the same index configuration.

## HTTP contract

| Route | Purpose |
| --- | --- |
| `GET /healthz` | Process liveness and version |
| `GET /v1/stats` | Counts, generation, storage segments and ANN overlay sizes |
| `POST /v1/train` | `vectors`, optional `iterations` |
| `POST /v1/vectors/upsert` | Atomic `documents` batch; complete replacement by `id` |
| `POST /v1/vectors/delete` | Delete one `id`; returns `deleted: false` when absent |
| `POST /v1/retrieve` | Hybrid retrieval and context selection, described below |
| `POST /v1/dense/index` | Build a named dense graph: `field`, optional `m`, `ef_construct` |
| `POST /v1/fde/index` | Build the default MUVERA graph: optional `m`, `ef_construct` |
| `POST /v1/query` | Legacy compressed token retrieval; `vectors`, `top_k`, candidate controls |
| `POST /v1/debug/candidates` | Candidate IDs/scores; `vectors`, `count`, optional backend |
| `POST /v1/debug/score` | Score `query` against exactly one of raw `document` or stored `id` |
| `POST /v1/compact` | Rewrite live vector records and report storage bytes before/after |
| `GET /v1/collections` | List named collections |
| `POST /v1/collections` | Create with `{"name":"manuals","config":{"dimension":2}}` |

Named collections expose the same index routes under
`/v1/collections/{name}/`, for example `/v1/collections/manuals/retrieve`.
Collections have independent configuration, IDs, storage and locks; names allow
1–64 ASCII letters, digits, underscores or hyphens. The unprefixed routes address
the default index. Collections are namespaces, not an authentication boundary:
the service does not authorize callers or enforce tenant ownership.

Unknown request fields are rejected. Invalid parameters return 4xx; storage
failures return 5xx. An uncertain commit requires the recovery procedure in the
[durability contract](../../docs/multivector-durability.md).
Request/work limits are enforced in [main.rs](src/main.rs) and
[retrieval.rs](src/retrieval.rs); the body limit is 256 MiB. Blocking retrieval,
ingest and compaction run outside Tokio's async workers, but do not yet have
separate bounded execution pools, admission control or cancellation.

## Document representations

Every document requires `id`. Optional fields are `metadata`, `text`,
`chunk: {"parent":"source-id","position":0}`, top-level token `vectors`, and
`representations`, an object whose keys are caller-selected field names:

| `kind` | Value | Stored/scored as |
| --- | --- | --- |
| `dense` | `vector: [float, ...]` | Normalized FP32, cosine similarity |
| `multivector` | `vectors: [[float, ...], ...]` | Normalized FP32 tokens, exact MaxSim |
| `sparse` | `vector: {"indices":[u32, ...],"values":[float, ...]}` | Sparse dot product |

Named fields have one persistent kind/dimension within an index, even after all
documents using a field are deleted. Named vectors currently
retain full FP32 storage; compression applies only to top-level token `vectors`.
Sparse weights must be finite and nonnegative. Duplicate feature IDs are summed,
zeros omitted, and arbitrary u32 IDs do not allocate vocabulary-sized arrays.
Upserts replace the whole document, including its text, metadata and fields;
there is no partial-update or source-replacement endpoint.

## Hybrid retrieval and context

`POST /v1/retrieve` takes 1–8 `prefetch` channels. Each has a candidate `limit`:

- `bm25`: query `text`, optional `k1` and `b`. The persisted collection
  analyzer is applied at ingest and query time. `plain` preserves lowercase
  Unicode alphanumeric terms; `english` adds accent folding, English stop-word
  removal, Snowball stemming and a 40-character token limit. Query term
  frequency is preserved.
- `sparse`: named `field` and sparse query `vector`.
- `dense`: named `field`, query `vector`, optional `backend` and `ef_search`.
- `multivector`: query `vectors`, optional named `field`, `backend`, `ef_search`.
  Omitting `field` selects MUVERA FDE candidates from top-level token vectors.

Dense and default multivector channels accept `auto`, `exact` or `hnsw`.
`auto` uses a built graph when possible, otherwise exact scoring. Explicit
`hnsw` requires a graph. Named multivectors accept `auto` or `exact` only.
Filtered retrieval uses exact scoring over eligible documents before channel
limits; BM25 keeps global live-document IDF and length statistics. Graphs are
rebuilt explicitly after restart. Writes preserve built graphs through exact
replacement deltas and tombstones; see the durability contract for maintenance.

`filter` supports scalar `eq`, scalar/array-member `in`, inclusive numeric `range`
with `gte`/`lte`, and boolean `and`, `or`, `not`. `field` is a metadata key or
JSON pointer. Example:

```json
{"op":"and","filters":[
  {"op":"eq","field":"team","value":"support"},
  {"op":"range","field":"/revision/year","gte":2025}
]}
```

The same predicate applies to every channel, reranking and neighbor expansion.
It is caller-supplied filtering, not authorization. The legacy `/v1/query` route
still rejects filters; use `/v1/retrieve` for scoped retrieval.

`fusion` defaults to `{"kind":"rrf","k":10}`. Weighted fusion uses
`{"kind":"weighted","weights":[0.5,0.5]}`, one nonnegative finite weight per
channel; callers must calibrate score scales. A single RRF channel preserves its
native scores. Optional `rerank` supplies token `vectors`, candidate `limit`,
and optional named multivector `field`; all selected candidates must have that
representation. Its opt-in `adaptive` policy reduces the limit to
`min_candidates` when channel agreement reaches `agreement_threshold`; this is
an explicit quality/latency tradeoff, not a guaranteed quality improvement.

`context` supports `per_parent`, exact-text `deduplicate`, adjacent chunk
`neighbors`, and `mmr` with a named dense `diversity_field`. `limit` bounds the
final context, including neighbors. Neighbor hits carry `expanded_from` and
inherit the seed score; they are not independently reranked. Grouping/deduplication
can return fewer hits than requested. MMR normalizes relevance within its pool;
its value is the relevance/diversity weight in [0, 1].

Responses include text, metadata, chunk positions, channel provenance (`sources`),
and a `trace` with generation, eligible-document count, executed channel backends,
candidate/rerank counts, channel agreement and timing. All stages read the same
immutable generation.

## Legacy candidate selection

For `POST /v1/query`:

- Omitted backend or `auto`: HNSW when available, exact FDE otherwise.
- `hnsw`: requires a built graph; accepts `ef_search` and `rerank_candidates`.
- `muvera`: exact FDE candidate scan. `probes` selects the centroid path;
  `rerank_candidates` enables centroid pruning. These two options are exclusive.
- HNSW rejects `probes`; MUVERA rejects `ef_search`; explicit `auto` rejects
  pruning/probe knobs. Legacy omitted-backend requests retain their
  MUVERA/centroid behavior when those knobs are supplied.

Debug candidates default to exact FDE. `explain: true` reports requested and
executed backends separately. FDE scores are raw inner products; centroid-only
results omit them. Current MUVERA empty-bucket fill uses a bucket average.

See [benchmarking](benchmark/README.md) and the
[remaining launch work](../../docs/launch-verification-plan.md).
