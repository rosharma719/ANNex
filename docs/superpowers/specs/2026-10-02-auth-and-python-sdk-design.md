# Auth and Python SDK Design

**Date:** 2026-10-02  
**Status:** Approved for implementation

## Purpose

Close two hard deployment blockers:

1. **Authentication** — the server currently runs open with no auth boundary. Any caller can read or write any collection. Production deployments require at minimum an API key gate.
2. **Python HTTP client** — the existing `annex-py` package is a PyO3 binary for local/embedded use. There is no installable pure-Python client for the HTTP service. ML practitioners who run ANNex as a service have no first-class way to use it from Python.

---

## Part 1: Authentication

### Model

Two global keys, both optional:

| Key | Flag | Grants |
|---|---|---|
| Write key | `--write-key <key>` | All operations (read + write) |
| Read key | `--read-key <key>` | Read-only operations |

**Semantics:**
- If neither flag is set: server runs open (backward compatible for local dev).
- If only `--write-key` is set: the write key also satisfies read-auth (no separate read key required).
- If only `--read-key` is set: write operations require a write key and will always 401 (intentional — a read-only deployment).
- A request that supplies the write key where a read key is expected is accepted.

### Read vs write classification

Write routes (require write key or better):
- `POST /v1/vectors/upsert`
- `POST /v1/vectors/delete`
- `POST /v1/train`
- `POST /v1/fde/index`
- `POST /v1/dense/index`
- `POST /v1/compact`
- `POST /v1/collections` (create)
- Any `POST` to a collection's write endpoints (same set, scoped)

Read routes (read key or write key accepted):
- `GET /healthz`
- `GET /v1/stats`
- `POST /v1/retrieve`
- `POST /v1/query`
- `POST /v1/plan`
- `POST /v1/debug/*`
- `GET /v1/collections`

Classification is done by a lookup table in the middleware, not by HTTP method alone (since most operations are POST).

### Wire protocol

Header: `Authorization: Bearer <key>`

Error responses:
- Missing key when auth is required: `401 Unauthorized`, body `{"error": "authentication required"}`
- Wrong key: `403 Forbidden`, body `{"error": "invalid key"}`

### Implementation

- `auth.rs` module in `crates/annex-multivector/src/`
- `AuthConfig` struct: `read_key: Option<String>`, `write_key: Option<String>`
- Axum `FromRequestParts` extractor or `layer` middleware
- Route classification via a `const` array of `(&str, Permission)` tuples matched against the request path prefix
- `--read-key` and `--write-key` CLI flags added to `args` struct in `main.rs`
- No key rotation, no key file, no management API in this version

### Non-goals

- Per-collection key scoping (future)
- Key rotation without restart (future)
- JWT or OAuth (future)

---

## Part 2: Python HTTP Client

### Package

Location: `python/annex/`  
Package name: `annex` (PyPI)  
Python: ≥ 3.9  
Dependencies: `httpx>=0.27` (sync + async transport in one package)

Distinct from `python/annex-py/` (the PyO3 local binding). The two serve different use cases and are published separately.

### API surface

```python
from annex import AnnexClient

# Sync
client = AnnexClient(
    base_url="http://localhost:7700",
    read_key=None,      # optional
    write_key=None,     # optional
    timeout=30.0,
)

# Writes
client.upsert(documents: list[dict]) -> dict
client.delete(id: str) -> dict
client.build_fde_ann(m: int = 16, ef_construct: int = 200) -> dict
client.build_dense_ann(field: str, m: int = 16, ef_construct: int = 200) -> dict
client.compact() -> dict
client.train(vectors: list[list[float]], iterations: int = 20) -> dict

# Reads
client.retrieve(request: dict) -> dict
client.query(vectors: list[list[float]], top_k: int, ...) -> dict
client.plan(request: dict) -> dict
client.stats() -> dict
client.health() -> dict

# Collections
client.list_collections() -> list[dict]
client.create_collection(name: str, config: dict) -> dict
client.collection(name: str) -> CollectionClient  # returns a scoped client

# Async: identical API, async def, use `async with`
async with AnnexClient(...) as client:
    results = await client.retrieve(...)
```

`CollectionClient` is a thin wrapper that prepends `/v1/collections/{name}` to each request path and delegates to the same transport.

### Auth injection

`AnnexClient` selects the appropriate key per request:
- Write operations: send `write_key` in `Authorization: Bearer`
- Read operations: send `read_key` if set, else `write_key`
- No key configured: send no header

The client uses the same read/write classification as the server (a shared `const` list of write-method path suffixes).

### Error handling

HTTP 4xx/5xx raises `AnnexError(status_code, message)`. The `message` is extracted from the response JSON `error` field if present, otherwise from the HTTP reason phrase.

### Package structure

```
python/annex/
  annex/
    __init__.py        # exports AnnexClient, AnnexError
    client.py          # AnnexClient (sync + async)
    errors.py          # AnnexError
  tests/
    test_client.py     # pytest, uses httpx MockTransport
  pyproject.toml
  README.md
```

### Non-goals

- OpenAPI generation or codegen
- Pydantic models for request/response (keep as plain dicts; typed wrappers are additive later)
- Batch retrieve (additive later)
- Streaming responses

---

## Testing

**Auth:**
- Unit tests in `auth.rs`: open-server pass-through, write-key-only accepts write and read routes, read-key rejects write routes, wrong key → 403, missing key → 401
- Integration test in `main.rs` tests: spin up server with keys, assert 401/403/200 on protected routes

**Python client:**
- `test_client.py` uses `httpx.MockTransport` to intercept requests without a live server
- Tests: key selection (write op sends write key, read op sends read key), error raising, collection scoping, sync and async parity
