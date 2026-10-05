# Bounded multivector serving

The multivector HTTP server admits work before parsing request bodies and runs
synchronous engine operations outside Tokio in three dedicated Rayon pools.
The default index and all named collections share these process-wide limits.

| Class | Routes | Default threads | Default admitted requests |
| --- | --- | --- | --- |
| Query | query, retrieve, plan, stats, debug scoring/candidates, list collections | available CPUs minus 2, minimum 1, maximum 256 | 16 |
| Ingest | upsert, delete, train, create collection | 1 | 2 |
| Maintenance | compact, dense/index, fde/index | 1 | 1 |

Configure `--query-threads`, `--ingest-threads`, `--maintenance-threads` and
`--query-concurrency`, `--ingest-concurrency`, `--maintenance-concurrency`.
Thread counts accept 1–256; admission capacities accept 1–65535. Capacities
include requests parsing their bodies, queued jobs and running jobs. Increasing
capacity does not increase worker threads. Choose limits for the deployment's
CPU and memory budget; one admitted request can still carry a 256 MiB body.

A saturated class immediately returns HTTP 503 with `Retry-After: 1` and
`{"error":"server busy","class":"query"}` (or ingest/maintenance).
Clients should use bounded retries with backoff. Authentication runs before
admission. `/healthz` and the unprefixed `GET /v1/runtime` bypass the worker
limits, so operators can inspect saturation. Runtime statistics are global,
including when accessed through a named collection.

Nested Rayon operations use the selected pool. HTTP graph builds link on the
maintenance worker instead of spawning hardware-sized native thread groups.
This trades graph-build throughput for predictable worker ownership. Existing
Rust library build methods retain parallel linking; explicit `*_with_threads`
methods are available for embedding applications.

Admitted responses include `x-annex-queue-ms`, `x-annex-work-ms` and
`x-annex-request-ms`. Queue/work timings sum the request's jobs, including a
named collection lookup. Request time includes parsing, routing and response
construction; it excludes network delivery. `GET /v1/runtime` exposes configured
threads/capacity, in-flight requests, admitted/rejected counts, queued/running
jobs, completed/panicked jobs, skipped cancelled jobs and cumulative queue/work
time. Counters are approximate concurrent snapshots and reset on restart.

A dropped request skips its queued job when a worker dequeues it. An already
running operation finishes and keeps its admission permit until it stops;
abandoning a request cannot free capacity while its CPU work continues.
Running kernels do not yet support cooperative cancellation or deadlines.
Pools separate execution capacity, but shared engine locks and storage I/O can
still cause interference. These controls do not establish a tail-latency SLO;
load tests, latency histograms and running-work deadlines remain follow-up work.
