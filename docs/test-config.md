# Dataset harness configuration

The NYT/H&M recall and performance harnesses are opt-in ignored tests. Ordinary
correctness tests run without datasets. Commands live in [benchmarks.md](benchmarks.md)
and downloads in [data-download.md](data-download.md).

Configuration has one source of truth:

| Concern | Implementation |
| --- | --- |
| Snapshot loading/building and dataset overrides | [shared harness configuration](../crates/annex-core/tests/common/mod.rs) |
| Search budgets and traversal options | [HNSW configuration](../crates/annex-core/src/vector/hnsw/config.rs) |
| Filter seeding | [filtered search](../crates/annex-core/src/vector/hnsw/filter.rs) |
| Progress and query traces | [telemetry](../crates/annex-core/src/utils/telemetry.rs) |
| Durability, recovery and RSS policy | [operations](operations.md) |

[.env.example](../.env.example) contains example settings, not an independent
copy of runtime defaults. Set experiment parameters explicitly and capture the
resulting environment with the report. Many search options are cached on first
use, so compare changed settings in fresh processes.

Dataset controls use `VECTORDB_NYT_*` and `VECTORDB_HNM_*`. Shared `VECTORDB_*`
keys take precedence where they appear first in the harness's lookup list;
unset shared overrides before using dataset-specific settings. Snapshot builds
require explicit `*_ALLOW_BUILD=1`; saving requires `*_SAVE_SNAPSHOT=1`.

Filter seeds are enabled by `VECTORDB_ENABLE_FILTER_SEEDS=1`. The older documented
`VECTORDB_DISABLE_FILTER_SEEDS` variable is not implemented. Use the actual enable
flag and report its value.

Use `VECTORDB_TEST_LOG=quiet|info|debug` for harness progress. JSONL trace paths
and sampling intervals are configured separately. Keep generated output under
ignored `logs/` or benchmark result directories, with unique paths per run.
