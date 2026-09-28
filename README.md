# ANNex

[![CI](https://github.com/rosharma719/ANNex/actions/workflows/ci.yml/badge.svg)](https://github.com/rosharma719/ANNex/actions/workflows/ci.yml)

ANNex is a Rust workspace for vector and late-interaction retrieval.

| Component | Purpose | Documentation |
| --- | --- | --- |
| `annex-core` | HNSW, payload filters, sparse retrieval primitives, snapshots and WAL | [Rust API](crates/annex-core/src/lib.rs), [operations](docs/operations.md) |
| `annex-multivector` | Persistent hybrid retrieval, named vector fields, filtered context selection and an HTTP API | [API and quickstart](crates/annex-multivector/README.md), [durability](docs/multivector-durability.md) |
| `annex-server` | Dense-vector HTTP benchmark adapter | [source](crates/annex-server/src/bin/dense_server.rs) |
| `annex-py` | Native Python/NumPy snapshot search and threaded batches | [Python installation and API](python/annex-py/README.md) |

Embeddings are supplied by the caller. Deployment limits and remaining launch
work are tracked in the [implementation plan](docs/launch-verification-plan.md).

## Build and use

Use the toolchain in [rust-toolchain.toml](rust-toolchain.toml):

```sh
cargo build --workspace
cargo test --workspace
cargo doc --workspace --no-deps --open
```

To embed the core library from a local checkout:

```toml
[dependencies]
annex = { package = "annex-core", path = "/path/to/ANNex/crates/annex-core" }
```

The tested Rust quickstart lives in the [crate documentation](crates/annex-core/src/lib.rs).
For the multivector server, follow its [quickstart](crates/annex-multivector/README.md).

## Development and evaluation

- [Contribution and test workflow](CONTRIBUTING.md)
- [Benchmark commands](docs/benchmarks.md) and [reporting policy](BENCHMARK_POLICY.md)
- [Dataset setup](docs/data-download.md), [test configuration](docs/test-config.md)
- [Snapshot construction](docs/index-construction.md), [recall analysis](docs/recall_frontier_pipeline.md)
- [Historical benchmark corrections](crates/annex-multivector/benchmark/RESULTS.md)

Runtime tuning uses the existing `VECTORDB_*` prefix; see [.env.example](.env.example).
Core diagnostics use the `log` crate. Applications supply their logging backend.
Benchmark artifacts and local logs are generated outside versioned source.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md), [Code of Conduct](CODE_OF_CONDUCT.md),
and [security reporting](SECURITY.md).

Licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
Contributions are offered under the same dual license.
