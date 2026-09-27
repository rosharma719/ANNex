# Contributing

Use [rust-toolchain.toml](rust-toolchain.toml) and run from the workspace root:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test --workspace
cargo doc --workspace --no-deps
cargo build -p annex-multivector --bin annex-multivector
ANNEX_TEST_BINARY=target/debug/annex-multivector \
  python3 -m unittest discover -s crates/annex-multivector/benchmark -p 'test_*.py'
```

Python tests require NumPy (no model or dataset downloads).

[CI](.github/workflows/ci.yml) validates Linux and macOS plus release benchmark
compilation. The [nightly workflow](.github/workflows/correctness-nightly.yml)
increases state-machine oracle seeds/steps.

Keep correctness tests deterministic and assert externally meaningful behavior.
Retain crash-boundary, scalar-oracle, mutation and concurrency coverage. Put
performance measurements in benchmark harnesses; avoid duplicate timing-only
tests or performance assertions on shared CI hardware.

Dataset-dependent harnesses are opt-in; see [benchmark commands](docs/benchmarks.md).
Generated results, caches, profiles and local agent settings are ignored. Publish
benchmark evidence according to [BENCHMARK_POLICY.md](BENCHMARK_POLICY.md).

Document each contract once: public API behavior in Rustdoc/the component README,
persistence in [the durability contract](docs/multivector-durability.md), benchmark
commands in the benchmark guide. Link those sources from summaries. Comments
should explain invariants or decisions, not narrate the code.

Public behavior changes require an entry under `Unreleased` in [CHANGELOG.md](CHANGELOG.md).
Modules marked `doc(hidden)` are implementation details. Submit a reproducer and
relevant runtime settings with bug reports. Use [SECURITY.md](SECURITY.md) for
private vulnerability reports.

Contributions are dual-licensed MIT/Apache-2.0 and subject to the
[Code of Conduct](CODE_OF_CONDUCT.md).
