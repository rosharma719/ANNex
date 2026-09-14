# Contributing to ANNex

Thanks for your interest in contributing.

## Development setup

- Install Rust stable (1.85+ — see `rust-version` in `Cargo.toml`).
- Clone the repo and build: `cargo build`.
- Run the fast local test suite: `cargo test`.

## Before opening a PR

Please make sure the following pass locally:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets
cargo test
cargo doc --no-deps
```

The GitHub Actions workflow runs the same checks on every push and PR.

## Datasets and benchmarks

The dataset-driven tests and benchmarks under `tests/nytimes.rs`,
`tests/hnm.rs`, and `src/bin/nyt_search_bench.rs` require downloaded data and
are marked `#[ignore]` or gated behind env vars. See
[`docs/data-download.md`](docs/data-download.md) and
[`docs/benchmarks.md`](docs/benchmarks.md).

## Style

- Rustfmt-formatted; no unformatted PRs.
- Prefer small, focused PRs. Large behaviour changes benefit from an issue
  first to align on approach.
- Public API changes require a note in `CHANGELOG.md` under `## [Unreleased]`.
- Anything under `vector::`, `payload_storage::`, and `analysis::` is treated
  as an implementation detail (marked `#[doc(hidden)]`) and may change without
  a major bump. New public surface goes through the crate root (`src/lib.rs`).

## Reporting bugs

Open a GitHub issue with a minimal reproducer, expected vs. actual behaviour,
and the relevant `VECTORDB_*` env vars if applicable.

## Security

Please do not file security issues in the public tracker. See
[SECURITY.md](SECURITY.md).

## Licensing

By contributing, you agree that your contributions will be dual-licensed under
the MIT and Apache-2.0 licenses, as described in the project [README](README.md).
