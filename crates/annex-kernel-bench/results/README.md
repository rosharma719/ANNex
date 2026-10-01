# Kernel benchmark results (2026-09-30)

Raw criterion output from a 4-vCPU Intel Xeon @ 2.1 GHz cloud VM (AVX2, FMA,
AVX-512 F/BW/VNNI), `RUSTFLAGS="-C target-cpu=native"`, commit 40d90c5.
Single runs on a shared VM: treat differences under ~15% as noise. Part of
the SQ8 section overlapped a clippy build and needs a clean re-run.

- `baseline-before-*.txt`: `annex-multivector` kernels bench on the parent commit.
- `distance-*.txt`: `cargo bench -p annex-kernel-bench --features blas --bench distance`
  (time is per 256 scores).
- `maxsim-*.txt`: `cargo bench -p annex-kernel-bench --features blas --bench maxsim`.

Measured zmm FMA peak on this VM: 161.7 GFLOP/s; prepared MaxSim reaches ~94%.

The optional `../cpp/` harness times C++ competitors (hnswlib, FAISS, Eigen,
OpenBLAS) for comparison only; ANNex itself is all Rust.
