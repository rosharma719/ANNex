#!/usr/bin/env bash
# Builds and runs the C++ competitor kernels (hnswlib, FAISS, Eigen, OpenBLAS).
# Requirements: g++, libeigen3-dev, libopenblas-dev, `pip install faiss-cpu`.
# HNSWLIB_INCLUDE may point at an hnswlib checkout; otherwise v0.8.0 headers
# are fetched into target/.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(git rev-parse --show-toplevel)"
OUT="$ROOT/target/kernel-bench-cpp"
mkdir -p "$OUT"

HNSWLIB_INCLUDE="${HNSWLIB_INCLUDE:-$OUT/hnswlib-0.8.0}"
if [ ! -f "$HNSWLIB_INCLUDE/hnswlib/hnswlib.h" ]; then
  mkdir -p "$HNSWLIB_INCLUDE/hnswlib"
  for f in hnswlib.h space_ip.h space_l2.h hnswalg.h bruteforce.h visited_list_pool.h stop_condition.h; do
    curl -sSfL "https://raw.githubusercontent.com/nmslib/hnswlib/v0.8.0/hnswlib/$f" \
      -o "$HNSWLIB_INCLUDE/hnswlib/$f"
  done
fi

FAISS_DIR="${FAISS_DIR:-$(python3 -c 'import faiss, os; print(os.path.dirname(faiss.__file__))')}"

g++ -O3 -march=native -std=c++17 -DNDEBUG competitors.cpp -o "$OUT/competitors" \
  -I"$HNSWLIB_INCLUDE" -I/usr/include/eigen3 -I/usr/include/x86_64-linux-gnu/openblas-pthread \
  "$FAISS_DIR/libfaiss.so" -Wl,-rpath,"$FAISS_DIR" -lopenblas -lpthread

python3 -c 'import faiss; print("# faiss", faiss.__version__, "simd:", faiss.get_compile_options())' || true
"$OUT/competitors"
