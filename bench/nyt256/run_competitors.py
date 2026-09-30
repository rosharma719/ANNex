#!/usr/bin/env python3
"""
Run hnswlib, usearch, FAISS HNSW, and FAISS IVF-PQ sweeps over the NYT-256-angular dataset.
Measures recall@20, p50/p95/p99 latency, QPS — single query, single thread, warm cache.

Usage:
    python3 bench/nyt256/run_competitors.py [--out bench/nyt256/results_competitors.jsonl]
    python3 bench/nyt256/run_competitors.py --libs faiss-ivfpq --out bench/nyt256/results_ivfpq.jsonl

Requires: hnswlib usearch faiss-cpu numpy  (pip install or use /tmp/annbench_env)
"""
import argparse, json, time
from pathlib import Path
import numpy as np

DATA = Path("data/nytimes-256-angular")
CACHE = DATA
N_QUERIES = 1000
K = 20
ROUNDS = 3


def load_data():
    base    = np.load(DATA / "base.npy").astype("float32")
    queries = np.load(DATA / "queries.npy")[:N_QUERIES].astype("float32")
    truth   = [t[:K] for t in json.loads((DATA / "ground_truth.json").read_text())[:N_QUERIES]]
    return base, queries, truth

def safe_norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n < 1e-10, 1.0, n)

def recall_at_k(results, truth):
    return sum(len(set(r) & set(t)) for r, t in zip(results, truth)) / (N_QUERIES * K)

def pct(times, p): return float(np.percentile(times, p))

def emit(records, lib, config, ef, rec, times_ms):
    n = len(times_ms)
    qps = n / (sum(times_ms) / 1000)
    records.append({
        "lib": lib, "config": config, "ef": ef,
        "recall": rec, "qps": round(qps, 2),
        "p50_ms": round(pct(times_ms, 50), 4),
        "p95_ms": round(pct(times_ms, 95), 4),
        "p99_ms": round(pct(times_ms, 99), 4),
    })

def _time_rounds(fn, queries):
    """Run fn(q) over queries for ROUNDS rounds; return flat list of per-query ms."""
    times = []
    for _ in range(ROUNDS):
        for q in queries:
            t0 = time.perf_counter()
            fn(q)
            times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_hnswlib(base, queries, truth, records):
    import hnswlib
    for M in [16, 32]:
        idx_path = CACHE / f"hnswlib-m{M}-efc300.bin"
        idx = hnswlib.Index(space="cosine", dim=256)
        if idx_path.exists():
            idx.load_index(str(idx_path))
        else:
            idx.init_index(max_elements=len(base), ef_construction=300, M=M, random_seed=42)
            idx.add_items(base, np.arange(len(base)), num_threads=4)
            idx.save_index(str(idx_path))

        idx.set_ef(64)
        for q in queries: idx.knn_query(q, k=K, num_threads=1)  # warmup

        for ef in [32, 64, 128, 256, 512]:
            idx.set_ef(ef)
            answers = [idx.knn_query(q, k=K, num_threads=1)[0][0].tolist() for q in queries]
            rec = recall_at_k(answers, truth)
            times = _time_rounds(lambda q: idx.knn_query(q, k=K, num_threads=1), queries)
            emit(records, "hnswlib", f"M={M} efc=300", ef, rec, times)
        print(f"  hnswlib M={M} done")


def bench_usearch(base, queries, truth, records):
    from usearch.index import Index, MetricKind
    for M in [16, 32]:
        idx_path = CACHE / f"usearch-m{M}-efc300.usearch"
        idx = Index(ndim=256, metric=MetricKind.Cos, connectivity=M, expansion_add=300)
        if idx_path.exists():
            idx.load(str(idx_path))
        else:
            idx.add(np.arange(len(base), dtype=np.int64), base, threads=4)
            idx.save(str(idx_path))

        idx.expansion_search = 64
        for q in queries: idx.search(q, K)  # warmup

        for ef in [32, 64, 128, 256, 512]:
            idx.expansion_search = ef
            answers = [idx.search(q, K).keys.tolist() for q in queries]
            rec = recall_at_k(answers, truth)
            times = _time_rounds(lambda q: idx.search(q, K), queries)
            emit(records, "usearch", f"M={M} efc=300", ef, rec, times)
        print(f"  usearch M={M} done")


def bench_faiss_hnsw(base, queries, truth, records):
    import faiss
    faiss.omp_set_num_threads(1)
    base_n = safe_norm(base)
    queries_n = safe_norm(queries)
    for M in [16, 32]:
        idx_path = CACHE / f"faiss-hnsw-m{M}-efc300.index"
        if idx_path.exists():
            idx = faiss.read_index(str(idx_path))
        else:
            idx = faiss.IndexHNSWFlat(256, M, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = 300
            idx.add(base_n)
            faiss.write_index(idx, str(idx_path))

        idx.hnsw.efSearch = 64
        for q in queries_n: idx.search(q[None], K)  # warmup

        for ef in [32, 64, 128, 256, 512]:
            idx.hnsw.efSearch = ef
            answers = [idx.search(q[None], K)[1][0].tolist() for q in queries_n]
            rec = recall_at_k(answers, truth)
            times = _time_rounds(lambda q: idx.search(q[None], K), queries_n)
            emit(records, "faiss-hnsw", f"M={M} efc=300", ef, rec, times)
        print(f"  faiss-hnsw M={M} done")


def bench_faiss_hnsw_sq8(base, queries, truth, records):
    """FAISS HNSW + 8-bit scalar quantizer: the direct analogue of ANNex m*+sq8.

    Without this the comparison is unfair to FAISS — our headline speedup comes
    from SQ8 distance screening, and FAISS ships the same trick in IndexHNSWSQ.
    """
    import faiss
    faiss.omp_set_num_threads(1)
    base_n = np.ascontiguousarray(safe_norm(base))
    queries_n = np.ascontiguousarray(safe_norm(queries))

    for M in [16, 32]:
        idx_path = CACHE / f"faiss-hnswsq8-m{M}-efc300.index"
        if idx_path.exists():
            idx = faiss.read_index(str(idx_path))
        else:
            idx = faiss.IndexHNSWSQ(256, faiss.ScalarQuantizer.QT_8bit, M,
                                    faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = 300
            idx.train(base_n)
            idx.add(base_n)
            faiss.write_index(idx, str(idx_path))

        config = f"M={M} efc=300 SQ8"
        print(f"  faiss-hnsw-sq8 M={M} ntotal={idx.ntotal} "
              f"index_bytes={idx_path.stat().st_size}")

        idx.hnsw.efSearch = 64
        for q in queries_n: idx.search(q[None], K)  # warmup

        for ef in [32, 64, 128, 256, 512]:
            idx.hnsw.efSearch = ef
            answers = [idx.search(q[None], K)[1][0].tolist() for q in queries_n]
            rec = recall_at_k(answers, truth)
            times = _time_rounds(lambda q: idx.search(q[None], K), queries_n)
            emit(records, "faiss-hnsw-sq8", config, ef, rec, times)
        print(f"  faiss-hnsw-sq8 M={M} done")


def bench_faiss_ivfpq(base, queries, truth, records):
    import faiss
    faiss.omp_set_num_threads(1)
    base_n = np.ascontiguousarray(safe_norm(base))
    queries_n = np.ascontiguousarray(safe_norm(queries))

    nlist, pq_m, pq_bits = 4096, 32, 8
    idx_path = CACHE / f"faiss-ivfpq-{nlist}-{pq_m}x{pq_bits}.index"
    if idx_path.exists():
        idx = faiss.read_index(str(idx_path))
    else:
        quantizer = faiss.IndexFlatIP(256)
        idx = faiss.IndexIVFPQ(quantizer, 256, nlist, pq_m, pq_bits, faiss.METRIC_INNER_PRODUCT)
        idx.train(base_n)
        idx.add(base_n)
        faiss.write_index(idx, str(idx_path))

    config = f"IVF{nlist} PQ{pq_m}x{pq_bits}"
    print(f"  faiss-ivfpq ntotal={idx.ntotal} index_bytes={idx_path.stat().st_size}")

    idx.nprobe = 64
    for q in queries_n: idx.search(q[None], K)  # warmup

    for nprobe in [8, 16, 32, 64, 128, 256, 512, 1024]:
        idx.nprobe = nprobe
        answers = [idx.search(q[None], K)[1][0].tolist() for q in queries_n]
        rec = recall_at_k(answers, truth)
        times = _time_rounds(lambda q: idx.search(q[None], K), queries_n)
        emit(records, "faiss-ivfpq", config, nprobe, rec, times)
    print(f"  faiss-ivfpq {config} done")

    # Reranked IVF-PQ: exact f32 inner products over k_factor*K PQ candidates.
    # Raw PQ caps recall far below the HNSW operating range (see plateau above),
    # so refine is how IVF-PQ is deployed when recall matters. Costs a full
    # f32 copy of the base vectors, which mostly erases the PQ memory win.
    flat = faiss.IndexFlatIP(256)
    flat.add(base_n)
    rr = faiss.IndexRefine(idx, flat)
    rr.own_fields = False  # idx and flat are owned by this scope, not by rr
    for k_factor in [5, 10, 20]:
        rr.k_factor = k_factor
        rr_config = f"{config} refine k={k_factor}"
        idx.nprobe = 64
        for q in queries_n: rr.search(q[None], K)  # warmup

        for nprobe in [64, 128, 256, 512]:
            idx.nprobe = nprobe
            answers = [rr.search(q[None], K)[1][0].tolist() for q in queries_n]
            rec = recall_at_k(answers, truth)
            times = _time_rounds(lambda q: rr.search(q[None], K), queries_n)
            emit(records, "faiss-ivfpq-rr", rr_config, nprobe, rec, times)
        print(f"  faiss-ivfpq-rr k_factor={k_factor} done")


def _update_manifest() -> None:
    import datetime
    import importlib.metadata as im
    p = Path("bench/nyt256/manifest.json")
    m = json.loads(p.read_text()) if p.exists() else {}
    versions: dict[str, str] = {}
    for lib in ("hnswlib", "usearch", "faiss"):
        try:
            mod = __import__(lib)
            version = getattr(mod, "__version__", None)
            if not version:
                # hnswlib exposes no __version__; fall back to the dist metadata
                dist = "faiss-cpu" if lib == "faiss" else lib
                version = im.version(dist)
            versions[lib] = version
        except (ImportError, im.PackageNotFoundError):
            pass
    m["competitors"] = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "rounds": ROUNDS,
        **versions,
    }
    p.write_text(json.dumps(m, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench/nyt256/results_competitors.jsonl")
    parser.add_argument(
        "--libs", default="hnswlib,usearch,faiss-hnsw,faiss-hnsw-sq8,faiss-ivfpq",
        help="comma-separated subset of: hnswlib,usearch,faiss-hnsw,faiss-hnsw-sq8,faiss-ivfpq",
    )
    args = parser.parse_args()
    libs = [x.strip() for x in args.libs.split(",") if x.strip()]

    print("Loading data...")
    base, queries, truth = load_data()

    records = []
    if "hnswlib" in libs:
        print("Running hnswlib M=16, M=32...")
        bench_hnswlib(base, queries, truth, records)
    if "usearch" in libs:
        print("Running usearch M=16, M=32...")
        bench_usearch(base, queries, truth, records)
    if "faiss-hnsw" in libs:
        print("Running faiss-hnsw M=16, M=32...")
        bench_faiss_hnsw(base, queries, truth, records)
    if "faiss-hnsw-sq8" in libs:
        print("Running faiss-hnsw-sq8 M=16, M=32...")
        bench_faiss_hnsw_sq8(base, queries, truth, records)
    if "faiss-ivfpq" in libs:
        print("Running faiss-ivfpq IVF4096 PQ32x8...")
        bench_faiss_ivfpq(base, queries, truth, records)

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    _update_manifest()
    print(f"\nResults written to {args.out}")
    print(f"{len(records)} data points")
