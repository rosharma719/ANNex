#!/usr/bin/env python3
"""
Run hnswlib, usearch, and FAISS HNSW sweeps over the NYT-256-angular dataset.
Measures recall@20, p50/p95/p99 latency, QPS — single query, single thread, warm cache.

Usage:
    python3 bench/nyt256/run_competitors.py [--out bench/nyt256/results_competitors.jsonl]

Requires: hnswlib usearch faiss-cpu numpy  (pip install or use /tmp/annbench_env)
"""
import argparse, json, time, os
from pathlib import Path
import numpy as np

DATA = Path("data/nytimes-256-angular")
CACHE = DATA
N_QUERIES = 1000
K = 20
ROUNDS = 1  # single round; increase for tighter numbers

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
    qps = N_QUERIES / (sum(times_ms) / 1000)
    records.append({
        "lib": lib, "config": config, "ef": ef,
        "recall": rec, "qps": round(qps, 2),
        "p50_ms": round(pct(times_ms, 50), 4),
        "p95_ms": round(pct(times_ms, 95), 4),
        "p99_ms": round(pct(times_ms, 99), 4),
    })

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

        # warmup
        idx.set_ef(64)
        for q in queries: idx.knn_query(q, k=K, num_threads=1)

        for ef in [32, 64, 128, 256, 512]:
            idx.set_ef(ef)
            answers = [idx.knn_query(q, k=K, num_threads=1)[0][0].tolist() for q in queries]
            rec = recall_at_k(answers, truth)
            times = []
            for q in queries:
                t0 = time.perf_counter()
                idx.knn_query(q, k=K, num_threads=1)
                times.append((time.perf_counter() - t0) * 1000)
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

        # warmup
        idx.expansion_search = 64
        for q in queries: idx.search(q, K)

        for ef in [32, 64, 128, 256, 512]:
            idx.expansion_search = ef
            answers = [idx.search(q, K).keys.tolist() for q in queries]
            rec = recall_at_k(answers, truth)
            times = []
            for q in queries:
                t0 = time.perf_counter()
                idx.search(q, K)
                times.append((time.perf_counter() - t0) * 1000)
            emit(records, "usearch", f"M={M} efc=300", ef, rec, times)
        print(f"  usearch M={M} done")

def bench_faiss_hnsw(base, queries, truth, records):
    import faiss
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

        # warmup
        idx.hnsw.efSearch = 64
        for q in queries_n: idx.search(q[None], K)

        for ef in [32, 64, 128, 256, 512]:
            idx.hnsw.efSearch = ef
            answers = [idx.search(q[None], K)[1][0].tolist() for q in queries_n]
            rec = recall_at_k(answers, truth)
            times = []
            for q in queries_n:
                t0 = time.perf_counter()
                idx.search(q[None], K)
                times.append((time.perf_counter() - t0) * 1000)
            emit(records, "faiss-hnsw", f"M={M} efc=300", ef, rec, times)
        print(f"  faiss-hnsw M={M} done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench/nyt256/results_competitors.jsonl")
    args = parser.parse_args()

    print("Loading data...")
    base, queries, truth = load_data()

    records = []
    print("Running hnswlib M=16, M=32...")
    bench_hnswlib(base, queries, truth, records)
    print("Running usearch M=16, M=32...")
    bench_usearch(base, queries, truth, records)
    print("Running faiss-hnsw M=16, M=32...")
    bench_faiss_hnsw(base, queries, truth, records)

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults written to {args.out}")
    print(f"{len(records)} data points")
