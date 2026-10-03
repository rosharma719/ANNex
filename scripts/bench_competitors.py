#!/usr/bin/env python3
"""
Generic competitor sweep for any ann-benchmarks-format dataset.

Measures recall@k, p50/p95/p99 latency, QPS for hnswlib, usearch, and faiss-hnsw.
Handles cosine, euclidean, and dot (inner-product) metrics.

Usage:
    python3 scripts/bench_competitors.py \\
        --data-dir data/sift-128-euclidean \\
        --metric euclidean \\
        --dims 128 \\
        --out bench/sift1m/results_competitors.jsonl

Requires: hnswlib usearch faiss-cpu numpy
"""
import argparse, json, time
from pathlib import Path
import numpy as np

ROUNDS = 3
K = 10


def load_data(data_dir: Path, n_queries: int):
    base    = np.load(data_dir / "base.npy").astype("float32", copy=False)
    queries = np.load(data_dir / "queries.npy")[:n_queries].astype("float32", copy=False)
    truth   = json.loads((data_dir / "ground_truth.json").read_text())[:n_queries]
    truth   = [t[:K] for t in truth]
    return base, queries, truth


def safe_norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n < 1e-10, 1.0, n)


def recall_at_k(results, truth):
    return sum(len(set(r) & set(t)) for r, t in zip(results, truth)) / (len(results) * K)


def pct(times, p):
    return float(np.percentile(times, p))


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
    times = []
    for _ in range(ROUNDS):
        for q in queries:
            t0 = time.perf_counter()
            fn(q)
            times.append((time.perf_counter() - t0) * 1000)
    return times


def hnswlib_space(metric: str) -> str:
    return {"cosine": "cosine", "euclidean": "l2", "dot": "ip"}[metric]


def bench_hnswlib(base, queries, truth, records, metric, data_dir, cleanup_indexes=False):
    import hnswlib
    space = hnswlib_space(metric)
    # For IP (dot), vectors should not be normalized; for cosine, hnswlib normalizes internally
    q_use = queries
    dims = base.shape[1]

    for M in [16, 32]:
        idx_path = data_dir / f"hnswlib-m{M}-efc300.bin"
        idx = hnswlib.Index(space=space, dim=dims)
        if idx_path.exists():
            idx.load_index(str(idx_path))
        else:
            idx.init_index(max_elements=len(base), ef_construction=300, M=M, random_seed=42)
            idx.add_items(base, np.arange(len(base)), num_threads=4)
            idx.save_index(str(idx_path))

        idx.set_ef(64)
        for q in q_use: idx.knn_query(q, k=K, num_threads=1)  # warmup

        for ef in [32, 64, 128, 256, 512]:
            idx.set_ef(ef)
            answers = [idx.knn_query(q, k=K, num_threads=1)[0][0].tolist() for q in q_use]
            rec = recall_at_k(answers, truth)
            times = _time_rounds(lambda q: idx.knn_query(q, k=K, num_threads=1), q_use)
            emit(records, "hnswlib", f"M={M} efc=300", ef, rec, times)
        print(f"  hnswlib M={M} done")
        if cleanup_indexes:
            del idx
            idx_path.unlink(missing_ok=True)


def bench_usearch(base, queries, truth, records, metric, data_dir, cleanup_indexes=False):
    from usearch.index import Index, MetricKind
    mk = {"cosine": MetricKind.Cos, "euclidean": MetricKind.L2sq, "dot": MetricKind.IP}[metric]
    dims = base.shape[1]

    for M in [16, 32]:
        idx_path = data_dir / f"usearch-m{M}-efc300.usearch"
        idx = Index(ndim=dims, metric=mk, connectivity=M, expansion_add=300)
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
        if cleanup_indexes:
            del idx
            idx_path.unlink(missing_ok=True)


def bench_faiss_hnsw(base, queries, truth, records, metric, data_dir, cleanup_indexes=False):
    import faiss
    faiss.omp_set_num_threads(1)
    dims = base.shape[1]

    if metric == "euclidean":
        faiss_metric = faiss.METRIC_L2
        base_use, queries_use = base, queries
    else:
        # cosine and dot: use inner product on normalised vectors
        faiss_metric = faiss.METRIC_INNER_PRODUCT
        base_use   = safe_norm(base)   if metric == "cosine" else base
        queries_use = safe_norm(queries) if metric == "cosine" else queries

    for M in [16, 32]:
        idx_path = data_dir / f"faiss-hnsw-m{M}-efc300.index"
        if idx_path.exists():
            idx = faiss.read_index(str(idx_path))
        else:
            idx = faiss.IndexHNSWFlat(dims, M, faiss_metric)
            idx.hnsw.efConstruction = 300
            idx.add(np.ascontiguousarray(base_use))
            faiss.write_index(idx, str(idx_path))

        idx.hnsw.efSearch = 64
        for q in queries_use: idx.search(q[None], K)  # warmup

        for ef in [32, 64, 128, 256, 512]:
            idx.hnsw.efSearch = ef
            answers = [idx.search(q[None], K)[1][0].tolist() for q in queries_use]
            rec = recall_at_k(answers, truth)
            times = _time_rounds(lambda q: idx.search(q[None], K), queries_use)
            emit(records, "faiss-hnsw", f"M={M} efc=300", ef, rec, times)
        print(f"  faiss-hnsw M={M} done")
        if cleanup_indexes:
            del idx
            idx_path.unlink(missing_ok=True)


def _update_manifest(out_path: Path) -> None:
    import datetime
    bench_dir = out_path.parent
    p = bench_dir / "manifest.json"
    m = json.loads(p.read_text()) if p.exists() else {}
    versions: dict = {}
    for lib in ("hnswlib", "usearch", "faiss"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    m["competitors"] = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        **versions,
    }
    p.write_text(json.dumps(m, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--metric", required=True, choices=["cosine", "euclidean", "dot"])
    ap.add_argument("--dims", required=True, type=int)
    ap.add_argument("--n-queries", default=1000, type=int)
    ap.add_argument("--k", default=10, type=int)
    ap.add_argument("--out", default=None, type=Path,
                    help="Output JSONL path (default: auto-derived from data-dir name)")
    ap.add_argument("--libs", default="hnswlib,usearch,faiss-hnsw",
                    help="comma-separated libs to run")
    ap.add_argument("--cleanup-indexes", action="store_true",
                    help="delete each cached competitor index after measuring it")
    args = ap.parse_args()
    global K
    K = args.k

    libs = [l.strip() for l in args.libs.split(",")]
    out = args.out
    if out is None:
        # Derive bench dir name from dataset dir name, e.g. sift-128-euclidean -> sift1m
        mapping = {
            "sift-128-euclidean": "sift1m",
            "glove-100-angular": "glove100",
            "lastfm-64-dot": "lastfm64",
            "mnist-784-euclidean": "mnist784",
            "nytimes-256-angular": "nyt256",
            "gist-960-euclidean": "gist1m",
        }
        bench = mapping.get(args.data_dir.name, args.data_dir.name)
        out = Path(f"bench/{bench}/results_competitors.jsonl")

    print(f"Loading data from {args.data_dir} ({args.metric}, {args.dims}D)...")
    base, queries, truth = load_data(args.data_dir, args.n_queries)
    print(f"  base={base.shape}  queries={queries.shape}")

    records = []
    if "hnswlib" in libs:
        print("Running hnswlib...")
        bench_hnswlib(base, queries, truth, records, args.metric, args.data_dir,
                      args.cleanup_indexes)
    if "usearch" in libs:
        print("Running usearch...")
        bench_usearch(base, queries, truth, records, args.metric, args.data_dir,
                      args.cleanup_indexes)
    if "faiss-hnsw" in libs:
        print("Running faiss-hnsw...")
        bench_faiss_hnsw(base, queries, truth, records, args.metric, args.data_dir,
                         args.cleanup_indexes)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    _update_manifest(out)
    print(f"\nResults written to {out}  ({len(records)} rows)")


if __name__ == "__main__":
    main()
