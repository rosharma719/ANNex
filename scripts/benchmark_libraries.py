#!/usr/bin/env python3
"""
Same-machine library comparison: hnswlib, usearch, faiss-hnsw, faiss-ivf, annoy.
All use the same NYT-256-angular vectors, recall@20, single-query single-thread.
Results are JSON lines, one per (library, config, ef/params, round).
"""
import json, time, math
from pathlib import Path
import numpy as np

DATA = Path("data/nytimes-256-angular")
CACHE = DATA  # store built indexes alongside data
N_QUERIES = 1000
K = 20        # recall@20 to match ANNex benchmarks
ROUNDS = 3

def load_data():
    base    = np.load(DATA / "base.npy").astype("float32")
    queries = np.load(DATA / "queries.npy").astype("float32")
    truth   = json.loads((DATA / "ground_truth.json").read_text())
    return base, queries[:N_QUERIES], [t[:K] for t in truth[:N_QUERIES]]

def recall_at_k(results, truth):
    hits = sum(len(set(r) & set(t)) for r, t in zip(results, truth))
    return hits / (N_QUERIES * K)

def p50(times): return float(np.percentile(times, 50))
def p99(times): return float(np.percentile(times, 99))

def emit(lib, config, ef_label, ef_val, rec, times_ms):
    med = float(np.median([sum(r)/len(r) for r in [times_ms[i::ROUNDS] for i in range(ROUNDS)]]))
    # times_ms is all rounds concatenated; group by round
    per_round = [times_ms[i*N_QUERIES:(i+1)*N_QUERIES] for i in range(ROUNDS)]
    for rnd, rt in enumerate(per_round):
        qps = N_QUERIES / (sum(rt) / 1000)
        print(json.dumps({
            "lib": lib, "config": config, "ef": ef_label, "ef_val": ef_val,
            "round": rnd, "recall": rec,
            "qps": qps,
            "p50_ms": p50(rt), "p99_ms": p99(rt),
        }), flush=True)

# ── hnswlib ──────────────────────────────────────────────────────────────────
def bench_hnswlib(base, queries, truth):
    import hnswlib
    idx_path = CACHE / "hnswlib-m16-efc300.bin"
    idx = hnswlib.Index(space="cosine", dim=256)
    if idx_path.exists():
        idx.load_index(str(idx_path))
    else:
        idx.init_index(max_elements=len(base), ef_construction=300, M=16, random_seed=42)
        idx.add_items(base, np.arange(len(base)), num_threads=4)
        idx.save_index(str(idx_path))

    for ef in [32, 64, 128, 256, 512]:
        idx.set_ef(ef)
        answers = [idx.knn_query(q, k=K, num_threads=1)[0][0].tolist() for q in queries]
        rec = recall_at_k(answers, truth)
        all_times = []
        for _ in range(ROUNDS):
            for q in queries:
                t0 = time.perf_counter()
                idx.knn_query(q, k=K, num_threads=1)
                all_times.append((time.perf_counter() - t0) * 1000)
        emit("hnswlib", "M=16 efc=300", ef, ef, rec, all_times)

# ── usearch ──────────────────────────────────────────────────────────────────
def bench_usearch(base, queries, truth):
    from usearch.index import Index, MetricKind, ScalarKind, CompiledMetric
    idx_path = CACHE / "usearch-m16-efc300.usearch"
    idx = Index(ndim=256, metric=MetricKind.Cos, connectivity=16,
                expansion_add=300, expansion_search=64)
    if idx_path.exists():
        idx.load(str(idx_path))
    else:
        keys = np.arange(len(base), dtype=np.int64)
        idx.add(keys, base, threads=4)
        idx.save(str(idx_path))

    for ef in [32, 64, 128, 256, 512]:
        idx.expansion_search = ef
        answers = []
        for q in queries:
            res = idx.search(q, K)
            answers.append(res.keys.tolist())
        rec = recall_at_k(answers, truth)
        all_times = []
        for _ in range(ROUNDS):
            for q in queries:
                t0 = time.perf_counter()
                idx.search(q, K)
                all_times.append((time.perf_counter() - t0) * 1000)
        emit("usearch", "M=16 efc=300", ef, ef, rec, all_times)

# ── FAISS HNSW ───────────────────────────────────────────────────────────────
def bench_faiss_hnsw(base, queries, truth):
    import faiss
    idx_path = CACHE / "faiss-hnsw-m32.index"
    norm = np.linalg.norm(base, axis=1, keepdims=True)
    base_n = (base / norm).astype("float32")
    if idx_path.exists():
        idx = faiss.read_index(str(idx_path))
    else:
        # HNSW with M=32; faiss uses inner product on L2-normalised vectors for cosine
        idx = faiss.IndexHNSWFlat(256, 32, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 300
        idx.add(base_n)
        faiss.write_index(idx, str(idx_path))

    norm_q = np.linalg.norm(queries, axis=1, keepdims=True)
    queries_n = (queries / norm_q).astype("float32")

    for ef in [32, 64, 128, 256, 512]:
        idx.hnsw.efSearch = ef
        answers = []
        for q in queries_n:
            _, labels = idx.search(q[None], K)
            answers.append(labels[0].tolist())
        rec = recall_at_k(answers, truth)
        all_times = []
        for _ in range(ROUNDS):
            for q in queries_n:
                t0 = time.perf_counter()
                idx.search(q[None], K)
                all_times.append((time.perf_counter() - t0) * 1000)
        emit("faiss-hnsw", "M=32 efc=300", ef, ef, rec, all_times)

# ── FAISS IVF-PQ ─────────────────────────────────────────────────────────────
def bench_faiss_ivfpq(base, queries, truth):
    import faiss
    idx_path = CACHE / "faiss-ivfpq-4096-32.index"
    norm = np.linalg.norm(base, axis=1, keepdims=True)
    base_n = (base / norm).astype("float32")
    norm_q = np.linalg.norm(queries, axis=1, keepdims=True)
    queries_n = (queries / norm_q).astype("float32")

    if idx_path.exists():
        idx = faiss.read_index(str(idx_path))
    else:
        # IVF4096, PQ32x8 (32 sub-vectors, 8 bits each → 32 bytes per vector)
        quantizer = faiss.IndexFlatIP(256)
        idx = faiss.IndexIVFPQ(quantizer, 256, 4096, 32, 8, faiss.METRIC_INNER_PRODUCT)
        idx.train(base_n)
        idx.add(base_n)
        faiss.write_index(idx, str(idx_path))

    for nprobe in [8, 16, 32, 64, 128, 256]:
        idx.nprobe = nprobe
        answers = []
        for q in queries_n:
            _, labels = idx.search(q[None], K)
            answers.append(labels[0].tolist())
        rec = recall_at_k(answers, truth)
        all_times = []
        for _ in range(ROUNDS):
            for q in queries_n:
                t0 = time.perf_counter()
                idx.search(q[None], K)
                all_times.append((time.perf_counter() - t0) * 1000)
        emit("faiss-ivfpq", "IVF4096 PQ32x8", nprobe, nprobe, rec, all_times)

# ── Annoy ────────────────────────────────────────────────────────────────────
def bench_annoy(base, queries, truth):
    from annoy import AnnoyIndex
    idx_path = CACHE / "annoy-angular-100trees.ann"
    n_trees = 100
    idx = AnnoyIndex(256, "angular")
    if idx_path.exists():
        idx.load(str(idx_path))
    else:
        for i, v in enumerate(base):
            idx.add_item(i, v.tolist())
        idx.build(n_trees)
        idx.save(str(idx_path))

    for search_k_mult in [100, 500, 2000, 8000, 32000]:
        search_k = search_k_mult
        answers = [idx.get_nns_by_vector(q.tolist(), K, search_k=search_k) for q in queries]
        rec = recall_at_k(answers, truth)
        all_times = []
        for _ in range(ROUNDS):
            for q in queries:
                t0 = time.perf_counter()
                idx.get_nns_by_vector(q.tolist(), K, search_k=search_k)
                all_times.append((time.perf_counter() - t0) * 1000)
        emit("annoy", f"{n_trees} trees", search_k, search_k, rec, all_times)

if __name__ == "__main__":
    base, queries, truth = load_data()
    print("# building/loading indexes and benchmarking — one JSON line per result",
          flush=True)
    bench_hnswlib(base, queries, truth)
    bench_usearch(base, queries, truth)
    bench_faiss_hnsw(base, queries, truth)
    bench_faiss_ivfpq(base, queries, truth)
    bench_annoy(base, queries, truth)
