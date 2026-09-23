#!/usr/bin/env python3
"""Measure the exhaustive compressed-MaxSim ceiling and ANN candidate recall."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from colbert_config import MODEL_ID, cache_config, load as load_colbert
from data import load_slice, slice_fingerprint
from embeddings import cached_ragged
from provenance import write_report
from run import colbert_encode, http, score


def percentile(values, percentile):
    return float(np.percentile(values, percentile) * 1000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dataset", default="beir/fiqa/test")
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--sampling", choices=["prefix", "qrels"], default="qrels")
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--centroids", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=250)
    parser.add_argument("--ef-search", type=int, default=256)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--report-dir", type=Path, default=Path("benchmark/reports"))
    parser.add_argument("--cache-dir", type=Path, default=Path("benchmark/cache"))
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    docs, queries, qrels = load_slice(
        args.dataset, args.limit_docs, args.limit_queries, args.sampling, args.sample_seed
    )
    manifest_path = args.index.parent / "slice.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        actual = slice_fingerprint(docs, queries)
        if manifest.get("fingerprint") != actual:
            raise RuntimeError(
                "benchmark slice does not match the index; rebuild with the same sampling arguments"
            )

    query_texts = [query.text for query in queries]
    vectors, query_cache = cached_ragged(
        args.cache_dir,
        MODEL_ID,
        "query",
        [query.query_id for query in queries],
        query_texts,
        lambda: colbert_encode(
            load_colbert(), query_texts, True, 32
        ),
        args.refresh_cache,
        cache_config("query"),
    )
    root = Path(__file__).resolve().parents[1]
    command = [
        "cargo", "run", "--release", "--bin", "multivector", "--", "--dimension", "128",
        "--centroids", str(args.centroids), "--probes", "8", "--path",
        str(args.index), "--listen", "127.0.0.1:18080",
    ]
    server = subprocess.Popen(command, cwd=root, stdout=subprocess.DEVNULL)
    base = "http://127.0.0.1:18080"
    try:
        for _ in range(60):
            try:
                stats = http(base, "/v1/stats")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not start")
        started = time.perf_counter()
        http(base, "/v1/fde/index", {"m": args.hnsw_m, "ef_construct": args.ef_search})
        build_seconds = time.perf_counter() - started

        runs = {"exhaustive": {}, "exact_fde": {}, "hnsw": {}}
        latencies = {name: [] for name in runs}
        overlaps, exact_relevant, ann_relevant = [], [], []
        for query, vector in zip(queries, vectors):
            vector = np.asarray(vector).tolist()
            request = {"vectors": vector, "count": args.candidates}
            exact = http(base, "/v1/debug/candidates", request)["candidates"]
            ann = http(
                base,
                "/v1/debug/candidates",
                {**request, "candidate_backend": "hnsw", "ef_search": args.ef_search},
            )["candidates"]
            exact_ids = {candidate["id"] for candidate in exact}
            ann_ids = {candidate["id"] for candidate in ann}
            overlaps.append(len(exact_ids & ann_ids) / max(1, len(exact_ids)))
            relevant = set(qrels[query.query_id])
            exact_relevant.append(len(exact_ids & relevant) / len(relevant))
            ann_relevant.append(len(ann_ids & relevant) / len(relevant))

            configurations = {
                "exhaustive": {"vectors": vector, "top_k": 100, "candidates": stats["documents"]},
                "exact_fde": {"vectors": vector, "top_k": 100, "candidates": args.candidates},
                "hnsw": {"vectors": vector, "top_k": 100, "candidates": args.candidates, "candidate_backend": "hnsw", "ef_search": args.ef_search},
            }
            for name, body in configurations.items():
                started = time.perf_counter()
                result = http(base, "/v1/query", body)
                latencies[name].append(time.perf_counter() - started)
                runs[name][query.query_id] = [match["id"] for match in result["matches"]]
    finally:
        server.terminate()
        server.wait(timeout=30)

    systems = {}
    for name, run in runs.items():
        systems[name] = {
            **score(run, qrels),
            "p50_ms": percentile(latencies[name], 50),
            "p95_ms": percentile(latencies[name], 95),
        }
    report = {
        "dataset": args.dataset,
        "sampling": args.sampling,
        "sample_seed": args.sample_seed,
        "documents": len(docs),
        "queries": len(queries),
        "candidates": args.candidates,
        "ef_search": args.ef_search,
        "hnsw_build_seconds": build_seconds,
        "embedding_cache": {"colbert_queries": query_cache},
        "candidate_recall_vs_exact_fde": float(np.mean(overlaps)),
        "exact_fde_relevant_recall": float(np.mean(exact_relevant)),
        "hnsw_relevant_recall": float(np.mean(ann_relevant)),
        "systems": systems,
    }
    path = write_report(args.report_dir, "diagnostic", report)
    print(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
