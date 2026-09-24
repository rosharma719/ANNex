#!/usr/bin/env python3
"""Sweep a completed index without rebuilding or re-encoding documents."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from colbert_config import MODEL_ID, cache_config, load as load_colbert
from data import load_slice
from embeddings import cached_ragged
from provenance import write_report
from run import colbert_encode, http, score


def report_for(queries, vectors, qrels, base, backend, candidates, probes, ef_search, documents, rerank_candidates):
    run, latency = {}, []
    for query, vector in zip(queries, vectors):
        body = {"vectors": np.asarray(vector).tolist(), "top_k": 100, "candidates": candidates}
        if rerank_candidates is not None:
            body["rerank_candidates"] = rerank_candidates
        if backend == "centroid":
            body["probes"] = probes
        if backend == "hnsw":
            body.update({"candidate_backend": "hnsw", "ef_search": ef_search})
        started = time.perf_counter()
        result = http(base, "/v1/query", body)
        latency.append(time.perf_counter() - started)
        run[query.query_id] = [match["id"] for match in result["matches"]]
    return {
        "dataset": None,
        "queries": len(queries),
        "backend": backend,
        "probes": probes if backend == "centroid" else None,
        "ef_search": ef_search if backend == "hnsw" else None,
        "candidates": candidates,
        "candidate_fraction": candidates / documents,
        "rerank_candidates": rerank_candidates,
        "rerank_fraction": None if rerank_candidates is None else rerank_candidates / documents,
        **score(run, qrels),
        "p50_ms": float(np.percentile(latency, 50) * 1000),
        "p95_ms": float(np.percentile(latency, 95) * 1000),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("benchmark/results/nfcorpus/plaid"))
    parser.add_argument("--dataset", default="beir/nfcorpus/test")
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--sampling", choices=["prefix", "qrels"], default="prefix")
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--centroids", type=int, default=256)
    parser.add_argument("--configured-probes", type=int, default=8)
    parser.add_argument("--backend", choices=["muvera", "centroid", "hnsw"], default="muvera")
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--ef-search", type=int, default=256)
    parser.add_argument("--ef-search-grid", type=int, nargs="+")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--candidate-grid", type=int, nargs="+")
    parser.add_argument("--rerank-candidates", type=int)
    parser.add_argument("--report-dir", type=Path, default=Path("benchmark/reports"))
    parser.add_argument("--cache-dir", type=Path, default=Path("benchmark/cache"))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--output", type=Path, help="deprecated; all records append to the version ledger")
    args = parser.parse_args()
    if args.output is not None:
        print("--output is deprecated and ignored; appending to the version ledger", file=sys.stderr)

    docs, queries, qrels = load_slice(
        args.dataset, args.limit_docs, args.limit_queries, args.sampling, args.sample_seed
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
    command = ["cargo", "run", "--release", "--bin", "multivector", "--", "--dimension", "128", "--centroids", str(args.centroids), "--probes", str(args.configured_probes), "--path", str(args.index), "--listen", "127.0.0.1:18080"]
    server = subprocess.Popen(command, cwd=root, stdout=subprocess.DEVNULL)
    base = "http://127.0.0.1:18080"
    candidates = args.candidate_grid or [args.candidates]
    ef_searches = args.ef_search_grid or [args.ef_search]
    try:
        for _ in range(60):
            try:
                http(base, "/healthz")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not start")
        if args.backend == "hnsw":
            # Build once: every grid point below reuses this graph.
            http(base, "/v1/fde/index", {"m": args.hnsw_m, "ef_construct": max(ef_searches)})
        for candidate_count in candidates:
            searches = ef_searches if args.backend == "hnsw" else [args.ef_search]
            for ef_search in searches:
                report = report_for(
                    queries, vectors, qrels, base, args.backend, candidate_count,
                    args.probes, ef_search, len(docs), args.rerank_candidates,
                )
                report["dataset"] = args.dataset
                report["sampling"] = args.sampling
                report["sample_seed"] = args.sample_seed
                report["embedding_cache"] = {"colbert_queries": query_cache}
                path = write_report(args.report_dir, "sweep", report)
                print(path)
                print(json.dumps(report, indent=2))
    finally:
        server.terminate()
        server.wait(timeout=30)


if __name__ == "__main__":
    main()
