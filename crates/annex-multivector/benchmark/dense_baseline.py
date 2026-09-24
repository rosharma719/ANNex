#!/usr/bin/env python3
"""Benchmark exact MiniLM search against the vectordb HNSW implementation."""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from data import load_slice
from embeddings import cached_fixed
from env import load_env
from provenance import write_report
from run import http, score, size

load_env()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="beir/fiqa/test")
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--sampling", choices=["prefix", "qrels"], default="qrels")
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-search", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=Path("benchmark/cache"))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/dense-vectordb"))
    parser.add_argument("--report-dir", type=Path, default=Path("benchmark/reports"))
    args = parser.parse_args()

    docs, queries, qrels = load_slice(
        args.dataset, args.limit_docs, args.limit_queries, args.sampling, args.sample_seed
    )
    doc_ids = [doc.doc_id for doc in docs]
    doc_texts = [(getattr(doc, "title", "") + " " + doc.text).strip() for doc in docs]
    query_ids = [query.query_id for query in queries]
    query_texts = [query.text for query in queries]
    model = None

    def encode(texts):
        nonlocal model
        if model is None:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model.encode(
            texts, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True
        )

    documents, document_cache = cached_fixed(
        args.cache_dir, "sentence-transformers/all-MiniLM-L6-v2", "document",
        doc_ids, doc_texts, lambda: encode(doc_texts), True, args.refresh_cache,
    )
    query_vectors, query_cache = cached_fixed(
        args.cache_dir, "sentence-transformers/all-MiniLM-L6-v2", "query",
        query_ids, query_texts, lambda: encode(query_texts), True, args.refresh_cache,
    )

    shutil.rmtree(args.output, ignore_errors=True)
    root = Path(__file__).resolve().parents[1]
    server = subprocess.Popen(
        [
            "cargo", "run", "--release", "--bin", "dense_server", "--",
            "--dimension", str(documents.shape[1]), "--path", str(args.output),
            "--listen", "127.0.0.1:18081",
        ],
        cwd=root,
        stdout=subprocess.DEVNULL,
    )
    base = "http://127.0.0.1:18081"
    try:
        for _ in range(60):
            try:
                http(base, "/healthz")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("dense baseline server did not start")
        for start in range(0, len(docs), 256):
            http(
                base,
                "/v1/vectors/upsert",
                {"documents": [
                    {"id": doc_ids[index], "vector": documents[index].tolist()}
                    for index in range(start, min(start + 256, len(docs)))
                ]},
            )
        started = time.perf_counter()
        http(base, "/v1/index", {"m": args.m, "ef_construct": args.ef_search})
        build_seconds = time.perf_counter() - started

        runs = {"exact_minilm": {}, "vectordb_hnsw_minilm": {}}
        latencies = {name: [] for name in runs}
        overlaps = []
        for query, vector in zip(queries, query_vectors):
            results = {}
            for name, backend in (("exact_minilm", "exact"), ("vectordb_hnsw_minilm", "hnsw")):
                started = time.perf_counter()
                result = http(
                    base, "/v1/query",
                    {"vector": vector.tolist(), "top_k": 100, "backend": backend, "ef_search": args.ef_search},
                )
                latencies[name].append(time.perf_counter() - started)
                results[name] = [match["id"] for match in result["matches"]]
                runs[name][query.query_id] = results[name]
            exact_ids = set(results["exact_minilm"])
            ann_ids = set(results["vectordb_hnsw_minilm"])
            overlaps.append(len(exact_ids & ann_ids) / max(1, len(exact_ids)))
        stats = http(base, "/v1/stats")
    finally:
        server.terminate()
        server.wait(timeout=30)

    systems = {}
    for name, run in runs.items():
        systems[name] = {
            **score(run, qrels),
            "p50_ms": float(np.percentile(latencies[name], 50) * 1000),
            "p95_ms": float(np.percentile(latencies[name], 95) * 1000),
        }
    systems["exact_minilm"]["storage_bytes"] = int(documents.nbytes)
    systems["vectordb_hnsw_minilm"].update(
        {
            "storage_bytes": size(args.output),
            "build_seconds": build_seconds,
            "top100_recall_vs_exact": float(np.mean(overlaps)),
            "m": args.m,
            "ef_search": args.ef_search,
            "index_stats": stats,
        }
    )
    report = {
        "dataset": args.dataset,
        "sampling": args.sampling,
        "sample_seed": args.sample_seed,
        "documents": len(docs),
        "queries": len(queries),
        "embedding_cache": {
            "minilm_documents": document_cache,
            "minilm_queries": query_cache,
        },
        "systems": systems,
    }
    path = write_report(args.report_dir, "dense-baseline", report)
    print(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
