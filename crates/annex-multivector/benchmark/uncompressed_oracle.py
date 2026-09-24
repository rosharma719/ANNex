#!/usr/bin/env python3
"""Compare cached uncompressed ColBERT with PLAID-compressed MaxSim."""
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
from significance import paired_bootstrap


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def maxsim_scores(query, documents, indices=None):
    query = normalize_rows(query)
    if indices is None:
        values = np.asarray(documents.values)
        offsets = np.asarray(documents.offsets)
    else:
        selected = [np.asarray(documents[index]) for index in indices]
        lengths = np.asarray([len(value) for value in selected], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        values = np.concatenate(selected, axis=0)
    # PyLate outputs normalized token vectors. Avoid copying the full corpus if
    # norms confirm that invariant; normalize only if a cache came from another encoder.
    sample = values[:: max(1, len(values) // 4096)]
    if not np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=1e-4):
        values = normalize_rows(values)
    similarities = query @ values.T
    maxima = np.maximum.reduceat(similarities, offsets[:-1], axis=1)
    return maxima.sum(axis=0)


def ranked_ids(ids, scores, top_k=100):
    order = sorted(range(len(ids)), key=lambda index: (-float(scores[index]), ids[index]))
    return [ids[index] for index in order[:top_k]]


def latency_report(values):
    return {
        "p50_ms": float(np.percentile(values, 50) * 1000),
        "p95_ms": float(np.percentile(values, 95) * 1000),
    }


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
    parser.add_argument("--scope", choices=["candidates", "exhaustive", "both"], default="candidates")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=Path("benchmark/cache"))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=Path("benchmark/reports"))
    args = parser.parse_args()

    docs, queries, qrels = load_slice(
        args.dataset, args.limit_docs, args.limit_queries, args.sampling, args.sample_seed
    )
    manifest_path = args.index.parent / "slice.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") != slice_fingerprint(docs, queries):
            raise RuntimeError("benchmark slice does not match the index")

    doc_ids = [doc.doc_id for doc in docs]
    doc_texts = [(getattr(doc, "title", "") + " " + doc.text).strip() for doc in docs]
    query_ids = [query.query_id for query in queries]
    query_texts = [query.text for query in queries]
    model = None

    def encode(texts, is_query):
        nonlocal model
        if model is None:
            model = load_colbert()
        return colbert_encode(model, texts, is_query, args.batch_size)

    document_vectors, document_cache = cached_ragged(
        args.cache_dir, MODEL_ID, "document", doc_ids, doc_texts,
        lambda: encode(doc_texts, False), args.refresh_cache, cache_config("document"),
    )
    query_vectors, query_cache = cached_ragged(
        args.cache_dir, MODEL_ID, "query", query_ids, query_texts,
        lambda: encode(query_texts, True), args.refresh_cache, cache_config("query"),
    )
    doc_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}

    root = Path(__file__).resolve().parents[1]
    command = [
        "cargo", "run", "--release", "--bin", "multivector", "--", "--dimension", "128", "--centroids",
        str(args.centroids), "--probes", "8", "--path", str(args.index),
        "--listen", "127.0.0.1:18080",
    ]
    server = subprocess.Popen(command, cwd=root, stdout=subprocess.DEVNULL)
    base = "http://127.0.0.1:18080"
    compressed_run, uncompressed_candidate_run, exhaustive_run = {}, {}, {}
    compressed_times, candidate_times, exhaustive_times = [], [], []
    try:
        for _ in range(60):
            try:
                http(base, "/healthz")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not start")

        for query, vector in zip(queries, query_vectors):
            vector_json = np.asarray(vector).tolist()
            candidates = http(
                base, "/v1/debug/candidates",
                {"vectors": vector_json, "count": args.candidates},
            )["candidates"]
            candidate_ids = [candidate["id"] for candidate in candidates]
            indices = [doc_index[doc_id] for doc_id in candidate_ids]

            started = time.perf_counter()
            result = http(
                base, "/v1/query",
                {"vectors": vector_json, "top_k": 100, "candidates": args.candidates},
            )
            compressed_times.append(time.perf_counter() - started)
            compressed_run[query.query_id] = [match["id"] for match in result["matches"]]

            if args.scope in ("candidates", "both"):
                started = time.perf_counter()
                scores = maxsim_scores(vector, document_vectors, indices)
                candidate_times.append(time.perf_counter() - started)
                uncompressed_candidate_run[query.query_id] = ranked_ids(candidate_ids, scores)

            if args.scope in ("exhaustive", "both"):
                started = time.perf_counter()
                scores = maxsim_scores(vector, document_vectors)
                exhaustive_times.append(time.perf_counter() - started)
                exhaustive_run[query.query_id] = ranked_ids(doc_ids, scores)
    finally:
        server.terminate()
        server.wait(timeout=30)

    systems = {
        "compressed_same_candidates": {
            **score(compressed_run, qrels),
            **latency_report(compressed_times),
            "latency_scope": "candidate_generation_and_maxsim",
        }
    }
    if uncompressed_candidate_run:
        systems["uncompressed_same_candidates"] = {
            **score(uncompressed_candidate_run, qrels),
            **latency_report(candidate_times),
            "latency_scope": "maxsim_only",
        }
    if exhaustive_run:
        systems["uncompressed_exhaustive"] = {
            **score(exhaustive_run, qrels),
            **latency_report(exhaustive_times),
            "latency_scope": "maxsim_only",
        }
    comparisons = {}
    if uncompressed_candidate_run:
        comparisons["uncompressed_vs_compressed_same_candidates"] = paired_bootstrap(
            compressed_run, uncompressed_candidate_run, qrels, seed=args.sample_seed
        )
    if exhaustive_run and uncompressed_candidate_run:
        comparisons["exhaustive_vs_same_candidates"] = paired_bootstrap(
            uncompressed_candidate_run, exhaustive_run, qrels, seed=args.sample_seed
        )
    report = {
        "dataset": args.dataset,
        "sampling": args.sampling,
        "sample_seed": args.sample_seed,
        "documents": len(docs),
        "queries": len(queries),
        "candidates": args.candidates,
        "scope": args.scope,
        "embedding_cache": {
            "colbert_documents": document_cache,
            "colbert_queries": query_cache,
        },
        "systems": systems,
        "comparisons": comparisons,
    }
    path = write_report(args.report_dir, "uncompressed-oracle", report)
    print(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
