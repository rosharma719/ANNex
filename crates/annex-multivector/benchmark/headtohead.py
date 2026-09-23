#!/usr/bin/env python3
"""Multi-vector head-to-head: annex-multivector vs Qdrant vs LanceDB.

Same corpus, same ColBERT-v2 embeddings, same query set, same M2 CPU.
Reuses the cached embeddings produced by benchmark/run.py so we don't
re-encode anything — all three engines see byte-identical inputs.

Usage:
    .venv/bin/python benchmark/headtohead.py \
        --dataset beir/fiqa/test --limit-docs 10000 --limit-queries 100 \
        --output benchmark/results/headtohead-fiqa
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np

from colbert_config import MODEL_ID as COLBERT_MODEL_ID, cache_config
from data import load_slice, write_slice_manifest
from embeddings import cached_ragged
from env import load_env


def evaluate(qrels, run, k=10):
    ndcg = []
    recall = []
    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])[:k]
        gains = [relevant.get(doc, 0) for doc in ranked]
        dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
        ndcg.append(dcg / idcg if idcg else 0)
        wanted = {doc for doc, g in relevant.items() if g > 0}
        recall.append(len(wanted.intersection(ranked)) / len(wanted) if wanted else 0)
    return {
        "ndcg@10": float(np.mean(ndcg)),
        "recall@10": float(np.mean(recall)),
    }


def http(base, route, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + route, data, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def bench_annex_multivector(docs, queries, multi_docs, multi_queries, qrels, workspace_root, candidates=250):
    """Run against our own annex-multivector server (same path as benchmark/run.py)."""
    bin_path = workspace_root / "target/release/annex-multivector"
    plaid_path = Path("/tmp/headtohead-plaid")
    shutil.rmtree(plaid_path, ignore_errors=True)
    dim = multi_docs[0].shape[1]
    server = subprocess.Popen(
        [
            str(bin_path),
            "--dimension", str(dim),
            "--centroids", "256",
            "--probes", "8",
            "--path", str(plaid_path),
            "--listen", "127.0.0.1:18090",
        ],
        stdout=subprocess.DEVNULL,
    )
    base = "http://127.0.0.1:18090"
    try:
        for _ in range(300):
            try:
                http(base, "/healthz")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("annex-multivector server did not start")

        # Sample tokens for training
        lengths = np.asarray([len(d) for d in multi_docs], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        rng = np.random.default_rng(13)
        total = int(offsets[-1])
        n_samples = min(total, 256 * 50)
        chosen = rng.choice(total, n_samples, replace=False)
        which_doc = np.searchsorted(offsets[1:], chosen, side="right")
        samples = [
            np.asarray(multi_docs[d][t - offsets[d]], dtype=np.float32).tolist()
            for d, t in zip(which_doc, chosen)
        ]
        http(base, "/v1/train", {"vectors": samples, "iterations": 20})

        # Build
        build_t0 = time.perf_counter()
        for start in range(0, len(docs), 100):
            http(base, "/v1/vectors/upsert", {
                "documents": [
                    {"id": d.doc_id, "vectors": np.asarray(v).tolist()}
                    for d, v in zip(docs[start:start + 100], multi_docs[start:start + 100])
                ],
            })
        build_s = time.perf_counter() - build_t0

        # Query
        run = {}
        times = []
        for q, v in zip(queries, multi_queries):
            t0 = time.perf_counter()
            result = http(base, "/v1/query", {
                "vectors": np.asarray(v).tolist(),
                "top_k": 100,
                "candidates": candidates,
            })
            times.append(time.perf_counter() - t0)
            run[q.query_id] = [x["id"] for x in result["matches"]]
    finally:
        server.terminate()
        server.wait(timeout=30)

    scores = evaluate(qrels, run)
    return {
        **scores,
        "p50_ms": float(np.percentile(times, 50) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "p99_ms": float(np.percentile(times, 99) * 1000),
        "build_s": build_s,
    }


def bench_qdrant(docs, queries, multi_docs, multi_queries, qrels):
    """Qdrant with native multi-vector (MAX_SIM comparator, in-process)."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        MultiVectorComparator,
        MultiVectorConfig,
        PointStruct,
        VectorParams,
    )

    client = QdrantClient(":memory:")
    dim = int(multi_docs[0].shape[1])
    client.create_collection(
        collection_name="mv",
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
        ),
    )

    # Build
    build_t0 = time.perf_counter()
    for start in range(0, len(docs), 100):
        client.upsert(
            collection_name="mv",
            points=[
                PointStruct(
                    id=i,
                    vector=np.asarray(multi_docs[i], dtype=np.float32).tolist(),
                    payload={"doc_id": docs[i].doc_id},
                )
                for i in range(start, min(start + 100, len(docs)))
            ],
        )
    build_s = time.perf_counter() - build_t0

    # Query
    run = {}
    times = []
    for q, v in zip(queries, multi_queries):
        t0 = time.perf_counter()
        result = client.query_points(
            collection_name="mv",
            query=np.asarray(v, dtype=np.float32).tolist(),
            limit=100,
            with_payload=True,
        )
        times.append(time.perf_counter() - t0)
        run[q.query_id] = [p.payload["doc_id"] for p in result.points]

    scores = evaluate(qrels, run)
    return {
        **scores,
        "p50_ms": float(np.percentile(times, 50) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "p99_ms": float(np.percentile(times, 99) * 1000),
        "build_s": build_s,
    }


def bench_lancedb(docs, queries, multi_docs, multi_queries, qrels):
    """LanceDB — currently no native MaxSim; store single mean vector as fallback.

    We flag this row in the output because it is NOT doing late-interaction —
    it's a mean-pool dense-vector baseline through LanceDB. That's the closest
    apples-to-apples comparison we can get with LanceDB's current API for
    multi-vector data (as of 0.39.0). If they ship native MaxSim later, swap
    this for the real path.
    """
    import lancedb

    dim = int(multi_docs[0].shape[1])
    doc_means = np.stack([np.mean(np.asarray(v, dtype=np.float32), axis=0) for v in multi_docs])
    # L2-normalize for cosine
    norms = np.linalg.norm(doc_means, axis=1, keepdims=True).clip(min=1e-8)
    doc_means = doc_means / norms
    query_means = np.stack([np.mean(np.asarray(v, dtype=np.float32), axis=0) for v in multi_queries])
    qnorms = np.linalg.norm(query_means, axis=1, keepdims=True).clip(min=1e-8)
    query_means = query_means / qnorms

    db_path = Path("/tmp/headtohead-lancedb")
    shutil.rmtree(db_path, ignore_errors=True)
    db = lancedb.connect(str(db_path))
    build_t0 = time.perf_counter()
    data = [
        {"doc_id": docs[i].doc_id, "vector": doc_means[i].tolist()}
        for i in range(len(docs))
    ]
    table = db.create_table("mv", data=data)
    # Build an ANN index if enough rows
    if len(docs) >= 256:
        try:
            table.create_index(metric="cosine", vector_column_name="vector")
        except Exception as error:
            print(f"LanceDB index create failed: {error}")
    build_s = time.perf_counter() - build_t0

    run = {}
    times = []
    for q, qv in zip(queries, query_means):
        t0 = time.perf_counter()
        result = table.search(qv.tolist()).limit(100).to_list()
        times.append(time.perf_counter() - t0)
        run[q.query_id] = [row["doc_id"] for row in result]

    scores = evaluate(qrels, run)
    return {
        **scores,
        "p50_ms": float(np.percentile(times, 50) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "p99_ms": float(np.percentile(times, 99) * 1000),
        "build_s": build_s,
        "note": "LanceDB row is mean-pool dense baseline, NOT late-interaction MaxSim",
    }


def main():
    load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="beir/fiqa/test")
    p.add_argument("--output", type=Path, default=Path("benchmark/results/headtohead"))
    p.add_argument("--cache-dir", type=Path, default=Path("benchmark/cache"))
    p.add_argument("--limit-docs", type=int, default=10000)
    p.add_argument("--limit-queries", type=int, default=100)
    p.add_argument("--sampling", choices=["prefix", "qrels"], default="prefix")
    p.add_argument("--sample-seed", type=int, default=13)
    p.add_argument("--engines", default="annex,qdrant,lancedb")
    p.add_argument("--annex-candidates", type=int, default=250)
    p.add_argument("--annex-sweep", default="", help="comma-separated candidate counts for annex Pareto sweep")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    docs, queries, qrels = load_slice(
        args.dataset, args.limit_docs, args.limit_queries, args.sampling, args.sample_seed
    )
    write_slice_manifest(args.output / "slice.json", args.dataset, args.sampling, args.sample_seed, docs, queries)

    texts = [(getattr(d, "title", "") + " " + d.text).strip() for d in docs]
    multi_docs, _ = cached_ragged(
        args.cache_dir, COLBERT_MODEL_ID, "document",
        [d.doc_id for d in docs], texts,
        lambda: (_ for _ in ()).throw(RuntimeError("cache miss for docs")),
        False, cache_config("document"),
    )
    query_texts = [q.text for q in queries]
    multi_queries, _ = cached_ragged(
        args.cache_dir, COLBERT_MODEL_ID, "query",
        [q.query_id for q in queries], query_texts,
        lambda: (_ for _ in ()).throw(RuntimeError("cache miss for queries")),
        False, cache_config("query"),
    )

    workspace_root = Path(__file__).resolve().parents[3]
    engines = args.engines.split(",")
    results = {"dataset": args.dataset, "documents": len(docs), "queries": len(qrels), "systems": {}}

    if "annex" in engines:
        sweep = [int(x) for x in args.annex_sweep.split(",") if x] or [args.annex_candidates]
        for cand in sweep:
            key = f"annex_multivector_c{cand}" if len(sweep) > 1 else "annex_multivector"
            print(f"== {key} ==")
            results["systems"][key] = bench_annex_multivector(
                docs, queries, multi_docs, multi_queries, qrels, workspace_root, candidates=cand
            )
    if "qdrant" in engines:
        print("== qdrant ==")
        results["systems"]["qdrant"] = bench_qdrant(docs, queries, multi_docs, multi_queries, qrels)
    if "lancedb" in engines:
        print("== lancedb ==")
        results["systems"]["lancedb"] = bench_lancedb(docs, queries, multi_docs, multi_queries, qrels)

    out_path = args.output / "matrix.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print()
    print(f"wrote {out_path}")
    print()
    print(f"{'system':<24} {'nDCG@10':>8} {'R@10':>7} {'p50 ms':>8} {'p95 ms':>8} {'build s':>8}")
    print("-" * 72)
    for name, s in results["systems"].items():
        print(
            f"{name:<24} {s.get('ndcg@10', 0):>8.4f} {s.get('recall@10', 0):>7.4f} "
            f"{s.get('p50_ms', 0):>8.2f} {s.get('p95_ms', 0):>8.2f} {s.get('build_s', 0):>8.2f}"
        )
        if s.get("note"):
            print(f"    note: {s['note']}")


if __name__ == "__main__":
    main()
