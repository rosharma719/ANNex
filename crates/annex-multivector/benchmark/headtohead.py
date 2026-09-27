#!/usr/bin/env python3
"""Compare explicit retrieval configurations; retain every query outcome."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from colbert_config import MODEL_ID, cache_config
from data import load_slice, write_slice_manifest
from embeddings import cached_ragged, ragged_fingerprint
from env import load_env
from measurement import Journal, summarize
from protocol import (
    add_protocol_arguments,
    environment_settings,
    file_digest,
    prepare_protocol,
    source_digest,
    validate_protocol_args,
)
from server import annex_server, http


def bench_annex(
    docs,
    queries,
    vectors,
    query_vectors,
    journal,
    name,
    binary,
    directory,
    backend,
    candidates,
    *,
    centroids=256,
    m=16,
    ef_construct=256,
    ef_search=256,
    durability="fsync",
    timeout=600,
):
    with annex_server(
        binary,
        directory,
        vectors[0].shape[1],
        centroids=centroids,
        durability=durability,
    ) as base:
        with journal.stage(name, "train"):
            lengths = np.asarray([len(v) for v in vectors], dtype=np.int64)
            offsets = np.concatenate(([0], np.cumsum(lengths)))
            rng = np.random.default_rng(13)
            selected = rng.choice(
                int(offsets[-1]), min(int(offsets[-1]), centroids * 50), replace=False
            )
            indices = np.searchsorted(offsets[1:], selected, side="right")
            samples = [
                np.asarray(vectors[d][t - offsets[d]], dtype=np.float32).tolist()
                for d, t in zip(indices, selected)
            ]
            http(base, "/v1/train", {"vectors": samples, "iterations": 20}, timeout)
        with journal.stage(name, "ingest"):
            for start in range(0, len(docs), 100):
                http(
                    base,
                    "/v1/vectors/upsert",
                    {
                        "documents": [
                            {"id": d.doc_id, "vectors": np.asarray(v).tolist()}
                            for d, v in zip(
                                docs[start : start + 100], vectors[start : start + 100]
                            )
                        ]
                    },
                    timeout,
                )
        with journal.stage(name, "index_ready"):
            if backend == "hnsw":
                http(
                    base,
                    "/v1/fde/index",
                    {"m": m, "ef_construct": ef_construct},
                    timeout,
                )
            stats = http(base, "/v1/stats", timeout=timeout)
            if stats["documents"] != len(docs) or (
                backend == "hnsw" and stats["fde_ann_nodes"] != len(docs)
            ):
                raise RuntimeError(
                    "ANNex indexed document count does not match the input"
                )
        journal.write("index_ready", name, index=stats)
        for query, vector in zip(queries, query_vectors):
            body = {
                "vectors": np.asarray(vector).tolist(),
                "top_k": 100,
                "candidates": candidates,
                "candidate_backend": backend,
                "explain": True,
            }
            if backend == "hnsw":
                body["ef_search"] = ef_search

            def call(body=body):
                result = http(base, "/v1/query", body, timeout)
                actual = result["stats"]["candidate_backend"]
                if actual != backend:
                    raise RuntimeError(f"requested {backend}, executed {actual}")
                return {"matches": result["matches"], "backend": actual}

            journal.query(name, query.query_id, backend, call)


def bench_qdrant(
    docs, queries, vectors, query_vectors, journal, name, server_url=None, timeout=600
):
    """Exact MAX_SIM reference. Server and client-local modes remain separate."""
    from qdrant_client import QdrantClient, models

    client = (
        QdrantClient(url=server_url, timeout=timeout)
        if server_url
        else QdrantClient(":memory:")
    )
    collection = "annex_bench_" + uuid.uuid4().hex
    created = False
    try:
        with journal.stage(name, "ingest"):
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=int(vectors[0].shape[1]),
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=0),
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            )
            created = True
            for start in range(0, len(docs), 100):
                client.upsert(
                    collection_name=collection,
                    wait=True,
                    points=[
                        models.PointStruct(
                            id=i,
                            vector=np.asarray(vectors[i]).tolist(),
                            payload={"doc_id": docs[i].doc_id},
                        )
                        for i in range(start, min(start + 100, len(docs)))
                    ],
                )
        with journal.stage(name, "index_ready"):
            deadline = time.monotonic() + timeout
            while True:
                info = client.get_collection(collection)
                if (
                    info.points_count == len(docs)
                    and info.status == models.CollectionStatus.GREEN
                ):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Qdrant did not reach the expected point count and green status"
                    )
                time.sleep(0.1)
        journal.write("index_ready", name, index=info.model_dump(mode="json"))
        for query, vector in zip(queries, query_vectors):

            def call(vector=vector):
                result = client.query_points(
                    collection_name=collection,
                    query=np.asarray(vector).tolist(),
                    search_params=models.SearchParams(exact=True)
                    if server_url
                    else None,
                    limit=100,
                    with_payload=["doc_id"],
                    with_vectors=False,
                )
                return {
                    "backend": "exact_maxsim",
                    "matches": [
                        {"id": p.payload["doc_id"], "score": p.score}
                        for p in result.points
                    ],
                }

            journal.query(name, query.query_id, "exact_maxsim", call)
    finally:
        try:
            if created:
                client.delete_collection(collection)
        finally:
            client.close()


def bench_lancedb(docs, queries, vectors, query_vectors, journal, name, directory):
    """Legacy mean-pool ablation; not a native multivector comparison."""
    import lancedb

    def pooled(values):
        matrix = np.stack(
            [np.mean(np.asarray(v, dtype=np.float32), axis=0) for v in values]
        )
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)

    with journal.stage(name, "pool_and_ingest"):
        doc_means = pooled(vectors)
        db = lancedb.connect(str(directory))
        table = db.create_table(
            "documents",
            data=[
                {"doc_id": d.doc_id, "vector": v.tolist()}
                for d, v in zip(docs, doc_means)
            ],
        )
    for query, vector in zip(queries, query_vectors):

        def call(vector=vector):
            # Pooling is part of this ablation's query cost.
            results = (
                table.search(pooled([vector])[0].tolist())
                .distance_type("cosine")
                .limit(100)
                .to_list()
            )
            return {
                "backend": "exact_mean_pool_cosine",
                "matches": [
                    {"id": row["doc_id"], "score": 1.0 - row["_distance"]}
                    for row in results
                ],
            }

        journal.query(name, query.query_id, "exact_mean_pool_cosine", call)


def main():
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="beir/fiqa/test")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark/results/headtohead")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("benchmark/cache"))
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--sampling", choices=["prefix", "qrels"], default="prefix")
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument(
        "--engines",
        default="annex_exact,annex_hnsw",
        help="annex_exact,annex_hnsw,qdrant_local,qdrant_server,lancedb_mean_pool",
    )
    parser.add_argument("--qdrant-server")
    parser.add_argument("--annex-candidates", type=int, default=250)
    parser.add_argument("--annex-sweep", default="")
    parser.add_argument("--centroids", type=int, default=256)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construct", type=int, default=256)
    parser.add_argument("--ef-search", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--durability", choices=["fsync", "buffered"], default="fsync")
    add_protocol_arguments(parser)
    args = parser.parse_args()
    engines = sorted(
        set("annex_hnsw" if x == "annex" else x for x in args.engines.split(","))
    )
    if not engines or set(engines) - {
        "annex_exact",
        "annex_hnsw",
        "qdrant_local",
        "qdrant_server",
        "lancedb_mean_pool",
    }:
        parser.error("unknown engine; use explicit execution-mode names from --help")
    sweep = [int(x) for x in args.annex_sweep.split(",") if x] or [
        args.annex_candidates
    ]
    if len(sweep) != len(set(sweep)) or any(n <= 0 for n in sweep) or args.timeout <= 0:
        parser.error(
            "candidate counts must be unique and positive; timeout must be positive"
        )
    validate_protocol_args(args, len(sweep))
    docs, all_queries, all_qrels = load_slice(
        args.dataset,
        args.limit_docs,
        args.limit_queries,
        args.sampling,
        args.sample_seed,
    )
    workspace = Path(__file__).resolve().parents[3]
    binary = workspace / "target/release/annex-multivector"
    settings = {
        "environment": environment_settings(),
        "harness": "headtohead",
        "source_sha256": source_digest(workspace),
        "engines": engines,
        "candidates": sweep,
        "durability": args.durability,
        "centroids": args.centroids,
        "hnsw_m": args.hnsw_m,
        "ef_construct": args.hnsw_ef_construct,
        "ef_search": args.ef_search,
        "timeout_s": args.timeout,
        "model": MODEL_ID,
        "top_k": 100,
        "train_iterations": 20,
        "train_seed": 13,
        "query_encoding": cache_config("query"),
        "document_encoding": cache_config("document"),
    }
    texts = [(getattr(d, "title", "") + " " + d.text).strip() for d in docs]
    settings["embeddings"] = {
        "documents": ragged_fingerprint(
            args.cache_dir,
            MODEL_ID,
            "document",
            [d.doc_id for d in docs],
            texts,
            cache_config("document"),
        ),
        "queries": ragged_fingerprint(
            args.cache_dir,
            MODEL_ID,
            "query",
            [q.query_id for q in all_queries],
            [q.text for q in all_queries],
            cache_config("query"),
        ),
    }
    if "qdrant_server" in engines:
        if not args.qdrant_server:
            parser.error("--qdrant-server URL required")
        settings["qdrant_server_info"] = http(args.qdrant_server.rstrip("/"), "/")
    queries, qrels, protocol = prepare_protocol(
        args, docs, all_queries, all_qrels, settings, operating_points=len(sweep)
    )
    if args.freeze_config:
        print(f"Frozen operating point: {args.freeze_config}")
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be empty; prior runs are never overwritten")
    texts = [(getattr(d, "title", "") + " " + d.text).strip() for d in docs]

    def cache_miss():
        raise RuntimeError("embedding cache miss; run cache_embeddings.py first")

    vectors, doc_cache = cached_ragged(
        args.cache_dir,
        MODEL_ID,
        "document",
        [d.doc_id for d in docs],
        texts,
        cache_miss,
        False,
        cache_config("document"),
    )
    query_vectors, query_cache = cached_ragged(
        args.cache_dir,
        MODEL_ID,
        "query",
        [q.query_id for q in all_queries],
        [q.text for q in all_queries],
        cache_miss,
        False,
        cache_config("query"),
    )
    if len(vectors) != len(docs) or len(query_vectors) != len(all_queries):
        raise ValueError("embedding count does not match input IDs")
    selected = {q.query_id for q in queries}
    query_vectors = [
        v for q, v in zip(all_queries, query_vectors) if q.query_id in selected
    ]
    if any(engine.startswith("annex_") for engine in engines):
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "annex-multivector",
                "--bin",
                "annex-multivector",
            ],
            cwd=workspace,
            check=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "dataset": args.dataset,
        "documents": len(docs),
        "qrels": qrels,
        "protocol": protocol,
        "embedding_cache": {"documents": doc_cache, "queries": query_cache},
        "binary_sha256": file_digest(binary)
        if any(e.startswith("annex_") for e in engines)
        else None,
        "measurement": "serial HTTP/client elapsed time; no warmup; encoding and compilation excluded",
    }
    with (args.output / "manifest.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.write("\n")
    write_slice_manifest(
        args.output / "slice.json",
        args.dataset,
        args.sampling,
        args.sample_seed,
        docs,
        queries,
        qrels,
    )
    journal = Journal(args.output / "events.jsonl")
    try:
        for engine in engines:
            for count in sweep if engine.startswith("annex_") else [None]:
                name = (
                    f"{engine}_c{count}"
                    if count
                    else {
                        "qdrant_local": "qdrant_client_local_exact",
                        "qdrant_server": "qdrant_server_exact",
                    }.get(engine, engine)
                )
                journal.write("system_started", name)
                try:
                    if engine.startswith("annex_"):
                        bench_annex(
                            docs,
                            queries,
                            vectors,
                            query_vectors,
                            journal,
                            name,
                            binary,
                            args.output / name,
                            "hnsw" if engine == "annex_hnsw" else "muvera",
                            count,
                            centroids=args.centroids,
                            m=args.hnsw_m,
                            ef_construct=args.hnsw_ef_construct,
                            ef_search=args.ef_search,
                            durability=args.durability,
                            timeout=args.timeout,
                        )
                    elif engine.startswith("qdrant"):
                        bench_qdrant(
                            docs,
                            queries,
                            vectors,
                            query_vectors,
                            journal,
                            name,
                            args.qdrant_server if engine == "qdrant_server" else None,
                            args.timeout,
                        )
                    else:
                        bench_lancedb(
                            docs,
                            queries,
                            vectors,
                            query_vectors,
                            journal,
                            name,
                            args.output / name,
                        )
                    journal.write("system_finished", name, status="complete")
                except Exception as error:
                    journal.write(
                        "system_finished",
                        name,
                        status="error",
                        error=f"{type(error).__name__}: {error}",
                    )
                result = summarize(args.output)
                (args.output / "matrix.json").write_text(
                    json.dumps(result, indent=2, allow_nan=False) + "\n"
                )
                print(
                    f"{name}: {result['systems'][name]['status']}, {result['systems'][name]['failed_queries']} failed queries"
                )
    finally:
        journal.close()
    print(f"Results: {args.output}")
    if any(
        s["status"] != "complete" or s["failed_queries"]
        for s in result["systems"].values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
