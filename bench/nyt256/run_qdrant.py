#!/usr/bin/env python3
"""
Run a Qdrant sweep over the NYT-256-angular dataset against a real server.

Qdrant is a server, not an in-process library, so every timed query includes
loopback network + protobuf serialization. That overhead is unavoidable in the
only configuration Qdrant ships, but it means these numbers are NOT directly
comparable to the in-process hnswlib/usearch/faiss rows: they measure the
product, not just the index. Prefer gRPC (--prefer-grpc) to keep it minimal.

Usage:
    docker run -d -p 6333:6333 -p 6334:6334 -v $PWD/data/qdrant_storage:/qdrant/storage qdrant/qdrant
    python3 bench/nyt256/run_qdrant.py --out bench/nyt256/results_qdrant.jsonl

Requires: qdrant-client numpy
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_competitors import (  # noqa: E402
    K, N_QUERIES, ROUNDS, _time_rounds, emit, load_data, recall_at_k,
)


def build_collection(client, name, base, m, ef_construct, batch_size, force):
    from qdrant_client import models

    if force and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=base.shape[1], distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
        )
        vectors = base.tolist()
        t0 = time.perf_counter()
        for start in range(0, len(vectors), batch_size):
            chunk = vectors[start:start + batch_size]
            client.upsert(
                collection_name=name,
                points=models.Batch(
                    ids=list(range(start, start + len(chunk))),
                    vectors=chunk,
                ),
                wait=True,
            )
        build_seconds = time.perf_counter() - t0
        print(f"  upserted {len(vectors)} points in {build_seconds:.1f}s")
    else:
        build_seconds = None
        print(f"  collection '{name}' already present, reusing")

    # HNSW is built asynchronously; wait until every vector is indexed.
    while True:
        info = client.get_collection(name)
        indexed = getattr(info, "indexed_vectors_count", 0) or 0
        if indexed >= len(base):
            break
        print(f"  indexing... {indexed}/{len(base)}", flush=True)
        time.sleep(5)
    return build_seconds


def bench_qdrant(client, name, queries, truth, records, efs, config):
    from qdrant_client import models

    def search(q, ef):
        return client.query_points(
            collection_name=name,
            query=q,
            limit=K,
            search_params=models.SearchParams(hnsw_ef=ef, exact=False),
            with_payload=False,
            with_vectors=False,
        )

    for ef in efs:
        answers = [[p.id for p in search(q, ef).points] for q in queries]
        rec = recall_at_k(answers, truth)
        times = _time_rounds(lambda q: search(q, ef), queries)
        emit(records, "qdrant", config, ef, rec, times)
        print(f"  qdrant ef={ef} recall@{K}={rec:.5f}", flush=True)


def _update_manifest(client, m, ef_construct, efs) -> None:
    import datetime

    p = Path("bench/nyt256/manifest.json")
    manifest = json.loads(p.read_text()) if p.exists() else {}
    try:
        import importlib.metadata as im
        client_version = im.version("qdrant-client")
    except Exception:
        client_version = "?"
    try:
        server_version = client.get_version() or "?"
    except Exception:
        server_version = "?"
    manifest["qdrant"] = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "client": client_version,
        "server": server_version,
        "transport": "grpc" if getattr(client, "_prefer_grpc", False) else "rest",
        "m": m,
        "ef_construct": ef_construct,
        "ef_values": efs,
        "rounds": ROUNDS,
        "note": "server-side; per-query loopback network + serialization included",
    }
    p.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench/nyt256/results_qdrant.jsonl")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--grpc-port", type=int, default=6334)
    parser.add_argument("--rest", action="store_true", help="use REST instead of gRPC")
    parser.add_argument("--collection", default="nyt256-angular")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construct", type=int, default=300)
    parser.add_argument("--ef", default="32,64,128,256,512")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    efs = [int(x) for x in args.ef.split(",") if x.strip()]
    client = QdrantClient(
        host=args.host,
        port=args.port,
        grpc_port=args.grpc_port,
        prefer_grpc=not args.rest,
        timeout=60,
    )
    config = f"M={args.m} efc={args.ef_construct} {'grpc' if not args.rest else 'rest'}"

    print("Loading data...")
    base, queries, truth = load_data()
    queries_list = [q.tolist() for q in queries]

    print(f"Qdrant server {client.get_version()} at {args.host}, config: {config}")
    build_collection(client, args.collection, base, args.m, args.ef_construct,
                     args.batch_size, args.recreate)

    # Warm cache: one full untimed pass at ef=64, matching the other harnesses.
    from qdrant_client import models
    for q in queries_list:
        client.query_points(collection_name=args.collection, query=q, limit=K,
                            search_params=models.SearchParams(hnsw_ef=64, exact=False),
                            with_payload=False, with_vectors=False)

    records = []
    print(f"Running qdrant ef={efs} ({ROUNDS} rounds)...")
    bench_qdrant(client, args.collection, queries_list, truth, records, efs, config)

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    _update_manifest(client, args.m, args.ef_construct, args.ef)
    print(f"\nResults written to {args.out}")
    print(f"{len(records)} data points")
