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
import sys
import time
import urllib.request
from pathlib import Path

# Make sibling modules (protocol, data, embeddings, ...) importable regardless
# of how this file is loaded — direct script, `python -m benchmark.headtohead`,
# or `importlib.util.spec_from_file_location` from a test harness. Without
# this, the bare `from protocol import ...` below only resolves in script
# mode from the benchmark/ working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from colbert_config import MODEL_ID as COLBERT_MODEL_ID, cache_config
from data import load_slice, write_slice_manifest
from embeddings import cached_ragged
from env import load_env
from protocol import add_protocol_arguments, prepare_protocol, source_digest, environment_settings


def evaluate(qrels, run, k=10):
    """Return aggregate {nDCG@10, R@10} plus per-query series so callers can
    plot tail distributions or feed a recall-difficulty analysis."""
    ndcg = []
    recall = []
    per_query = []
    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])[:k]
        gains = [relevant.get(doc, 0) for doc in ranked]
        dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
        q_ndcg = float(dcg / idcg) if idcg else 0.0
        wanted = {doc for doc, g in relevant.items() if g > 0}
        q_recall = float(len(wanted.intersection(ranked)) / len(wanted)) if wanted else 0.0
        ndcg.append(q_ndcg)
        recall.append(q_recall)
        # ranked_ids: keep the top-K returned IDs so offline analyses can
        # compute rank disagreement between two candidate levels (a proxy
        # for candidate-pool stability / query difficulty).
        per_query.append({
            "qid": qid,
            "ndcg@10": q_ndcg,
            "recall@10": q_recall,
            "ranked_ids": list(ranked),
        })
    return {
        "ndcg@10": float(np.mean(ndcg)),
        "recall@10": float(np.mean(recall)),
        # p10 / p01 of per-query nDCG — reframes benchmarks on tail quality
        # rather than just the mean. A high mean can hide a fat left tail.
        "ndcg@10_p10": float(np.percentile(ndcg, 10)),
        "ndcg@10_p01": float(np.percentile(ndcg, 1)),
        "recall@10_p10": float(np.percentile(recall, 10)),
        "recall@10_p01": float(np.percentile(recall, 1)),
        "queries_meeting_ndcg_0_5": float(np.mean([1 if n >= 0.5 else 0 for n in ndcg])),
        "queries_meeting_recall_0_8": float(np.mean([1 if r >= 0.8 else 0 for r in recall])),
        "per_query": per_query,
    }


def merge_latency(scores, qids, times_s, per_query_extra=None):
    """Merge per-query latency + optional extra features (top score, top-vs-2nd
    margin) into scores['per_query']. Keeps aggregate percentiles too."""
    times_ms = [t * 1000.0 for t in times_s]
    by_qid = {q["qid"]: q for q in scores["per_query"]}
    for qid, t_ms in zip(qids, times_ms):
        entry = by_qid.get(qid)
        if entry is not None:
            entry["latency_ms"] = float(t_ms)
    if per_query_extra:
        for qid, extra in per_query_extra.items():
            entry = by_qid.get(qid)
            if entry is not None:
                entry.update(extra)
    scores["p50_ms"] = float(np.percentile(times_ms, 50))
    scores["p95_ms"] = float(np.percentile(times_ms, 95))
    scores["p99_ms"] = float(np.percentile(times_ms, 99))
    return scores


def http(base, route, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + route, data, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def bench_annex_multivector(docs, queries, multi_docs, multi_queries, qrels, workspace_root, candidates=250, durability="fsync"):
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
            "--durability", durability,
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

        # Query. Record per-query extras: top hit score and top-vs-2nd margin.
        # These are the two cheapest per-query difficulty signals — small
        # margin means the top candidate is close to the runners-up, which
        # correlates with "this query is hard, we might need more candidates".
        run = {}
        times = []
        extras = {}
        qids = []
        for q, v in zip(queries, multi_queries):
            t0 = time.perf_counter()
            result = http(base, "/v1/query", {
                "vectors": np.asarray(v).tolist(),
                "top_k": 100,
                "candidates": candidates,
                "candidate_backend": "muvera",
            })
            times.append(time.perf_counter() - t0)
            qids.append(q.query_id)
            matches = result["matches"]
            run[q.query_id] = [x["id"] for x in matches]
            top_score = float(matches[0]["score"]) if matches else 0.0
            second_score = float(matches[1]["score"]) if len(matches) > 1 else top_score
            # FDE-vs-MaxSim rank disagreement signals. Each hit carries the
            # original FDE score (fde_score field). We compute:
            #  - fde_top_score: FDE score of the #1-by-MaxSim hit
            #  - fde_top_rank_in_fde: 0-indexed position of the #1-by-MaxSim
            #    hit when hits are sorted by FDE score descending. High value
            #    = MaxSim ranked something the FDE stage buried, which is
            #    exactly the "candidate pool was unstable / borderline" case.
            #  - fde_maxsim_agreement: fraction of returned hits whose FDE
            #    rank and MaxSim rank differ by <= 3 (a robust "did the two
            #    stages agree on the head" score).
            fde_scores = [float(m.get("fde_score", 0.0)) for m in matches]
            if len(matches) > 1:
                # positions sorted by FDE score descending
                fde_order = sorted(range(len(matches)), key=lambda i: -fde_scores[i])
                # 0-indexed rank of the #1-by-MaxSim hit inside that ordering
                fde_top_rank = fde_order.index(0)
                # per-hit rank differences
                fde_rank_by_pos = {p: r for r, p in enumerate(fde_order)}
                agree = sum(
                    1 for i in range(len(matches)) if abs(fde_rank_by_pos[i] - i) <= 3
                )
                agreement = agree / len(matches)
            else:
                fde_top_rank = 0
                agreement = 1.0
            extras[q.query_id] = {
                "top_score": top_score,
                "top_minus_second": top_score - second_score,
                "returned": len(matches),
                "fde_top_score": fde_scores[0] if fde_scores else 0.0,
                "fde_top_rank_in_fde": fde_top_rank,
                "fde_maxsim_agreement": agreement,
            }
    finally:
        server.terminate()
        server.wait(timeout=30)

    scores = evaluate(qrels, run)
    scores = merge_latency(scores, qids, times, extras)
    scores["build_s"] = build_s
    return scores


def bench_qdrant(docs, queries, multi_docs, multi_queries, qrels, server_url=None):
    """Qdrant with native multi-vector (MAX_SIM comparator).

    If `server_url` is provided (e.g. http://127.0.0.1:6333), talks to a real
    running Qdrant Server — the production configuration a user would deploy.
    That's the honest comparator per BENCHMARK_POLICY.md.

    If `server_url` is None, falls back to the qdrant-client local (":memory:")
    mode. That mode reuses much of the Rust core in-process but is NOT a
    substitute for the server on the wire: no protocol serialisation, no
    connection pooling, no rocksdb payload store, no WAL, no snapshot layer.
    We keep the fallback path so anyone can run this benchmark without
    Docker, but the result should be labelled 'qdrant-client local' — never
    just 'Qdrant'. See BENCHMARK_POLICY.md.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        MultiVectorComparator,
        MultiVectorConfig,
        PointStruct,
        VectorParams,
    )

    if server_url is None:
        client = QdrantClient(":memory:")
    else:
        client = QdrantClient(url=server_url)
        # Drop any prior collection with the same name so runs are idempotent.
        try:
            client.delete_collection(collection_name="mv")
        except Exception:
            pass
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
    qids = []
    extras = {}
    for q, v in zip(queries, multi_queries):
        t0 = time.perf_counter()
        result = client.query_points(
            collection_name="mv",
            query=np.asarray(v, dtype=np.float32).tolist(),
            limit=100,
            with_payload=True,
        )
        times.append(time.perf_counter() - t0)
        qids.append(q.query_id)
        pts = result.points
        run[q.query_id] = [p.payload["doc_id"] for p in pts]
        top_score = float(pts[0].score) if pts else 0.0
        second_score = float(pts[1].score) if len(pts) > 1 else top_score
        extras[q.query_id] = {
            "top_score": top_score,
            "top_minus_second": top_score - second_score,
            "returned": len(pts),
        }

    scores = evaluate(qrels, run)
    scores = merge_latency(scores, qids, times, extras)
    scores["build_s"] = build_s
    return scores


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
    qids = []
    for q, qv in zip(queries, query_means):
        t0 = time.perf_counter()
        result = table.search(qv.tolist()).limit(100).to_list()
        times.append(time.perf_counter() - t0)
        qids.append(q.query_id)
        run[q.query_id] = [row["doc_id"] for row in result]

    scores = evaluate(qrels, run)
    scores = merge_latency(scores, qids, times)
    scores["build_s"] = build_s
    scores["note"] = "LanceDB row is mean-pool dense baseline, NOT late-interaction MaxSim"
    return scores


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
    p.add_argument(
        "--engines",
        default="annex,qdrant_local,lancedb",
        help="comma-separated: annex, qdrant_local, qdrant_server, lancedb",
    )
    p.add_argument(
        "--qdrant-server",
        default=None,
        help="URL of a running Qdrant Server (e.g. http://127.0.0.1:6333). "
        "Use a pinned Qdrant image; server version is recorded in the frozen configuration.",
    )
    p.add_argument("--annex-candidates", type=int, default=250)
    p.add_argument("--annex-sweep", default="", help="comma-separated candidate counts for annex Pareto sweep")
    p.add_argument("--durability", choices=["fsync", "buffered"], default="fsync")
    add_protocol_arguments(p)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    docs, queries, qrels = load_slice(
        args.dataset, args.limit_docs, args.limit_queries, args.sampling, args.sample_seed
    )
    workspace_root = Path(__file__).resolve().parents[3]
    engines = sorted(set(args.engines.split(",")))
    if set(engines) - {"annex", "qdrant", "qdrant_local", "qdrant_server", "lancedb"}:
        p.error("unknown benchmark engine")
    sweep = [int(x) for x in args.annex_sweep.split(",") if x] or [args.annex_candidates]
    if any(n <= 0 for n in sweep):
        p.error("candidate counts must be positive")
    all_queries = queries
    settings = {
        "environment": environment_settings(),
        "harness": "headtohead", "source_sha256": source_digest(workspace_root),
        "engines": engines, "annex_candidates": sweep, "durability": args.durability,
        "annex": {"centroids": 256, "residual_bits": 2, "probes": 8,
                  "fde_repetitions": 20, "fde_ksim": 4, "fde_projected": 8,
                  "train_iterations": 20, "train_seed": 13},
        "model": COLBERT_MODEL_ID, "top_k": 100,
        "query_encoding": cache_config("query"), "document_encoding": cache_config("document"),
    }
    if "qdrant_server" in engines:
        if not args.qdrant_server:
            p.error("--qdrant-server URL required for qdrant_server engine")
        settings["qdrant_server_info"] = http(args.qdrant_server.rstrip("/"), "/")
    queries, qrels, protocol = prepare_protocol(
        args, docs, queries, qrels, settings, operating_points=len(sweep),
    )
    if args.freeze_config:
        print(f"Frozen operating point: {args.freeze_config}")
        return
    if (args.output / "matrix.json").exists() or (args.output / "slice.json").exists():
        p.error("output already contains a run; use a new output directory to retain prior results")
    write_slice_manifest(args.output / "slice.json", args.dataset, args.sampling, args.sample_seed, docs, queries)

    texts = [(getattr(d, "title", "") + " " + d.text).strip() for d in docs]
    multi_docs, _ = cached_ragged(
        args.cache_dir, COLBERT_MODEL_ID, "document",
        [d.doc_id for d in docs], texts,
        lambda: (_ for _ in ()).throw(RuntimeError("cache miss for docs")),
        False, cache_config("document"),
    )
    query_texts = [q.text for q in all_queries]
    multi_queries, _ = cached_ragged(
        args.cache_dir, COLBERT_MODEL_ID, "query",
        [q.query_id for q in all_queries], query_texts,
        lambda: (_ for _ in ()).throw(RuntimeError("cache miss for queries")),
        False, cache_config("query"),
    )

    if len(multi_docs) != len(docs) or len(multi_queries) != len(all_queries):
        raise ValueError("embedding cache length does not match document/query IDs")
    selected_ids = {q.query_id for q in queries}
    multi_queries = [v for q, v in zip(all_queries, multi_queries) if q.query_id in selected_ids]
    results = {"protocol": protocol, "dataset": args.dataset, "documents": len(docs), "queries": len(qrels), "systems": {}}

    if "annex" in engines:
        subprocess.run(["cargo", "build", "--release", "-p", "annex-multivector", "--bin", "annex-multivector"],
                       cwd=workspace_root, check=True)
        for cand in sweep:
            key = f"annex_multivector_c{cand}" if len(sweep) > 1 else "annex_multivector"
            print(f"== {key} ==")
            results["systems"][key] = bench_annex_multivector(
                docs, queries, multi_docs, multi_queries, qrels, workspace_root, candidates=cand, durability=args.durability
            )
    if "qdrant_local" in engines or "qdrant" in engines:
        # "qdrant" kept as an alias for back-compat; explicit new name is
        # 'qdrant_local' so the label makes the mode obvious in matrices.
        print("== qdrant_client_local (:memory: — NOT the production server) ==")
        results["systems"]["qdrant_client_local"] = bench_qdrant(
            docs, queries, multi_docs, multi_queries, qrels, server_url=None,
        )
    if "qdrant_server" in engines:
        if not args.qdrant_server:
            raise SystemExit(
                "--qdrant-server URL required for qdrant_server engine. "
                "Use a pinned Qdrant image and provide its URL."
            )
        print(f"== qdrant_server ({args.qdrant_server}) ==")
        results["systems"]["qdrant_server"] = bench_qdrant(
            docs, queries, multi_docs, multi_queries, qrels, server_url=args.qdrant_server,
        )
    if "lancedb" in engines:
        print("== lancedb ==")
        results["systems"]["lancedb"] = bench_lancedb(docs, queries, multi_docs, multi_queries, qrels)

    out_path = args.output / "matrix.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print()
    print(f"wrote {out_path}")
    print()
    print(
        f"{'system':<24} {'nDCG@10':>8} {'p10-nDCG':>9} {'R@10':>7} "
        f"{'>=0.5':>6} {'p50 ms':>7} {'p95 ms':>7} {'build s':>8}"
    )
    print("-" * 82)
    for name, s in results["systems"].items():
        print(
            f"{name:<24} {s.get('ndcg@10', 0):>8.4f} {s.get('ndcg@10_p10', 0):>9.4f} "
            f"{s.get('recall@10', 0):>7.4f} {s.get('queries_meeting_ndcg_0_5', 0):>6.1%} "
            f"{s.get('p50_ms', 0):>7.2f} {s.get('p95_ms', 0):>7.2f} {s.get('build_s', 0):>8.2f}"
        )
        if s.get("note"):
            print(f"    note: {s['note']}")


if __name__ == "__main__":
    main()
