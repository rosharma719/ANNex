"""Full-corpus dense/BM25/hybrid ablations through the real retrieval API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data import load_slice
from embeddings import cached_fixed, fixed_fingerprint
from measurement import Journal, summarize
from protocol import (
    add_protocol_arguments,
    environment_settings,
    file_digest,
    prepare_protocol,
    source_digest,
    validate_protocol_args,
)
from quality_engines import STRATEGIES, document_text, open_engine
from significance import paired_bootstrap

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATASETS = ["beir/nfcorpus/test", "beir/scifact/test", "beir/arguana"]


def embeddings(args, docs, queries):
    model = None

    def encode(texts):
        nonlocal model
        if not args.prepare:
            raise RuntimeError("run quality.py --prepare before evaluating or freezing")
        if model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(4)
            model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
        return model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )

    output = []
    for role, ids, texts in [
        (
            "document",
            [d.doc_id for d in docs],
            [(getattr(d, "title", "") + " " + d.text).strip() for d in docs],
        ),
        ("query", [q.query_id for q in queries], [q.text for q in queries]),
    ]:
        started = time.perf_counter()
        values, info = cached_fixed(
            args.cache_dir, MODEL, role, ids, texts, lambda: encode(texts), True
        )
        if info["model_revision"] != REVISION:
            raise RuntimeError(
                "cached checkpoint differs from the predeclared revision"
            )
        output.append((values, info))
        print(
            json.dumps(
                {
                    "stage": "embeddings",
                    "role": role,
                    "items": len(ids),
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default=DATASETS[0])
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("benchmark/cache/quality")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark/results/quality")
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--engines", default="annex,qdrant,lancedb")
    parser.add_argument("--qdrant-binary", type=Path)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument(
        "--binary", type=Path, default=Path("target/release/annex-multivector")
    )
    add_protocol_arguments(parser)
    args = parser.parse_args()
    args.sampling, args.sample_seed = "prefix", 13
    args.limit_docs = args.limit_queries = None
    if not args.prepare:
        validate_protocol_args(args, 1)
    docs, all_queries, all_qrels = load_slice(args.dataset)
    if args.prepare:
        embeddings(args, docs, all_queries)
        return
    if not 100 <= args.candidates <= 100_000:
        parser.error("candidates must be in 100..=100000")
    engines = args.engines.split(",")
    if len(set(engines)) != len(engines) or not set(engines) <= {
        "annex",
        "qdrant",
        "lancedb",
    }:
        parser.error("choose unique engines from annex,qdrant,lancedb")
    if "qdrant" in engines and args.qdrant_binary is None:
        parser.error("qdrant requires --qdrant-binary (native server)")
    fingerprints = [
        fixed_fingerprint(args.cache_dir, MODEL, role, ids, texts)
        for role, ids, texts in [
            ("document", [d.doc_id for d in docs], [document_text(d) for d in docs]),
            ("query", [q.query_id for q in all_queries], [q.text for q in all_queries]),
        ]
    ]
    workspace = Path(__file__).resolve().parents[3]
    settings = {
        "source_sha256": source_digest(workspace),
        "environment": environment_settings(),
        "model": MODEL,
        "revision": REVISION,
        "candidates": args.candidates,
        "rrf_k": 60,
        "quality_metrics": "trec_eval linear-gain nDCG@10, Recall@10/20/100, MRR@10",
        "bm25": {
            "k1": 1.2,
            "b": 0.75,
            "tokenizer": "unicode-alphanumeric-lowercase-v1",
        },
        "embedding_hashes": [f["files_sha256"] for f in fingerprints],
        "annex_binary_sha256": file_digest(args.binary),
        "qdrant_binary_sha256": file_digest(args.qdrant_binary)
        if "qdrant" in engines
        else None,
        "engines": engines,
        "strategies": STRATEGIES,
        "dense_search": "exact cosine for every engine; no quantization",
        "qdrant": (
            "native server, sparse explicit BM25(k1=1.2,b=.75), RRF k=61 "
            "zero-based, HTTP"
        ),
        "lancedb": (
            "embedded native FTS default English "
            "stemming/stopwords/ascii-folding/max-token-length40; native RRF "
            "K=60 one-based"
        ),
        "annex": "fsync, native lexical BM25 and RRF, HTTP",
        "query_order": (
            "dataset order, engine then dense/BM25/hybrid; no randomized latency trials"
        ),
        "ingest": (
            "ANNex/Qdrant batches100, LanceDB bulk Arrow; build costs are not "
            "comparable durability guarantees"
        ),
        "exclude_query_id": True,
        "warmup": 0,
        "encoding": "normalized MiniLM, max_seq_length=256, CPU, batch=32",
    }
    queries, qrels, protocol = prepare_protocol(
        args, docs, all_queries, all_qrels, settings, operating_points=1
    )
    if args.freeze_config:
        print(f"Frozen: {args.freeze_config}")
        return
    (vectors, doc_cache), (query_vectors, query_cache) = embeddings(
        args, docs, all_queries
    )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "dataset": args.dataset,
        "documents": len(docs),
        "qrels": qrels,
        "protocol": protocol,
        "embedding_cache": {"documents": doc_cache, "queries": query_cache},
        "binary_sha256": file_digest(args.binary),
        "measurement": (
            "serial client-visible retrieval; ANNex/Qdrant HTTP, LanceDB "
            "embedded; no warmup; encoding excluded; no engine speed ranking"
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    journal = Journal(args.output / "events.jsonl")
    selected = {q.query_id for q in queries}
    try:
        for engine in engines:
            systems = [f"{engine}_{strategy}" for strategy in STRATEGIES]
            for name in systems:
                journal.write("system_started", name)
            try:
                with open_engine(engine, args, docs, vectors) as backend:
                    with journal.stage("_preparation", f"{engine}_ingest"):
                        backend.build()
                    disk_bytes = sum(
                        p.stat().st_size
                        for p in (args.output / engine).rglob("*")
                        if p.is_file()
                    )
                    for strategy, name in zip(STRATEGIES, systems):
                        journal.write(
                            "index_ready",
                            name,
                            index={"disk_bytes_after_build": disk_bytes},
                        )
                        for i, q in enumerate(all_queries):
                            if q.query_id not in selected:
                                continue
                            journal.query(
                                name,
                                q.query_id,
                                strategy,
                                lambda q=q, i=i, strategy=strategy: backend.query(
                                    strategy, q, query_vectors[i], args.candidates
                                ),
                            )
                        journal.write("system_finished", name, status="complete")
                        print(
                            json.dumps({"system": name, "queries": len(queries)}),
                            flush=True,
                        )
            except Exception as error:
                for name in systems:
                    journal.write(
                        "system_finished",
                        name,
                        status="error",
                        error=f"{type(error).__name__}: {error}",
                    )
                print(json.dumps({"engine": engine, "error": str(error)}), flush=True)
    finally:
        journal.close()
    result = summarize(args.output)
    runs = {
        name: {r["qid"]: r["ranked_ids"] for r in system["per_query"]}
        for name, system in result["systems"].items()
    }
    # Descriptive paired intervals; no multiple-comparison-adjusted winner claim.
    result["paired_quality"] = (
        {
            baseline: paired_bootstrap(runs[baseline], runs["annex_hybrid_rrf"], qrels)
            for baseline in runs
            if baseline != "annex_hybrid_rrf"
        }
        if "annex_hybrid_rrf" in runs
        else {}
    )
    (args.output / "matrix.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                name: {
                    k: v
                    for k, v in system.items()
                    if k
                    in [
                        "ndcg@10",
                        "recall@10",
                        "p50_ms",
                        "p95_ms",
                        "p99_ms",
                        "failed_queries",
                    ]
                }
                for name, system in result["systems"].items()
            },
            indent=2,
        )
    )
    if any(s["failed_queries"] for s in result["systems"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
