"""Deterministic holdouts and immutable operating-point files (stdlib only).

This prevents accidental tuning sweeps on test queries. It cannot prove that a
human has never inspected those queries in an earlier experiment.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import json
from datetime import datetime, timezone
from pathlib import Path


def add_protocol_arguments(parser):
    parser.add_argument("--partition", choices=["dev", "test", "exploratory"], default="dev")
    parser.add_argument("--split-seed", type=int, default=20260924)
    parser.add_argument("--freeze-config", type=Path,
                        help="write one selected dev operating point and exit without evaluation")
    parser.add_argument("--frozen-config", type=Path,
                        help="required for test; all settings and corpus content must match")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def slice_digest(docs, queries, qrels):
    # Include actual content/judgments, not only IDs: changed qrels or embeddings'
    # source texts must never silently masquerade as the same held-out benchmark.
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda x: x.doc_id):
        h.update(bytes.fromhex(_digest(["doc", d.doc_id, getattr(d, "title", ""), d.text])))
    for q in sorted(queries, key=lambda x: x.query_id):
        h.update(bytes.fromhex(_digest(["query", q.query_id, q.text, qrels.get(q.query_id, {})])))
    return h.hexdigest()


def query_split(queries, seed):
    ids = [q.query_id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate query IDs in benchmark slice")
    ordered = sorted(ids, key=lambda qid: (_digest([seed, qid]), qid))
    cut = len(ordered) // 2
    return sorted(ordered[:cut]), sorted(ordered[cut:])


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_digest(workspace):
    h = hashlib.sha256()
    paths = sorted((workspace / "crates").glob("*/src/**/*.rs"))
    paths += sorted((workspace / "crates").glob("*/Cargo.toml"))
    paths += sorted((workspace / "crates/annex-multivector/benchmark").glob("*.py"))
    paths += [workspace / "Cargo.lock", workspace / "Cargo.toml"]
    for path in paths:
        h.update(str(path.relative_to(workspace)).encode() + b"\0")
        h.update(bytes.fromhex(file_digest(path)))
    return h.hexdigest()


def environment_settings():
    packages = {}
    for name in ("numpy", "ir-datasets", "pylate", "torch", "qdrant-client", "lancedb"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"packages": packages, "python": platform.python_version(),
            "machine": platform.machine(),
            "threads": {k: os.environ.get(k) for k in
                        ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "rustflags": os.environ.get("RUSTFLAGS")}


def prepare_protocol(args, docs, queries, qrels, settings, *, operating_points):
    """Validate before loading models, starting servers, or submitting queries."""
    if not docs or not queries:
        raise ValueError("benchmark slice must contain documents and evaluable queries")
    if len({d.doc_id for d in docs}) != len(docs):
        raise ValueError("duplicate document IDs in benchmark slice")
    if set(qrels) != {q.query_id for q in queries}:
        raise ValueError("every benchmark query must have retained qrels")
    if args.freeze_config and args.frozen_config:
        raise ValueError("choose --freeze-config or --frozen-config")
    if args.partition == "test" and not args.frozen_config:
        raise ValueError("test requires --frozen-config; tune only on dev")
    if args.partition == "test" and operating_points != 1:
        raise ValueError("test evaluation cannot sweep operating points")
    if args.freeze_config and (args.partition != "dev" or operating_points != 1):
        raise ValueError("freeze exactly one operating point from dev")
    if args.frozen_config and args.partition != "test":
        raise ValueError("--frozen-config is used only with --partition test")
    dev_ids, test_ids = query_split(queries, args.split_seed)
    if args.partition != "exploratory" and (not dev_ids or not test_ids):
        raise ValueError("need at least two evaluable queries for disjoint dev/test")
    settings = json.loads(json.dumps(settings, allow_nan=False))
    contract = {
        "protocol_version": 1,
        "split_seed": args.split_seed,
        "slice": {"dataset": args.dataset, "sampling": args.sampling,
                  "sample_seed": args.sample_seed, "limit_docs": args.limit_docs,
                  "limit_queries": args.limit_queries, "documents": len(docs),
                  "evaluable_queries_before_split": len(queries),
                  "content_sha256": slice_digest(docs, queries, qrels)},
        "dev_query_ids": dev_ids, "test_query_ids": test_ids,
        "settings": settings,
    }
    contract_hash = _digest(contract)
    if args.freeze_config:
        artifact = {"created_at": datetime.now(timezone.utc).isoformat(),
                    "contract_sha256": contract_hash, "contract": contract}
        args.freeze_config.parent.mkdir(parents=True, exist_ok=True)
        # Never replace a previously frozen artifact.
        with args.freeze_config.open("x", encoding="utf-8") as f:
            json.dump(artifact, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
    if args.frozen_config:
        artifact = json.loads(args.frozen_config.read_text())
        if artifact.get("contract_sha256") != _digest(artifact.get("contract")):
            raise ValueError("frozen configuration digest mismatch")
        if artifact["contract"] != contract:
            raise ValueError("test settings, source, dataset content, or split differ from frozen dev configuration")
    ids = set(dev_ids if args.partition == "dev" else test_ids)
    selected = list(queries) if args.partition == "exploratory" else [q for q in queries if q.query_id in ids]
    selected_qrels = {q.query_id: qrels[q.query_id] for q in selected}
    return selected, selected_qrels, {
        **contract, "partition": args.partition, "contract_sha256": contract_hash,
        "evaluated_query_ids": [q.query_id for q in selected],
        "frozen_config": str(args.frozen_config) if args.frozen_config else None,
        "label": "held-out frozen-config test" if args.partition == "test" else "development exploration",
    }
