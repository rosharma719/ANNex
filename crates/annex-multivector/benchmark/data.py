"""Deterministic BEIR benchmark slices without synthetic relevance labels."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import ir_datasets


def _negative_key(seed: int, doc_id: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{doc_id}".encode(), digest_size=8).digest()


def load_slice(dataset_name, limit_docs=None, limit_queries=None, sampling="prefix", seed=13):
    dataset = ir_datasets.load(dataset_name)
    queries = list(dataset.queries_iter())[:limit_queries]
    query_ids = {query.query_id for query in queries}
    relevant = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        if qrel.query_id in query_ids and qrel.relevance > 0:
            relevant[qrel.query_id][qrel.doc_id] = qrel.relevance

    if sampling == "prefix":
        docs = list(dataset.docs_iter())[:limit_docs]
    elif sampling == "qrels":
        required_ids = {doc_id for judgments in relevant.values() for doc_id in judgments}
        if limit_docs is not None and len(required_ids) > limit_docs:
            raise ValueError(
                f"{len(required_ids)} judged documents exceed --limit-docs={limit_docs}"
            )
        all_docs = list(dataset.docs_iter())
        if limit_docs is None:
            docs = all_docs
        else:
            negative_slots = limit_docs - len(required_ids)
            negatives = sorted(
                (doc for doc in all_docs if doc.doc_id not in required_ids),
                key=lambda doc: _negative_key(seed, doc.doc_id),
            )[:negative_slots]
            selected_ids = required_ids | {doc.doc_id for doc in negatives}
            docs = [doc for doc in all_docs if doc.doc_id in selected_ids]
    else:
        raise ValueError(f"unknown sampling mode: {sampling}")

    doc_ids = {doc.doc_id for doc in docs}
    qrels = defaultdict(dict)
    for query_id, judgments in relevant.items():
        for doc_id, relevance in judgments.items():
            if doc_id in doc_ids:
                qrels[query_id][doc_id] = relevance
    queries = [query for query in queries if query.query_id in qrels]
    return docs, queries, qrels


def slice_fingerprint(docs, queries) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for value in sorted(doc.doc_id for doc in docs):
        digest.update(b"d\0" + value.encode() + b"\0")
    for value in sorted(query.query_id for query in queries):
        digest.update(b"q\0" + value.encode() + b"\0")
    return digest.hexdigest()


def write_slice_manifest(path, dataset, sampling, seed, docs, queries) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "sampling": sampling,
                "sample_seed": seed,
                "documents": len(docs),
                "queries": len(queries),
                "fingerprint": slice_fingerprint(docs, queries),
            },
            indent=2,
        )
        + "\n"
    )
