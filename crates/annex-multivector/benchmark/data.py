"""Deterministic BEIR benchmark slices without synthetic relevance labels."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import defaultdict
from itertools import islice


def _negative_key(seed: int, doc_id: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{doc_id}".encode(), digest_size=8).digest()


def load_slice(
    dataset_name, limit_docs=None, limit_queries=None, sampling="prefix", seed=13
):
    import ir_datasets

    dataset = ir_datasets.load(dataset_name)
    for name, limit in [("limit_docs", limit_docs), ("limit_queries", limit_queries)]:
        if limit is not None and limit <= 0:
            raise ValueError(f"{name} must be positive or omitted for the full corpus")
    queries = list(islice(dataset.queries_iter(), limit_queries))
    query_ids = {query.query_id for query in queries}
    relevant = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        if qrel.query_id in query_ids and qrel.relevance > 0:
            relevant[qrel.query_id][qrel.doc_id] = qrel.relevance

    if sampling == "prefix":
        docs = list(islice(dataset.docs_iter(), limit_docs))
    elif sampling == "qrels":
        required_ids = {
            doc_id for judgments in relevant.values() for doc_id in judgments
        }
        if limit_docs is not None and len(required_ids) > limit_docs:
            raise ValueError(
                f"{len(required_ids)} judged documents exceed --limit-docs={limit_docs}"
            )
        if limit_docs is None:
            docs = list(dataset.docs_iter())
        else:
            required = []

            def negatives():
                for doc in dataset.docs_iter():
                    if doc.doc_id in required_ids:
                        required.append(doc)
                    else:
                        yield doc

            negative_slots = limit_docs - len(required_ids)
            # Keep O(limit_docs) documents, not a full corpus sorted in memory.
            selected = heapq.nsmallest(
                negative_slots,
                negatives(),
                key=lambda doc: (_negative_key(seed, doc.doc_id), doc.doc_id),
            )
            if negative_slots == 0:
                # nsmallest(0, ...) does not consume the source.
                required = [
                    doc for doc in dataset.docs_iter() if doc.doc_id in required_ids
                ]
            if {doc.doc_id for doc in required} != required_ids:
                raise ValueError(
                    "relevance judgments reference documents absent from the corpus"
                )
            docs = sorted(required + selected, key=lambda doc: doc.doc_id)
    else:
        raise ValueError(f"unknown sampling mode: {sampling}")

    # Retain original positive judgments, including documents outside a diagnostic
    # slice. Dropping them would inflate Recall and remove the hardest queries.
    queries = [query for query in queries if query.query_id in relevant]
    return docs, queries, dict(relevant)


def slice_fingerprint(docs, queries) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for value in sorted(doc.doc_id for doc in docs):
        digest.update(b"d\0" + value.encode() + b"\0")
    for value in sorted(query.query_id for query in queries):
        digest.update(b"q\0" + value.encode() + b"\0")
    return digest.hexdigest()


def write_slice_manifest(
    path, dataset, sampling, seed, docs, queries, qrels=None
) -> None:
    doc_ids = {doc.doc_id for doc in docs}
    coverage = (
        None
        if qrels is None
        else {
            "positive_judgments": sum(len(v) for v in qrels.values()),
            "out_of_corpus_positive_judgments": sum(
                d not in doc_ids for v in qrels.values() for d in v
            ),
            "queries_without_in_corpus_positive": sum(
                not doc_ids.intersection(v) for v in qrels.values()
            ),
        }
    )
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "sampling": sampling,
                "sample_seed": seed,
                "documents": len(docs),
                "queries": len(queries),
                "fingerprint": slice_fingerprint(docs, queries),
                "relevance_coverage": coverage,
            },
            indent=2,
        )
        + "\n"
    )
