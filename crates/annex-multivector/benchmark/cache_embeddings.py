#!/usr/bin/env python3
"""Precompute benchmark embeddings without building retrieval indexes."""
import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from colbert_config import MODEL_ID, cache_config, load as load_colbert
from data import load_slice
from embeddings import cached_fixed, cached_ragged
from env import load_env
from run import colbert_encode

load_env()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="beir/nfcorpus/test")
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--sampling", choices=["prefix", "qrels"], default="prefix")
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=Path("benchmark/cache"))
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    docs, queries, _ = load_slice(
        args.dataset, args.limit_docs, args.limit_queries, args.sampling, args.sample_seed
    )
    doc_ids = [doc.doc_id for doc in docs]
    doc_texts = [(getattr(doc, "title", "") + " " + doc.text).strip() for doc in docs]
    query_ids = [query.query_id for query in queries]
    query_texts = [query.text for query in queries]

    colbert = None

    def encode_colbert(texts, is_query):
        nonlocal colbert
        if colbert is None:
            colbert = load_colbert()
        return colbert_encode(colbert, texts, is_query, args.batch_size)

    _, colbert_docs = cached_ragged(
        args.cache_dir, MODEL_ID, "document", doc_ids, doc_texts,
        lambda: encode_colbert(doc_texts, False), args.refresh_cache, cache_config("document"),
    )
    _, colbert_queries = cached_ragged(
        args.cache_dir, MODEL_ID, "query", query_ids, query_texts,
        lambda: encode_colbert(query_texts, True), args.refresh_cache, cache_config("query"),
    )

    minilm = None

    def encode_minilm(texts):
        nonlocal minilm
        if minilm is None:
            minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return minilm.encode(
            texts, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True
        )

    _, minilm_docs = cached_fixed(
        args.cache_dir, "sentence-transformers/all-MiniLM-L6-v2", "document",
        doc_ids, doc_texts, lambda: encode_minilm(doc_texts), True, args.refresh_cache,
    )
    _, minilm_queries = cached_fixed(
        args.cache_dir, "sentence-transformers/all-MiniLM-L6-v2", "query",
        query_ids, query_texts, lambda: encode_minilm(query_texts), True, args.refresh_cache,
    )
    print(
        json.dumps(
            {
                "documents": len(docs),
                "queries": len(queries),
                "cache": {
                    "colbert_documents": colbert_docs,
                    "colbert_queries": colbert_queries,
                    "minilm_documents": minilm_docs,
                    "minilm_queries": minilm_queries,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
