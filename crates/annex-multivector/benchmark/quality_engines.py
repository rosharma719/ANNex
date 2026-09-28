"""Local quality comparators: exact dense search, lexical search, and native RRF.

Qdrant uses shared, explicit BM25 weights; LanceDB uses its native default FTS.
These distinctions are part of the frozen contract, not hidden normalization.
"""

from collections import Counter
from contextlib import contextmanager
import json
import math
import socket
import subprocess
import time

import numpy as np
from server import annex_server, http

STRATEGIES = ("dense", "bm25", "hybrid_rrf")


def document_text(doc):
    return (getattr(doc, "title", "") + " " + doc.text).strip()


def terms(text):
    # Rust's char::is_alphanumeric split followed by Unicode lowercase.
    return "".join(c if c.isalnum() else " " for c in text).lower().split()


class BM25Weights:
    def __init__(self, texts):
        self.documents = [Counter(terms(text)) for text in texts]
        df = Counter(term for doc in self.documents for term in doc)
        self.ids = {term: i for i, term in enumerate(sorted(df))}
        self.idf = {
            term: math.log1p((len(texts) - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }
        self.average = sum(sum(doc.values()) for doc in self.documents) / len(texts)

    def document(self, index):
        doc = self.documents[index]
        norm = (
            1.2 * (0.25 + 0.75 * sum(doc.values()) / self.average)
            if self.average
            else 1.2
        )
        return {
            "indices": [self.ids[t] for t in doc],
            "values": [n * 2.2 / (n + norm) for n in doc.values()],
        }

    def query(self, text):
        tokens = sorted(set(terms(text)).intersection(self.ids))
        return {
            "indices": [self.ids[t] for t in tokens],
            "values": [self.idf[t] for t in tokens],
        }


@contextmanager
def qdrant_server(binary, directory):
    directory.mkdir(parents=True, exist_ok=True)
    # The native server cannot report a port-0 listener. Bind briefly to obtain
    # an available loopback port; startup checks also catch a subsequent race.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = directory / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "telemetry_disabled": True,
                "service": {"host": "127.0.0.1", "http_port": port, "grpc_port": None},
                "storage": {"storage_path": str((directory / "index").resolve())},
            }
        )
    )
    with (directory / "server.log").open("w") as log:
        process = subprocess.Popen(
            [str(binary.resolve()), "--config-path", str(config.resolve())],
            cwd=directory,
            stdout=log,
            stderr=log,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(300):
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Qdrant startup failed; see {directory / 'server.log'}"
                    )
                try:
                    http(base, "/", timeout=1)
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                raise TimeoutError("Qdrant startup timeout")
            yield base
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@contextmanager
def open_engine(name, args, docs, vectors):
    directory = args.output / name
    if name == "annex":
        with annex_server(args.binary.resolve(), directory, vectors.shape[1]) as base:
            yield Annex(base, docs, vectors)
    elif name == "qdrant":
        with qdrant_server(args.qdrant_binary, directory) as base:
            yield Qdrant(base, docs, vectors)
    elif name == "lancedb":
        yield Lance(directory, docs, vectors)
    else:
        raise ValueError(f"unsupported engine: {name}")


class Annex:
    def __init__(self, base, docs, vectors):
        self.base, self.docs, self.vectors = base, docs, vectors

    def build(self):
        for start in range(0, len(self.docs), 100):
            http(
                self.base,
                "/v1/vectors/upsert",
                {
                    "documents": [
                        {
                            "id": d.doc_id,
                            "metadata": {"_benchmark_doc_id": d.doc_id},
                            "text": document_text(d),
                            "representations": {
                                "semantic": {
                                    "kind": "dense",
                                    "vector": self.vectors[start + i].tolist(),
                                }
                            },
                        }
                        for i, d in enumerate(self.docs[start : start + 100])
                    ]
                },
            )
        if http(self.base, "/v1/stats")["documents"] != len(self.docs):
            raise RuntimeError("ANNex corpus count mismatch")

    def query(self, strategy, query, vector, candidates):
        dense = {
            "kind": "dense",
            "field": "semantic",
            "vector": vector.tolist(),
            "limit": candidates,
            "backend": "exact",
        }
        lexical = {"kind": "bm25", "text": query.text, "limit": candidates}
        result = http(
            self.base,
            "/v1/retrieve",
            {
                "prefetch": {
                    "dense": [dense],
                    "bm25": [lexical],
                    "hybrid_rrf": [dense, lexical],
                }[strategy],
                "limit": 100,
                "fusion": {"kind": "rrf", "k": 60},
                "filter": {
                    "op": "not",
                    "filter": {
                        "op": "eq",
                        "field": "_benchmark_doc_id",
                        "value": query.query_id,
                    },
                },
            },
        )
        return {
            "matches": result["matches"],
            "backend": "+".join(c["backend"] for c in result["trace"]["channels"]),
        }


class Qdrant:
    def __init__(self, base, docs, vectors):
        self.base, self.docs, self.vectors = base, docs, vectors
        self.by_id = {d.doc_id: i for i, d in enumerate(docs)}

    def build(self):
        self.bm25 = BM25Weights([document_text(d) for d in self.docs])
        http(
            self.base,
            "/collections/quality",
            {
                "vectors": {
                    "semantic": {"size": self.vectors.shape[1], "distance": "Cosine"}
                },
                "sparse_vectors": {"lexical": {}},
                "optimizers_config": {"indexing_threshold": 0},
            },
            method="PUT",
        )
        for start in range(0, len(self.docs), 100):
            http(
                self.base,
                "/collections/quality/points?wait=true",
                {
                    "points": [
                        {
                            "id": start + i,
                            "vector": {
                                "semantic": self.vectors[start + i].tolist(),
                                "lexical": self.bm25.document(start + i),
                            },
                        }
                        for i, _ in enumerate(self.docs[start : start + 100])
                    ]
                },
                method="PUT",
            )
        count = http(self.base, "/collections/quality/points/count", {"exact": True})[
            "result"
        ]["count"]
        if count != len(self.docs):
            raise RuntimeError("Qdrant corpus count mismatch")

    def query(self, strategy, query, vector, candidates):
        exclude = self.by_id.get(query.query_id)
        filter_ = {"must_not": [{"has_id": [exclude]}]} if exclude is not None else None
        dense = {
            "query": vector.tolist(),
            "using": "semantic",
            "params": {"exact": True},
            "limit": candidates,
            "filter": filter_,
        }
        lexical = {
            "query": self.bm25.query(query.text),
            "using": "lexical",
            "limit": candidates,
            "filter": filter_,
        }
        if strategy == "hybrid_rrf":
            # Qdrant ranks from zero; k=61 matches 1/(60 + one-based rank).
            body = {"prefetch": [dense, lexical], "query": {"rrf": {"k": 61}}}
        else:
            body = (dense if strategy == "dense" else lexical).copy()
        body.update(limit=100, with_payload=False, with_vector=False)
        result = http(self.base, "/collections/quality/points/query", body)["result"][
            "points"
        ]
        return {
            "matches": [
                {"id": self.docs[p["id"]].doc_id, "score": p["score"]} for p in result
            ],
            "backend": f"qdrant_server_exact_{strategy}_shared_bm25",
        }


class Lance:
    def __init__(self, directory, docs, vectors):
        self.directory, self.docs, self.vectors = directory, docs, vectors

    def build(self):
        import lancedb
        import pyarrow as pa
        from lancedb.index import FTS

        db = lancedb.connect(self.directory)
        data = pa.table(
            {
                "id": [d.doc_id for d in self.docs],
                "text": [document_text(d) for d in self.docs],
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(np.asarray(self.vectors).ravel()), self.vectors.shape[1]
                ),
            }
        )
        self.table = db.create_table("quality", data)
        self.table.create_index("text", config=FTS())
        if self.table.count_rows() != len(self.docs):
            raise RuntimeError("LanceDB corpus count mismatch")

    def query(self, strategy, query, vector, candidates):
        from lancedb.rerankers import RRFReranker

        if strategy == "dense":
            search = (
                self.table.search(vector, query_type="vector")
                .distance_type("cosine")
                .bypass_vector_index()
            )
            score = "_distance"
        elif strategy == "bm25":
            search = self.table.search(query.text, query_type="fts", fts_columns="text")
            score = "_score"
        else:
            search = (
                self.table.search(query_type="hybrid", fts_columns="text")
                .vector(vector)
                .text(query.text)
                .distance_type("cosine")
                .bypass_vector_index()
                .rerank(RRFReranker(K=60))
            )
            score = "_relevance_score"
        escaped = query.query_id.replace("'", "''")
        rows = (
            search.where(f"id != '{escaped}'", prefilter=True)
            .select(["id"] if strategy == "hybrid_rrf" else ["id", score])
            .limit(candidates)
            .to_list()[:100]
        )
        return {
            "matches": [
                {"id": r["id"], "score": -r[score] if strategy == "dense" else r[score]}
                for r in rows
            ],
            "backend": f"lancedb_native_exact_{strategy}_default_fts",
        }
