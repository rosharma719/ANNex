"""Real HTTP lifecycle and benchmark adapter tests; no datasets or model downloads."""

import json
import os
import tempfile
import urllib.request
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from headtohead import bench_annex
from measurement import Journal, summarize
from server import annex_server, http


@unittest.skipUnless(
    os.environ.get("ANNEX_TEST_BINARY"), "set ANNEX_TEST_BINARY to run real HTTP tests"
)
class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = Path(os.environ["ANNEX_TEST_BINARY"]).resolve()

    def test_serving_configuration_and_global_runtime_counters(self):
        flags = (
            "--query-threads",
            "2",
            "--query-concurrency",
            "3",
            "--ingest-threads",
            "1",
            "--ingest-concurrency",
            "2",
            "--maintenance-threads",
            "1",
            "--maintenance-concurrency",
            "1",
        )
        with annex_server(self.binary, self.root, 2, centroids=2, extra=flags) as base:
            before = http(base, "/v1/runtime")
            for name, threads, capacity in [
                ("query", 2, 3),
                ("ingest", 1, 2),
                ("maintenance", 1, 1),
            ]:
                self.assertEqual(before[name]["threads"], threads)
                self.assertEqual(before[name]["capacity"], capacity)
                self.assertEqual(before[name]["admitted"], 0)
            http(base, "/v1/collections", {"name": "a", "config": {"dimension": 2}})
            body = {
                "documents": [
                    {
                        "id": "a",
                        "representations": {
                            "semantic": {"kind": "dense", "vector": [1, 0]}
                        },
                    }
                ]
            }
            http(base, "/v1/collections/a/vectors/upsert", body)
            http(
                base,
                "/v1/collections/a/dense/index",
                {"field": "semantic", "m": 4, "ef_construct": 16},
            )
            with urllib.request.urlopen(base + "/v1/collections/a/stats") as response:
                self.assertEqual(response.status, 200)
                for header in [
                    "x-annex-queue-ms",
                    "x-annex-work-ms",
                    "x-annex-request-ms",
                ]:
                    self.assertGreaterEqual(float(response.headers[header]), 0)
            after = http(base, "/v1/runtime")
            self.assertEqual(after["ingest"]["admitted"], 2)
            self.assertEqual(after["maintenance"]["admitted"], 1)
            self.assertEqual(after["query"]["admitted"], 1)
            # Named routing performs a lookup and engine job under one permit.
            self.assertEqual(after["query"]["completed_jobs"], 2)
            for pool in after.values():
                self.assertEqual(pool["in_flight"], 0)
                self.assertEqual(pool["queued_jobs"], 0)
                self.assertEqual(pool["running_jobs"], 0)

    def test_http_contract_delete_and_reopen(self):
        with annex_server(self.binary, self.root, 2, centroids=2) as base:
            http(base, "/v1/train", {"vectors": [[1, 0], [0, 1]], "iterations": 2})
            http(
                base,
                "/v1/vectors/upsert",
                {
                    "documents": [
                        {"id": "a", "vectors": [[1, 0]]},
                        {"id": "b", "vectors": [[0, 1]]},
                    ]
                },
            )
            request = {"vectors": [[1, 0]], "top_k": 1, "explain": True}
            self.assertEqual(
                http(base, "/v1/query", request)["stats"]["candidate_backend"], "muvera"
            )
            generation = http(base, "/v1/stats")["generation"]
            invalid = [
                (
                    "/v1/debug/candidates",
                    {"vectors": [[1, 0]], "count": 1, "ef_search": 2},
                ),
                ("/v1/query", {**request, "filter": {"tenant": "other"}}),
                ("/v1/query", {**request, "candidate_backend": "hnsw", "probes": 2}),
                (
                    "/v1/query",
                    {**request, "candidate_backend": "muvera", "ef_search": 2},
                ),
                ("/v1/query", {**request, "top_k": 0}),
                ("/v1/query", {**request, "candidate_backend": "unknown"}),
                ("/v1/query", {**request, "vectors": [[1, 0]] * 1025}),
                ("/v1/fde/index", {"m": 2**64 - 1}),
                ("/v1/fde/index", {"ef_construct": 2**64 - 1}),
                ("/v1/train", {"vectors": [[1, 0]], "iterations": 2**64 - 1}),
                (
                    "/v1/vectors/upsert",
                    {
                        "documents": [
                            {"id": "c", "vectors": [[1, 0]], "tenant": "ignored"}
                        ]
                    },
                ),
            ]
            for route, body in invalid:
                with self.subTest(route=route, fields=list(body)):
                    with self.assertRaisesRegex(RuntimeError, "HTTP 4"):
                        http(base, route, body, timeout=5)
                    self.assertEqual(http(base, "/v1/stats")["generation"], generation)
                    self.assertEqual(http(base, "/healthz")["status"], "ok")
            http(base, "/v1/fde/index", {"m": 4, "ef_construct": 16})
            self.assertEqual(
                http(base, "/v1/query", request)["stats"]["candidate_backend"], "hnsw"
            )
            self.assertTrue(http(base, "/v1/vectors/delete", {"id": "a"})["deleted"])
            self.assertFalse(http(base, "/v1/vectors/delete", {"id": "a"})["deleted"])
            self.assertEqual(http(base, "/v1/query", request)["matches"][0]["id"], "b")
        with annex_server(self.binary, self.root, 2, centroids=2) as base:
            result = http(base, "/v1/query", request)
            self.assertEqual(result["stats"]["candidate_backend"], "muvera")
            self.assertEqual([hit["id"] for hit in result["matches"]], ["b"])

    def test_collection_hybrid_named_ann_and_compaction_lifecycle(self):
        def route(name, operation):
            return f"/v1/collections/{name}/{operation}"

        query = {
            "prefetch": [
                {"kind": "dense", "field": "semantic", "vector": [1, 0], "limit": 10},
                {"kind": "bm25", "text": "E123", "limit": 10},
            ],
            "limit": 2,
        }
        with annex_server(self.binary, self.root, 2) as base:
            for name, text in [("a", "E123 repair"), ("b", "different")]:
                http(
                    base, "/v1/collections", {"name": name, "config": {"dimension": 2}}
                )
                http(
                    base,
                    route(name, "vectors/upsert"),
                    {
                        "documents": [
                            {
                                "id": "shared",
                                "text": text,
                                "metadata": {"tenant": name},
                                "representations": {
                                    "semantic": {"kind": "dense", "vector": [1, 0]}
                                },
                            }
                        ]
                    },
                )
            self.assertEqual(http(base, "/v1/stats")["documents"], 0)
            for name in ("a", "b"):
                result = http(base, route(name, "retrieve"), query)
                self.assertEqual(result["matches"][0]["metadata"]["tenant"], name)
            http(
                base,
                route("a", "dense/index"),
                {"field": "semantic", "m": 4, "ef_construct": 16},
            )
            result = http(base, route("a", "retrieve"), query)
            self.assertEqual(result["trace"]["channels"][0]["backend"], "exact_dense")
            http(base, route("a", "compact"), {})
            self.assertEqual(
                http(base, route("a", "retrieve"), query)["matches"], result["matches"]
            )
            for body in [
                {**query, "made_up": True},
                {
                    **query,
                    "filter": {"op": "range", "field": "year", "gte": 2, "lte": 1},
                },
                {
                    **query,
                    "prefetch": [{"kind": "dense", "field": "semantic", "vector": [1]}],
                },
                {
                    **query,
                    "prefetch": [
                        {
                            "kind": "sparse",
                            "field": "semantic",
                            "vector": {"indices": [1], "values": []},
                        }
                    ],
                },
            ]:
                with self.assertRaisesRegex(RuntimeError, "HTTP 4"):
                    http(base, route("a", "retrieve"), body)
            with self.assertRaisesRegex(RuntimeError, "HTTP 4"):
                http(
                    base,
                    "/v1/collections",
                    {"name": "../escape", "config": {"dimension": 2}},
                )
        with annex_server(self.binary, self.root, 2) as base:
            self.assertEqual(http(base, "/v1/collections")["collections"], ["a", "b"])
            result = http(base, route("a", "retrieve"), query)
            self.assertEqual(result["trace"]["channels"][0]["backend"], "exact_dense")
            http(base, route("a", "vectors/delete"), {"id": "shared"})
            self.assertEqual(
                http(
                    base,
                    route("a", "retrieve"),
                    {"prefetch": [{"kind": "bm25", "text": "E123"}]},
                )["matches"],
                [],
            )
            self.assertEqual(http(base, route("b", "stats"))["documents"], 1)

    def test_collection_analyzer_config_is_persisted_and_validated(self):
        english = {
            "dimension": 2,
            "analyzer": {
                "stem": True,
                "stopwords": "english",
                "ascii_folding": True,
                "max_token_length": 40,
            },
        }
        document = {
            "documents": [
                {
                    "id": "d",
                    "text": "The runners are running",
                    "representations": {
                        "semantic": {"kind": "dense", "vector": [1, 0]}
                    },
                }
            ]
        }
        stemmed = {"prefetch": [{"kind": "bm25", "text": "running", "limit": 5}]}
        with annex_server(self.binary, self.root, 2) as base:
            http(base, "/v1/collections", {"name": "english", "config": english})
            http(base, "/v1/collections/english/vectors/upsert", document)
            result = http(base, "/v1/collections/english/retrieve", stemmed)
            self.assertEqual([m["id"] for m in result["matches"]], ["d"])
            self.assertEqual(
                http(
                    base,
                    "/v1/collections/english/retrieve",
                    {"prefetch": [{"kind": "bm25", "text": "the", "limit": 5}]},
                )["matches"],
                [],
            )
            for bad in [
                {"dimension": 2, "analyzer": {"max_token_length": 0}},
                {"dimension": 2, "analyzer": {"stopwords": "klingon"}},
                {"dimension": 2, "analyzer": {"unknown": True}},
            ]:
                with self.assertRaisesRegex(RuntimeError, "HTTP 4"):
                    http(base, "/v1/collections", {"name": "bad", "config": bad})
        # Collections reopen from the persisted analyzer without CLI flags.
        with annex_server(self.binary, self.root, 2) as base:
            result = http(base, "/v1/collections/english/retrieve", stemmed)
            self.assertEqual([m["id"] for m in result["matches"]], ["d"])

    def test_real_adapters_report_backend_build_cost_and_scalar_order(self):
        docs = [NS(doc_id="a"), NS(doc_id="b")]
        queries = [NS(query_id="q")]
        vectors = [
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 1.0]], dtype=np.float32),
        ]
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset": "fixture",
                    "documents": 2,
                    "protocol": {},
                    "qrels": {"q": {"a": 1}},
                }
            )
        )
        journal = Journal(self.root / "events.jsonl")
        try:
            for backend in ("muvera", "hnsw"):
                journal.write("system_started", backend)
                bench_annex(
                    docs,
                    queries,
                    vectors,
                    vectors[:1],
                    journal,
                    backend,
                    self.binary,
                    self.root / backend,
                    backend,
                    2,
                    centroids=2,
                    m=4,
                    ef_construct=16,
                )
                journal.write("system_finished", backend, status="complete")
        finally:
            journal.close()
        for backend, result in summarize(self.root)["systems"].items():
            self.assertEqual(result["failed_queries"], 0)
            self.assertEqual(set(result["stages"]), {"train", "ingest", "index_ready"})
            self.assertEqual(result["build_s"], sum(result["stages"].values()))
            row = result["per_query"][0]
            self.assertEqual(row["executed_backend"], backend)
            self.assertEqual(row["ranked_ids"], ["a", "b"])
            np.testing.assert_allclose(row["scores"], [1.0, 0.0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
