"""Real HTTP lifecycle and benchmark adapter tests; no datasets or model downloads."""

import json
import os
import tempfile
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
