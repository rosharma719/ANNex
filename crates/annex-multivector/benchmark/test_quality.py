"""Optional real comparator checks; no dataset/model downloads."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from quality_engines import STRATEGIES, open_engine


class QualityAdapterTests(unittest.TestCase):
    def check_engine(self, name):
        docs = [
            NS(doc_id="a", text="red apple orchard"),
            NS(doc_id="b", text="blue ocean sea"),
            NS(doc_id="c", text="red fruit"),
        ]
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            args = NS(
                output=Path(temporary),
                binary=Path(os.environ.get("ANNEX_TEST_BINARY", "unused")),
                qdrant_binary=Path(os.environ.get("QDRANT_TEST_BINARY", "unused")),
            )
            with open_engine(name, args, docs, vectors) as engine:
                engine.build()
                for strategy in STRATEGIES:
                    result = engine.query(
                        strategy, NS(query_id="a", text="red apple"), vectors[0], 100
                    )
                    ids = [hit["id"] for hit in result["matches"]]
                    self.assertEqual(ids[0], "c")
                    self.assertNotIn("a", ids)
                    self.assertEqual(len(ids), len(set(ids)))
                    if strategy == "hybrid_rrf":
                        self.assertAlmostEqual(
                            result["matches"][0]["score"], 2 / 61, places=7
                        )
                    # No lexical matches must not fabricate positive candidates.
                    empty = engine.query(
                        "bm25",
                        NS(query_id="absent", text="qxzznotaword"),
                        vectors[0],
                        100,
                    )
                    self.assertEqual(empty["matches"], [])

    @unittest.skipUnless(
        os.environ.get("ANNEX_TEST_BINARY"), "requires ANNEX_TEST_BINARY"
    )
    def test_annex(self):
        self.check_engine("annex")

    @unittest.skipUnless(
        os.environ.get("QDRANT_TEST_BINARY"), "requires QDRANT_TEST_BINARY"
    )
    def test_qdrant_server(self):
        self.check_engine("qdrant")

    @unittest.skipUnless(
        importlib.util.find_spec("lancedb"), "requires optional lancedb"
    )
    def test_lancedb(self):
        self.check_engine("lancedb")
