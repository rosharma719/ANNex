import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
from data import load_slice
from embeddings import (
    cached_fixed,
    cached_ragged,
    fixed_fingerprint,
    ragged_fingerprint,
)
from measurement import Journal, evaluate, summarize


class MeasurementTests(unittest.TestCase):
    def test_trec_linear_gains_and_full_recall_denominator(self):
        result = evaluate({"q": {"a": 1, "b": 2}}, {"q": ["a", "b"]})
        self.assertAlmostEqual(result["ndcg@10"], 0.8597186998521972)
        self.assertEqual(result["mrr@10"], 1.0)
        self.assertEqual(result["recall@100"], 1.0)
        missing = evaluate({"q": {"a": 1, "b": 2}}, {"q": ["a"]})
        self.assertEqual(missing["recall@20"], 0.5)

    def setUp(self):
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_embedding_cache_rejects_changed_bytes_and_records_resolved_identity(self):
        for kind in ("ragged", "fixed"):
            with (
                self.subTest(kind=kind),
                tempfile.TemporaryDirectory() as temporary,
                patch("embeddings._model_revision", return_value="fixture-commit"),
            ):
                root = Path(temporary)
                args = (root, "fixture", "document", ["a"], ["alpha"])

                def load(encoder):
                    if kind == "ragged":
                        return cached_ragged(*args, encoder, encoder_config={})
                    return cached_fixed(*args, encoder, normalized=True)

                def forbidden():
                    raise AssertionError("cache hit must not encode")

                _, info = load(
                    lambda: (
                        [np.asarray([[1.0, 0.0]], dtype=np.float32)]
                        if kind == "ragged"
                        else [[1.0, 0.0]]
                    )
                )
                _, reused = load(forbidden)
                self.assertEqual(info["files_sha256"], reused["files_sha256"])
                self.assertEqual(reused["model_revision"], "fixture-commit")
                if kind == "ragged":
                    self.assertEqual(
                        ragged_fingerprint(*args, {})["files_sha256"],
                        info["files_sha256"],
                    )
                else:
                    self.assertEqual(
                        fixed_fingerprint(*args)["files_sha256"], info["files_sha256"]
                    )
                with (Path(info["path"]) / "values.npy").open("r+b") as file:
                    file.seek(-1, 2)
                    value = file.read(1)
                    file.seek(-1, 2)
                    file.write(bytes([value[0] ^ 1]))
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    load(forbidden)

    def test_errors_and_interruption_survive_replay_with_original_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "dataset": "fixture",
                        "documents": 1,
                        "protocol": {},
                        "qrels": {str(i): {"a": 1} for i in range(4)},
                    }
                )
            )
            journal = Journal(root / "events.jsonl")
            journal.write("system_started", "fixture")
            journal.query(
                "fixture",
                "0",
                "exact",
                lambda: {"matches": [{"id": "a", "score": 1.0}], "backend": "exact"},
            )

            def timeout():
                raise TimeoutError("injected timeout")

            journal.query("fixture", "1", "exact", timeout)
            journal.write(
                "query_started", "fixture", qid="2", requested_backend="exact"
            )
            journal.close()
            with (root / "events.jsonl").open("ab") as stream:
                stream.write(b'{"event":')
            result = summarize(root)["systems"]["fixture"]
            self.assertEqual(result["status"], "interrupted")
            self.assertEqual(result["failed_queries"], 3)
            self.assertEqual(result["attempted_queries"], 3)
            self.assertEqual(result["ndcg@10"], 0.25)
            self.assertEqual(
                [r["status"] for r in result["per_query"]],
                ["ok", "error", "interrupted", "not_attempted"],
            )
            self.assertEqual(result["per_query"][0]["scores"], [1.0])
            self.assertIn("TimeoutError", result["per_query"][1]["error"])
            with self.assertRaises(FileExistsError):
                Journal(root / "events.jsonl")

    def test_slices_keep_out_of_corpus_judgments_and_do_not_load_full_prefix(self):
        def documents():
            yield NS(doc_id="a", text="alpha")
            raise AssertionError("prefix loaded beyond requested documents")

        dataset = NS(
            docs_iter=documents,
            queries_iter=lambda: iter([NS(query_id="q", text="query")]),
            qrels_iter=lambda: iter(
                [
                    NS(query_id="q", doc_id="a", relevance=1),
                    NS(query_id="q", doc_id="b", relevance=1),
                ]
            ),
        )
        with patch.dict("sys.modules", {"ir_datasets": NS(load=lambda _: dataset)}):
            docs, queries, qrels = load_slice("fixture", 1, 1)
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(queries), 1)
        self.assertEqual(qrels, {"q": {"a": 1, "b": 1}})


if __name__ == "__main__":
    unittest.main()
