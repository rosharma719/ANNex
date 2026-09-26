"""Exercise real CLI control flow without downloading datasets or models."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch


HERE = Path(__file__).parent


def module(name, **members):
    result = ModuleType(name)
    result.__dict__.update(members)
    return result


def forbidden(*args, **kwargs):
    raise AssertionError("guard must run before model loading, encoding, or queries")


class HarnessProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.index = self.root / "index"
        self.index.mkdir()
        (self.index / "manifest.json").write_text("{}")
        self.docs = [NS(doc_id="a", text="alpha"), NS(doc_id="b", text="beta")]
        self.queries = [NS(query_id=str(i), text=f"query{i}") for i in range(8)]
        self.qrels = {q.query_id: {"a": 1} for q in self.queries}
        self.modules = {
            "numpy": module("numpy", percentile=lambda values, _: sorted(values)[len(values)//2],
                            asarray=lambda value: NS(tolist=lambda: value)),
            "colbert_config": module("colbert_config", MODEL_ID="fixture-model",
                                     cache_config=lambda role: {"role": role}, load=forbidden),
            "data": module("data", load_slice=lambda *args: (self.docs, self.queries, self.qrels),
                           write_slice_manifest=forbidden),
            "embeddings": module("embeddings", cached_ragged=forbidden),
            "provenance": module("provenance", write_report=forbidden),
            "run": module("run", colbert_encode=forbidden, http=forbidden,
                          score=lambda run, qrels: {"denominator": len(qrels)}),
            "env": module("env", load_env=lambda: None),
        }

    def load(self, name):
        spec = importlib.util.spec_from_file_location(f"fixture_{name}", HERE / f"{name}.py")
        result = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, self.modules):
            spec.loader.exec_module(result)
        return result

    def arguments(self, name):
        if name == "sweep":
            return ["--index", str(self.index)]
        return ["--engines", "annex", "--output", str(self.root / "results")]

    def test_both_clis_reject_unfrozen_test_before_embeddings(self):
        for name in ("sweep", "headtohead"):
            with self.subTest(name=name):
                harness = self.load(name)
                argv = [name, *self.arguments(name), "--partition", "test"]
                with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "requires --frozen-config"):
                    harness.main()

    def test_both_clis_freeze_without_evaluating_queries(self):
        for name in ("sweep", "headtohead"):
            with self.subTest(name=name):
                harness = self.load(name)
                artifact = self.root / f"{name}.json"
                argv = [name, *self.arguments(name), "--freeze-config", str(artifact)]
                with patch.object(sys, "argv", argv):
                    harness.main()
                self.assertTrue(artifact.exists())
                # A changed operating point must fail before hitting the cache/encoder.
                flag = "--candidates" if name == "sweep" else "--annex-candidates"
                argv = [name, *self.arguments(name), "--partition", "test", "--frozen-config", str(artifact), flag, "999"]
                with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "differ"):
                    harness.main()

    def test_sweep_retains_failures_in_per_query_records_and_denominator(self):
        harness = self.load("sweep")
        calls = 0
        def http(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TimeoutError("fixture timeout")
            return {"matches": [{"id": "a"}]}
        harness.http = http
        report = harness.report_for(self.queries[:2], [[1.], [2.]],
            {k: self.qrels[k] for k in ["0", "1"]}, "unused", "muvera", 100, 4, 256, 2, None)
        self.assertEqual(report["denominator"], 2)
        self.assertEqual(report["failed_queries"], 1)
        self.assertEqual(report["per_query"][0]["ranked_ids"], ["a"])
        self.assertEqual(report["per_query"][1]["ranked_ids"], [])
        self.assertIn("TimeoutError", report["per_query"][1]["error"])
        self.assertGreaterEqual(report["per_query"][1]["latency_ms"], 0)


if __name__ == "__main__":
    unittest.main()
