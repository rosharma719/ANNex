import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import prepare_protocol, query_split


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "point.json"
        self.docs = [NS(doc_id="a", text="alpha", title=""), NS(doc_id="b", text="beta")]
        self.queries = [NS(query_id=str(i), text=f"q{i}") for i in range(10)]
        self.qrels = {q.query_id: {"a": 1, "b": 2} for q in self.queries}
        self.settings = {"candidate_count": 500, "ef_search": 256, "fde": [20, 4, 8]}

    def args(self, **kwargs):
        return NS(**({"partition": "dev", "split_seed": 13, "freeze_config": None,
                      "frozen_config": None, "dataset": "fixture", "sampling": "qrels",
                      "sample_seed": 13, "limit_docs": None, "limit_queries": None} | kwargs))

    def prepare(self, args, settings=None, points=1):
        return prepare_protocol(args, self.docs, self.queries, self.qrels,
                                self.settings if settings is None else settings, operating_points=points)

    def freeze(self):
        return self.prepare(self.args(freeze_config=self.path))

    def test_disjoint_stable_complete_split_and_preserved_qrels(self):
        dev, test = query_split(self.queries, 13)
        self.assertFalse(set(dev) & set(test))
        self.assertEqual(set(dev + test), set(self.qrels))
        self.assertEqual((dev, test), query_split(list(reversed(self.queries)), 13))
        selected, qrels, report = self.prepare(self.args())
        self.assertEqual(set(qrels), {q.query_id for q in selected})
        self.assertTrue(all(v == {"a": 1, "b": 2} for v in qrels.values()))
        self.assertEqual(report["partition"], "dev")

    def test_freeze_then_test_matches_and_cannot_overwrite(self):
        dev, _, _ = self.freeze()
        test, _, report = self.prepare(self.args(partition="test", frozen_config=self.path))
        self.assertFalse({q.query_id for q in dev} & {q.query_id for q in test})
        self.assertEqual(report["label"], "held-out frozen-config test")
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_no_test_without_freeze_or_with_sweep(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            self.prepare(self.args(partition="test"))
        self.freeze()
        with self.assertRaisesRegex(ValueError, "sweep"):
            self.prepare(self.args(partition="test", frozen_config=self.path), points=2)
        with self.assertRaisesRegex(ValueError, "one operating point"):
            self.prepare(self.args(freeze_config=self.path), points=2)

    def test_changed_hyperparameters_qrels_and_content_rejected(self):
        self.freeze()
        test_args = self.args(partition="test", frozen_config=self.path)
        with self.assertRaisesRegex(ValueError, "differ"):
            self.prepare(test_args, {**self.settings, "ef_search": 512})
        self.docs[0].text = "modified corpus"
        with self.assertRaisesRegex(ValueError, "differ"):
            self.prepare(test_args)
        self.docs[0].text = "alpha"
        self.qrels["0"]["a"] = 2
        with self.assertRaisesRegex(ValueError, "differ"):
            self.prepare(test_args)

    def test_changed_split_or_tampered_artifact_rejected(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, "differ"):
            self.prepare(self.args(partition="test", frozen_config=self.path, split_seed=999))
        artifact = json.loads(self.path.read_text())
        artifact["contract"]["settings"]["ef_search"] = 99
        self.path.write_text(json.dumps(artifact))
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.prepare(self.args(partition="test", frozen_config=self.path))

    def test_empty_duplicate_and_unjudgeable_queries_rejected(self):
        self.queries.append(self.queries[0])
        with self.assertRaisesRegex(ValueError, "duplicate query"):
            self.prepare(self.args())
        self.queries.pop()
        self.qrels.pop("0")
        with self.assertRaisesRegex(ValueError, "retained qrels"):
            self.prepare(self.args())

    def test_exploratory_is_never_labelled_test(self):
        selected, _, report = self.prepare(self.args(partition="exploratory"), points=10)
        self.assertEqual(len(selected), len(self.queries))
        self.assertEqual(report["label"], "development exploration")
        with self.assertRaisesRegex(ValueError, "from dev"):
            self.prepare(self.args(partition="exploratory", freeze_config=self.path))


if __name__ == "__main__":
    unittest.main()
