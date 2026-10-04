import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import annexdb

os.environ.setdefault("VECTORDB_EXACT_FALLBACK_ENABLED", "true")
os.environ.setdefault("VECTORDB_EXACT_FALLBACK_THRESHOLD", "1000")

POINT_IDS = np.array([101, 103, 107, 109, 113, 127, 131, 137], dtype=np.uint64)
POINTS = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 2.0],
        [3.0, 1.0],
        [-2.0, -1.0],
        [1.5, 2.5],
        [4.0, -2.0],
        [-3.0, 3.0],
    ],
    dtype=np.float32,
)


def exact(query, k):
    distances = np.sum((POINTS - query) ** 2, axis=1)
    order = np.argsort(distances, kind="stable")[:k]
    return POINT_IDS[order], distances[order]


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snapshot = os.environ.get("ANNEX_TEST_SNAPSHOT")
        if not snapshot:
            raise unittest.SkipTest("ANNEX_TEST_SNAPSHOT is not set")
        cls.index = annexdb.Index(snapshot)

    def test_load_and_search(self):
        self.assertEqual(len(self.index), len(POINTS))
        self.assertEqual(self.index.dim(), 2)
        query = np.array([0.25, 0.5], dtype=np.float32)
        ids, scores = self.index.search(query, k=5)
        self.assertEqual(ids.shape, (5,))
        self.assertEqual(scores.shape, (5,))
        self.assertEqual(ids.dtype, np.uint64)
        self.assertEqual(scores.dtype, np.float32)
        expected_ids, expected_scores = exact(query, 5)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(scores, expected_scores, rtol=0, atol=1e-6)

    def test_quantized_index_loads_and_searches(self):
        quantized = annexdb.Index(os.environ["ANNEX_TEST_SNAPSHOT"], quantize=True)
        query = np.array([0.25, 0.5], dtype=np.float32)
        ids, scores = quantized.search(query, k=4, sq8_screen=True)
        expected_ids, expected_scores = exact(query, 4)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(scores, expected_scores, rtol=0, atol=1e-6)

    def test_single_query_validation(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            self.index.search(np.array([1.0], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.index.search(np.array([np.nan, 1.0], dtype=np.float32))
        ids, scores = self.index.search(np.array([1.0, 1.0], dtype=np.float32), k=0)
        self.assertEqual(ids.shape, (0,))
        self.assertEqual(scores.shape, (0,))

        huge = int(np.iinfo(np.uintp).max)
        ids, scores = self.index.search(
            np.array([1.0, 1.0], dtype=np.float32), k=huge, ef=huge
        )
        self.assertEqual(ids.shape, (len(POINTS),))
        self.assertEqual(scores.shape, (len(POINTS),))

    def test_noncontiguous_inputs_are_rejected(self):
        with self.assertRaises((TypeError, ValueError)):
            self.index.search(np.arange(4, dtype=np.float32)[::2])
        with self.assertRaises((TypeError, ValueError)):
            self.index.search_batch(np.arange(10, dtype=np.float32).reshape(2, 5).T)

    def test_batch_handles_skewed_and_excessive_thread_counts(self):
        queries = np.array(
            [[0.25, 0.5], [1.25, 1.75], [-1.5, -0.5], [3.5, -1.0], [-2.0, 2.0]],
            dtype=np.float32,
        )
        single = [self.index.search(query, k=4) for query in queries]
        for threads in (4, 10_000):
            ids, scores = self.index.search_batch(queries, k=4, threads=threads)
            self.assertEqual(ids.shape, (5, 4))
            self.assertEqual(scores.shape, (5, 4))
            for row, (single_ids, single_scores) in enumerate(single):
                np.testing.assert_array_equal(ids[row], single_ids)
                np.testing.assert_allclose(scores[row], single_scores, rtol=0, atol=1e-6)

    def test_concurrent_python_searches(self):
        queries = [
            np.array([i / 10.0, (i % 5) / 7.0], dtype=np.float32) for i in range(32)
        ]
        expected = [exact(query, 3) for query in queries]
        with ThreadPoolExecutor(max_workers=8) as executor:
            actual = list(executor.map(lambda query: self.index.search(query, k=3), queries))
        for (ids, scores), (expected_ids, expected_scores) in zip(actual, expected):
            np.testing.assert_array_equal(ids, expected_ids)
            np.testing.assert_allclose(scores, expected_scores, rtol=0, atol=1e-6)

    def test_batch_handles_empty_rows_and_zero_k(self):
        ids, scores = self.index.search_batch(np.empty((0, 2), dtype=np.float32), k=3)
        self.assertEqual(ids.shape, (0, 3))
        self.assertEqual(scores.shape, (0, 3))

        ids, scores = self.index.search_batch(np.zeros((2, 2), dtype=np.float32), k=0)
        self.assertEqual(ids.shape, (2, 0))
        self.assertEqual(scores.shape, (2, 0))

        with self.assertRaises(OverflowError):
            self.index.search_batch(
                np.zeros((2, 2), dtype=np.float32),
                k=int(np.iinfo(np.uintp).max),
            )

    def test_batch_validation(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            self.index.search_batch(np.empty((0, 0), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.index.search_batch(
                np.array([[1.0, 2.0], [3.0, np.inf]], dtype=np.float32)
            )

    def test_real_hnsw_batch_and_concurrent_searches(self):
        script = r'''
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import annexdb

index = annexdb.Index(os.environ["ANNEX_TEST_SNAPSHOT"])
queries = np.array(
    [[0.25, 0.5], [1.25, 1.75], [-1.5, -0.5], [3.5, -1.0], [-2.0, 2.0]],
    dtype=np.float32,
)
single = [index.search(query, k=4, ef=64) for query in queries]
ids, scores = index.search_batch(queries, k=4, ef=64, threads=4)
for row, (single_ids, single_scores) in enumerate(single):
    np.testing.assert_array_equal(ids[row], single_ids)
    np.testing.assert_allclose(scores[row], single_scores, rtol=0, atol=1e-6)
with ThreadPoolExecutor(max_workers=8) as executor:
    concurrent_queries = list(queries) * 4
    concurrent = list(executor.map(lambda query: index.search(query, k=4, ef=64), concurrent_queries))
for position, (actual_ids, actual_scores) in enumerate(concurrent):
    expected_ids, expected_scores = single[position % len(single)]
    np.testing.assert_array_equal(actual_ids, expected_ids)
    np.testing.assert_allclose(actual_scores, expected_scores, rtol=0, atol=1e-6)
'''
        environment = os.environ.copy()
        environment["VECTORDB_EXACT_FALLBACK_ENABLED"] = "false"
        subprocess.run([sys.executable, "-c", script], env=environment, check=True)


def unit_rows(values):
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


class BuildTests(unittest.TestCase):
    """Indexes built from NumPy arrays; these need no snapshot fixture."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.vectors = rng.standard_normal((300, 12)).astype(np.float32)
        cls.query = rng.standard_normal(12).astype(np.float32)

    # What each metric reports as its score, and which direction is closer.
    def oracle(self, metric):
        if metric == "cosine":
            return 1.0 - unit_rows(self.vectors) @ unit_rows(self.query), False
        if metric == "dot":
            return self.vectors @ self.query, True
        return ((self.vectors - self.query) ** 2).sum(axis=1), False

    def test_matches_numpy_for_every_metric(self):
        for metric in ("cosine", "dot", "euclidean"):
            with self.subTest(metric=metric):
                index = annexdb.Index.build(self.vectors, metric=metric)
                self.assertEqual(index.metric(), metric)
                self.assertEqual(len(index), len(self.vectors))
                self.assertEqual(index.dim(), self.vectors.shape[1])
                # ef >= n explores the whole graph, so results are exact.
                ids, scores = index.search(self.query, k=5, ef=len(self.vectors))
                values, higher_is_closer = self.oracle(metric)
                order = np.argsort(-values if higher_is_closer else values)[:5]
                np.testing.assert_array_equal(ids, order.astype(np.uint64))
                np.testing.assert_allclose(scores, values[order], rtol=1e-4, atol=1e-4)

    def test_default_ids_are_row_numbers_and_custom_ids_are_kept(self):
        index = annexdb.Index.build(self.vectors)
        ids, _ = index.search(self.vectors[0], k=1, ef=300)
        self.assertEqual(int(ids[0]), 0)
        custom = np.arange(len(self.vectors), dtype=np.uint64) * 3 + 1000
        index = annexdb.Index.build(self.vectors, ids=custom)
        ids, _ = index.search(self.vectors[5], k=1, ef=300)
        self.assertEqual(int(ids[0]), 1015)

    def test_save_and_reload_round_trip(self):
        index = annexdb.Index.build(self.vectors, metric="euclidean")
        before = index.search(self.query, k=10, ef=300)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "index.bin")
            index.save(path)
            reloaded = annexdb.Index(path)
            self.assertEqual(len(reloaded), len(index))
            self.assertEqual(reloaded.dim(), index.dim())
            self.assertEqual(reloaded.metric(), "euclidean")
            after = reloaded.search(self.query, k=10, ef=300)
            np.testing.assert_array_equal(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            # A loaded index can be saved again.
            second = os.path.join(directory, "again.bin")
            reloaded.save(second)
            np.testing.assert_array_equal(
                before[0], annexdb.Index(second).search(self.query, k=10, ef=300)[0]
            )
            # And loaded with SQ8 screening.
            quantized = annexdb.Index(path, quantize=True)
            ids, _ = quantized.search(self.query, k=3, sq8_screen=True)
            self.assertEqual(ids.shape, (3,))

    def test_batch_search_on_a_built_index(self):
        index = annexdb.Index.build(self.vectors)
        ids, scores = index.search_batch(self.vectors[:40], k=3, ef=300, threads=4)
        self.assertEqual(ids.shape, (40, 3))
        np.testing.assert_array_equal(ids[:, 0], np.arange(40, dtype=np.uint64))

    def test_rejects_bad_input(self):
        build = annexdb.Index.build
        cases = {
            "no rows": lambda: build(np.zeros((0, 4), dtype=np.float32)),
            "no dimensions": lambda: build(np.zeros((3, 0), dtype=np.float32)),
            "non-finite": lambda: build(np.array([[1.0, np.nan]], dtype=np.float32)),
            "fortran order": lambda: build(np.asfortranarray(self.vectors)),
            "unknown metric": lambda: build(self.vectors, metric="manhattan"),
            "wrong id count": lambda: build(self.vectors, ids=np.arange(5, dtype=np.uint64)),
            "duplicate ids": lambda: build(
                self.vectors[:4], ids=np.array([1, 2, 2, 3], dtype=np.uint64)
            ),
            "m too small": lambda: build(self.vectors, m=1),
            "zero ef_construction": lambda: build(self.vectors, ef_construction=0),
        }
        for label, call in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                call()

    def test_save_reports_io_errors(self):
        index = annexdb.Index.build(self.vectors[:10])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                index.save(os.path.join(directory, "missing", "index.bin"))

    def test_search_validates_query_dimension(self):
        index = annexdb.Index.build(self.vectors)
        with self.assertRaises(ValueError):
            index.search(np.zeros(3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
