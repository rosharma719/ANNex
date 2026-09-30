import os
import subprocess
import sys
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


if __name__ == "__main__":
    unittest.main()
