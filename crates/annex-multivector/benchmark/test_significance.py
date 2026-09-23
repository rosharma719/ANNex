import unittest

from significance import paired_bootstrap


class PairedBootstrapTests(unittest.TestCase):
    def test_identical_runs_are_not_significant(self):
        qrels = {str(i): {"good": 1} for i in range(20)}
        run = {str(i): ["good"] for i in range(20)}
        result = paired_bootstrap(run, run, qrels, samples=1000)
        self.assertEqual(result["delta_b_minus_a"], 0.0)
        self.assertFalse(result["significant_at_0.05"])

    def test_consistent_improvement_is_significant(self):
        qrels = {str(i): {"good": 1} for i in range(20)}
        worse = {str(i): ["bad"] for i in range(20)}
        better = {str(i): ["good"] for i in range(20)}
        result = paired_bootstrap(worse, better, qrels, samples=1000)
        self.assertGreater(result["delta_b_minus_a"], 0)
        self.assertTrue(result["significant_at_0.05"])


if __name__ == "__main__":
    unittest.main()
