"""Deterministic paired-bootstrap comparisons for retrieval runs."""

import numpy as np
from measurement import evaluate


def per_query_ndcg(run, qrels, k=10):
    return np.asarray(
        [row[f"ndcg@{k}"] for row in evaluate(qrels, run, k)["per_query"]],
        dtype=np.float64,
    )


def paired_bootstrap(run_a, run_b, qrels, k=10, samples=10_000, seed=13):
    """Return B-A delta, confidence interval, and two-sided paired p-value."""
    differences = per_query_ndcg(run_b, qrels, k) - per_query_ndcg(run_a, qrels, k)
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        boot[start : start + count] = differences[indices].mean(axis=1)
    probability_nonpositive = (np.count_nonzero(boot <= 0) + 1) / (samples + 1)
    probability_nonnegative = (np.count_nonzero(boot >= 0) + 1) / (samples + 1)
    return {
        "metric": f"ndcg@{k}",
        "queries": len(differences),
        "samples": samples,
        "seed": seed,
        "delta_b_minus_a": float(differences.mean()),
        "confidence_interval_95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "p_value_two_sided": float(
            min(1.0, 2 * min(probability_nonpositive, probability_nonnegative))
        ),
        "significant_at_0.05": bool(
            np.percentile(boot, 2.5) > 0 or np.percentile(boot, 97.5) < 0
        ),
    }
