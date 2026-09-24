#!/usr/bin/env python3
"""C-spike: does per-query difficulty predict which queries need higher c?

Ingests one or more headtohead --annex-sweep matrix.json files (each contains
per_query rows at multiple candidate counts). For each query, computes:

  - "improvement from raising c": did nDCG@10 go up when we spent more effort?
  - "score margin" at the cheap c: how close were top hits together?
  - "top score" at the cheap c: absolute confidence of the best match

Then checks whether the cheap-c difficulty signals correlate with which
queries actually benefit from spending more. If they do, we can build a
runtime policy: "predicted low margin -> escalate to higher c automatically."

If correlations are weak (r < ~0.2), the thesis fails on this data and we
know not to invest months in an SLO/planner story built on these features.

Usage:
    .venv/bin/python benchmark/recall_difficulty.py \\
        benchmark/results/headtohead-fiqa-v2/matrix.json \\
        benchmark/results/headtohead-scifact-v2/matrix.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_annex_series(matrix_path: Path):
    """Return dict {candidate_count: per_query_dict_by_qid} for annex systems."""
    data = json.loads(matrix_path.read_text())
    dataset = data.get("dataset", str(matrix_path))
    series = {}
    for name, sys in data["systems"].items():
        if not name.startswith("annex_multivector"):
            continue
        # Extract candidate count from name (e.g. annex_multivector_c500)
        if "_c" in name:
            c = int(name.rsplit("_c", 1)[1])
        else:
            c = None
        rows = {row["qid"]: row for row in sys.get("per_query", [])}
        if rows:
            series[c] = rows
    return dataset, series


def paired_qids(series):
    """Return qids present in every candidate level."""
    if not series:
        return []
    common = None
    for rows in series.values():
        s = set(rows.keys())
        common = s if common is None else (common & s)
    return sorted(common) if common else []


def pearson(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def spearman(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(xs))
    ry = np.argsort(np.argsort(ys))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze(dataset, series):
    print()
    print(f"### {dataset}")
    print(f"annex candidate levels: {sorted(series.keys())}")
    qids = paired_qids(series)
    print(f"queries with data at every level: {len(qids)}")
    if not qids:
        print("  (nothing to analyze)")
        return

    levels = sorted(series.keys())
    cheap, mid, high = levels[0], levels[len(levels) // 2], levels[-1]
    print(f"cheap c={cheap}, mid c={mid}, high c={high}")

    # Aggregate: mean nDCG at each level
    print()
    print(f"{'c':>6} {'mean nDCG@10':>13} {'p10 nDCG':>10} {'queries<0.3':>12}")
    for c in levels:
        vals = [series[c][qid]["ndcg@10"] for qid in qids]
        print(
            f"{c:>6} {np.mean(vals):>13.4f} {np.percentile(vals, 10):>10.4f} "
            f"{np.mean([1 if v < 0.3 else 0 for v in vals]):>12.1%}"
        )

    # Improvement pattern
    ndcg_cheap = np.array([series[cheap][qid]["ndcg@10"] for qid in qids])
    ndcg_high = np.array([series[high][qid]["ndcg@10"] for qid in qids])
    delta = ndcg_high - ndcg_cheap
    print()
    print(f"nDCG improvement (c={high} vs c={cheap}):")
    print(f"  mean delta:     {np.mean(delta):+.4f}")
    print(f"  queries helped:  {np.mean(delta > 0.01):.1%}")
    print(f"  queries hurt:    {np.mean(delta < -0.01):.1%}")
    print(f"  queries flat:    {np.mean(np.abs(delta) <= 0.01):.1%}")

    # Correlations between cheap-c difficulty signals and delta
    if "top_score" in series[cheap][qids[0]]:
        top_score_cheap = np.array([series[cheap][qid]["top_score"] for qid in qids])
        margin_cheap = np.array([series[cheap][qid]["top_minus_second"] for qid in qids])

        print()
        print("Correlations between cheap-c difficulty signals and improvement delta:")
        r_top_pearson = pearson(top_score_cheap, delta)
        r_top_spearman = spearman(top_score_cheap, delta)
        r_margin_pearson = pearson(margin_cheap, delta)
        r_margin_spearman = spearman(margin_cheap, delta)
        print(f"  top_score      → improvement:   pearson={r_top_pearson:+.3f}  spearman={r_top_spearman:+.3f}")
        print(f"  top-vs-2nd     → improvement:   pearson={r_margin_pearson:+.3f}  spearman={r_margin_spearman:+.3f}")

        # Absolute correlation with per-query nDCG at cheap c
        r_top_ndcg = pearson(top_score_cheap, ndcg_cheap)
        r_margin_ndcg = pearson(margin_cheap, ndcg_cheap)
        print()
        print("Correlations between cheap-c difficulty signals and cheap-c nDCG:")
        print(f"  top_score      → nDCG@10:       pearson={r_top_ndcg:+.3f}")
        print(f"  top-vs-2nd     → nDCG@10:       pearson={r_margin_ndcg:+.3f}")

        # Rank-disagreement signal: if the top-K ids at cheap c differ a lot
        # from the top-K ids at high c, the candidate pool is unstable — a
        # richer difficulty signal than just top score / margin. Uses top-10
        # Jaccard as the cheap approximation. Higher disagreement should
        # correlate with higher improvement delta if the C thesis holds.
        if "ranked_ids" in series[cheap][qids[0]]:
            def jaccard(a, b, k=10):
                sa = set(a[:k])
                sb = set(b[:k])
                if not sa and not sb:
                    return 1.0
                return len(sa & sb) / len(sa | sb)

            disagree = np.array([
                1.0 - jaccard(series[cheap][qid]["ranked_ids"], series[high][qid]["ranked_ids"])
                for qid in qids
            ])
            r_disagree_pearson = pearson(disagree, delta)
            r_disagree_spearman = spearman(disagree, delta)
            print()
            print("Rank-disagreement signal (top-10 Jaccard distance between cheap and high c):")
            print(f"  disagreement   → improvement:   pearson={r_disagree_pearson:+.3f}  spearman={r_disagree_spearman:+.3f}")
            print(f"  mean disagreement: {np.mean(disagree):.3f}   queries with any disagreement: {np.mean(disagree > 0):.1%}")

        # Decile analysis: bucket queries by top_minus_second, look at
        # how often each bucket benefits from more candidates.
        print()
        print("Decile analysis of top-vs-2nd margin (cheap c):")
        print(f"  {'decile':>7} {'range':>18} {'mean nDCG@cheap':>16} {'mean nDCG@high':>15} {'delta':>7}")
        order = np.argsort(margin_cheap)
        for d in range(10):
            lo = (d * len(order)) // 10
            hi = ((d + 1) * len(order)) // 10
            bucket = order[lo:hi]
            if bucket.size == 0:
                continue
            m_lo = float(margin_cheap[bucket[0]])
            m_hi = float(margin_cheap[bucket[-1]])
            nc = float(np.mean(ndcg_cheap[bucket]))
            nh = float(np.mean(ndcg_high[bucket]))
            print(
                f"  {d+1:>7} [{m_lo:>7.3f},{m_hi:>7.3f}] {nc:>16.4f} {nh:>15.4f} {nh-nc:>+7.4f}"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("matrix_paths", nargs="+", type=Path)
    args = p.parse_args()

    for path in args.matrix_paths:
        dataset, series = load_annex_series(path)
        analyze(dataset, series)


if __name__ == "__main__":
    main()
