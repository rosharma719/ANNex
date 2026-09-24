#!/usr/bin/env python3
"""Simulate adaptive-escalation policy on committed head-to-head sweep data.

Given per_query outputs at c=250 (cheap) and c=1000 (high) from the sweep
matrices, simulate an adaptive policy:

  1. Every query runs at c=250 first (~10 ms on FiQA / 9 ms on Scifact / 8 ms on Nfcorpus).
  2. For queries where the c=250 top-vs-2nd margin is below threshold T,
     re-run at c=1000 (~15-25 ms extra).
  3. Report: mean latency vs mean latency of fixed-c policies, plus the
     quality delta each policy achieves.

The policy is a client-side decision using signals the engine already
emits — no protocol change required to try it. If the simulation shows a
meaningful latency win at unchanged quality, ship it as a runtime option.

Usage:
    .venv/bin/python benchmark/adaptive_policy_sim.py \\
        benchmark/reports/headtohead-fiqa-v3-matrix.json \\
        benchmark/reports/headtohead-scifact-v3-matrix.json \\
        benchmark/reports/headtohead-nfcorpus-v3-matrix.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_sweep(path: Path):
    data = json.loads(path.read_text())
    dataset = data.get("dataset", str(path))
    levels = {}
    for name, sys in data["systems"].items():
        if not name.startswith("annex_multivector_c"):
            continue
        c = int(name.rsplit("_c", 1)[1])
        levels[c] = {row["qid"]: row for row in sys.get("per_query", [])}
    return dataset, levels


def simulate_adaptive(cheap_rows, high_rows, threshold_percentile):
    """Run every query at cheap c. Escalate the fraction of queries whose
    top_minus_second margin at cheap c falls in the bottom `threshold_percentile`
    percent. Return per-query nDCG + total latency ms."""
    qids = sorted(set(cheap_rows.keys()) & set(high_rows.keys()))
    margins = np.asarray([cheap_rows[q]["top_minus_second"] for q in qids])
    # Threshold = value at the given percentile of margins. Queries with margin
    # <= threshold are escalated. So threshold_percentile=20 escalates the
    # bottom 20%.
    threshold = np.percentile(margins, threshold_percentile) if threshold_percentile > 0 else -1.0
    escalated = 0
    ndcgs = []
    latencies = []
    for qid in qids:
        cheap = cheap_rows[qid]
        high = high_rows[qid]
        if cheap["top_minus_second"] <= threshold:
            # Escalate: pay both the cheap probe AND the full high-c run.
            # (In a real implementation we'd cache the cheap-c neighbors.)
            escalated += 1
            ndcgs.append(high["ndcg@10"])
            latencies.append(cheap["latency_ms"] + high["latency_ms"])
        else:
            ndcgs.append(cheap["ndcg@10"])
            latencies.append(cheap["latency_ms"])
    return {
        "n": len(qids),
        "escalated": escalated,
        "escalated_frac": escalated / len(qids) if qids else 0.0,
        "mean_ndcg": float(np.mean(ndcgs)),
        "p10_ndcg": float(np.percentile(ndcgs, 10)),
        "p50_latency": float(np.percentile(latencies, 50)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "mean_latency": float(np.mean(latencies)),
    }


def simulate_fixed(rows):
    qids = sorted(rows.keys())
    ndcgs = [rows[q]["ndcg@10"] for q in qids]
    latencies = [rows[q]["latency_ms"] for q in qids]
    return {
        "n": len(qids),
        "mean_ndcg": float(np.mean(ndcgs)),
        "p10_ndcg": float(np.percentile(ndcgs, 10)),
        "p50_latency": float(np.percentile(latencies, 50)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "mean_latency": float(np.mean(latencies)),
    }


def report_corpus(dataset, levels):
    print(f"\n### {dataset}")
    if 250 not in levels or 1000 not in levels:
        print("  need c=250 and c=1000, skipping")
        return
    cheap = levels[250]
    high = levels[1000]

    fixed_cheap = simulate_fixed(cheap)
    fixed_high = simulate_fixed(high)

    print(
        f"  {'policy':<28} {'nDCG':>7} {'p10':>7} {'p50 lat':>9} {'p95 lat':>9} "
        f"{'mean lat':>9} {'escalated':>10}"
    )
    print(
        f"  {'fixed c=250':<28} {fixed_cheap['mean_ndcg']:>7.4f} {fixed_cheap['p10_ndcg']:>7.4f} "
        f"{fixed_cheap['p50_latency']:>9.2f} {fixed_cheap['p95_latency']:>9.2f} "
        f"{fixed_cheap['mean_latency']:>9.2f} {'—':>10}"
    )
    print(
        f"  {'fixed c=1000':<28} {fixed_high['mean_ndcg']:>7.4f} {fixed_high['p10_ndcg']:>7.4f} "
        f"{fixed_high['p50_latency']:>9.2f} {fixed_high['p95_latency']:>9.2f} "
        f"{fixed_high['mean_latency']:>9.2f} {'—':>10}"
    )
    for pct in [5, 10, 20, 30, 50]:
        result = simulate_adaptive(cheap, high, pct)
        print(
            f"  {'adaptive: bottom '+str(pct)+'% by margin':<28} "
            f"{result['mean_ndcg']:>7.4f} {result['p10_ndcg']:>7.4f} "
            f"{result['p50_latency']:>9.2f} {result['p95_latency']:>9.2f} "
            f"{result['mean_latency']:>9.2f} {result['escalated_frac']:>10.1%}"
        )

    # Oracle upper bound: escalate only queries where c=1000 actually beat c=250.
    # This is what a perfect predictor would achieve.
    qids = sorted(set(cheap.keys()) & set(high.keys()))
    latencies_oracle = []
    ndcgs_oracle = []
    for qid in qids:
        c = cheap[qid]
        h = high[qid]
        if h["ndcg@10"] > c["ndcg@10"] + 1e-6:
            latencies_oracle.append(c["latency_ms"] + h["latency_ms"])
            ndcgs_oracle.append(h["ndcg@10"])
        else:
            latencies_oracle.append(c["latency_ms"])
            ndcgs_oracle.append(c["ndcg@10"])
    print(
        f"  {'oracle (perfect predictor)':<28} "
        f"{float(np.mean(ndcgs_oracle)):>7.4f} {float(np.percentile(ndcgs_oracle, 10)):>7.4f} "
        f"{float(np.percentile(latencies_oracle, 50)):>9.2f} "
        f"{float(np.percentile(latencies_oracle, 95)):>9.2f} "
        f"{float(np.mean(latencies_oracle)):>9.2f}  "
        f"{sum(1 for h, c in zip(ndcgs_oracle, [cheap[q]['ndcg@10'] for q in qids]) if h > c + 1e-6) / len(qids):>10.1%}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("matrix_paths", nargs="+", type=Path)
    args = p.parse_args()

    for path in args.matrix_paths:
        dataset, levels = load_sweep(path)
        report_corpus(dataset, levels)


if __name__ == "__main__":
    main()
