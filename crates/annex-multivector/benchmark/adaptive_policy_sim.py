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
import math
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


def simulate_adaptive(cheap_rows, high_rows, threshold_percentile, signal="margin"):
    """Run every query at cheap c. Escalate the fraction of queries whose
    chosen difficulty signal (at cheap c) falls in the bottom
    `threshold_percentile` percent. Signals:
      - 'margin'         : top_minus_second (small = low confidence)
      - 'fde_disagree'   : 1.0 - fde_maxsim_agreement (large = disagreement)
      - 'fde_top_rank'   : position of MaxSim #1 in FDE order (large = MaxSim
                           rescued something FDE buried; disagreement)
      - 'combined'       : margin × (1 - fde_agreement); small = confident
    """
    qids = sorted(set(cheap_rows.keys()) & set(high_rows.keys()))

    def score_for(qid):
        r = cheap_rows[qid]
        if signal == "margin":
            return r.get("top_minus_second", 0.0)
        if signal == "fde_disagree":
            # We escalate when disagreement is HIGH → convert to "small = escalate"
            # by negating so the percentile threshold works uniformly.
            return -(1.0 - r.get("fde_maxsim_agreement", 1.0))
        if signal == "fde_top_rank":
            return -r.get("fde_top_rank_in_fde", 0)
        if signal == "combined":
            margin = r.get("top_minus_second", 0.0)
            disagree = 1.0 - r.get("fde_maxsim_agreement", 1.0)
            # Small margin OR high disagreement → escalate. Invert to match
            # "small value = escalate" convention.
            return margin * (1.0 - disagree * 0.5)
        raise ValueError(signal)

    scores_v = np.asarray([score_for(q) for q in qids])
    threshold = (
        np.percentile(scores_v, threshold_percentile) if threshold_percentile > 0 else -math.inf
    )
    escalated = 0
    ndcgs = []
    latencies = []
    for qid in qids:
        cheap = cheap_rows[qid]
        high = high_rows[qid]
        if score_for(qid) <= threshold:
            # Escalate: pay both the cheap probe AND the full high-c run.
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
        f"  {'policy':<38} {'nDCG':>7} {'p10':>7} {'p50 lat':>9} {'p95 lat':>9} "
        f"{'mean lat':>9} {'escalated':>10}"
    )
    print(
        f"  {'fixed c=250':<38} {fixed_cheap['mean_ndcg']:>7.4f} {fixed_cheap['p10_ndcg']:>7.4f} "
        f"{fixed_cheap['p50_latency']:>9.2f} {fixed_cheap['p95_latency']:>9.2f} "
        f"{fixed_cheap['mean_latency']:>9.2f} {'—':>10}"
    )
    print(
        f"  {'fixed c=1000':<38} {fixed_high['mean_ndcg']:>7.4f} {fixed_high['p10_ndcg']:>7.4f} "
        f"{fixed_high['p50_latency']:>9.2f} {fixed_high['p95_latency']:>9.2f} "
        f"{fixed_high['mean_latency']:>9.2f} {'—':>10}"
    )
    signals = ["margin", "fde_disagree", "fde_top_rank", "combined"]
    have_fde = "fde_maxsim_agreement" in next(iter(cheap.values()), {})
    if not have_fde:
        signals = ["margin"]
    for signal in signals:
        for pct in [10, 20, 30]:
            result = simulate_adaptive(cheap, high, pct, signal=signal)
            label = f"adaptive: bottom {pct}% by {signal}"
            print(
                f"  {label:<38} "
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
        f"  {'oracle (perfect predictor)':<38} "
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
