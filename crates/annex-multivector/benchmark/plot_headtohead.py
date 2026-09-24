#!/usr/bin/env python3
"""Plot the annex-vs-Qdrant Pareto curve from one or more matrix.json files.

Renders one figure per matrix: quality (nDCG@10) on x, latency (log p50 ms)
on y, one point per system (annex at each candidate count + Qdrant).

Usage:
    .venv/bin/python benchmark/plot_headtohead.py \\
        benchmark/results/headtohead-fiqa-sweep/matrix.json \\
        benchmark/results/headtohead-scifact-sweep/matrix.json \\
        --output benchmark/reports/pareto.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path):
    data = json.loads(path.read_text())
    dataset = data.get("dataset", str(path))
    docs = data.get("documents", "?")
    queries = data.get("queries", "?")
    rows = []
    for name, sys in data["systems"].items():
        rows.append({
            "system": name,
            "ndcg": sys.get("ndcg@10", 0.0),
            "recall": sys.get("recall@10", 0.0),
            "p50": sys.get("p50_ms", 0.0),
            "p95": sys.get("p95_ms", 0.0),
            "build_s": sys.get("build_s", 0.0),
        })
    return {"dataset": dataset, "docs": docs, "queries": queries, "rows": rows}


def annex_points(rows):
    xs, ys, labels = [], [], []
    for r in sorted(rows, key=lambda r: r["p50"]):
        if not r["system"].startswith("annex"):
            continue
        xs.append(r["ndcg"])
        ys.append(r["p50"])
        if "_c" in r["system"]:
            labels.append(f"c={r['system'].rsplit('_c', 1)[1]}")
        else:
            labels.append(r["system"])
    return xs, ys, labels


def other_points(rows, prefix):
    for r in rows:
        if r["system"].startswith(prefix):
            return r["ndcg"], r["p50"]
    return None, None


def plot_matrix(ax, matrix):
    ax.set_title(f"{matrix['dataset']}  ({matrix['docs']} docs, {matrix['queries']} queries)")
    ax.set_xlabel("nDCG@10 (higher is better)")
    ax.set_ylabel("p50 latency (ms, log scale)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    xs, ys, labels = annex_points(matrix["rows"])
    if xs:
        ax.plot(xs, ys, marker="o", linestyle="-", color="#2b8cbe", label="annex-multivector (sweep)")
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")

    q_ndcg, q_p50 = other_points(matrix["rows"], "qdrant")
    if q_ndcg is not None:
        ax.plot([q_ndcg], [q_p50], marker="s", markersize=10, color="#d7301f", label="Qdrant (native MaxSim)")
        ax.annotate("qdrant", (q_ndcg, q_p50), fontsize=9, xytext=(6, 3), textcoords="offset points")

    l_ndcg, l_p50 = other_points(matrix["rows"], "lancedb")
    if l_ndcg is not None:
        ax.plot([l_ndcg], [l_p50], marker="^", markersize=8, color="#888", alpha=0.5,
                label="LanceDB (mean-pool, not MaxSim)")

    # Down-and-left is better. Annotate that.
    ax.text(0.02, 0.98, "↙ better (faster, higher quality)",
            transform=ax.transAxes, fontsize=8, va="top", ha="left", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("matrix_paths", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=Path("benchmark/reports/pareto.png"))
    args = p.parse_args()

    matrices = [load(path) for path in args.matrix_paths]
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, m in zip(axes[0], matrices):
        plot_matrix(ax, m)
    fig.suptitle("Multi-vector retrieval Pareto — annex vs Qdrant on identical M2 + ColBERTv2 inputs")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
