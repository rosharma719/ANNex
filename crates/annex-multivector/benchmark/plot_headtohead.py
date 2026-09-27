#!/usr/bin/env python3
"""Plot every named operating point from retained benchmark summaries."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_paths", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark/results/pareto.png")
    )
    args = parser.parse_args()
    figure, axes = plt.subplots(
        1,
        len(args.matrix_paths),
        figsize=(7 * len(args.matrix_paths), 5),
        squeeze=False,
    )
    for axis, path in zip(axes[0], args.matrix_paths):
        matrix = json.loads(path.read_text())
        axis.set_title(
            f"{matrix['dataset']} ({matrix['documents']} docs, {matrix['queries']} queries)"
        )
        axis.set_xlabel("nDCG@10 (higher is better)")
        axis.set_ylabel("p50 elapsed ms (lower is better)")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        unplottable = []
        for name, result in matrix["systems"].items():
            latency = result.get("p50_ms")
            failures = result.get("failed_queries", 0)
            label = name + (f" ({failures} failed)" if failures else "")
            if latency is None or latency <= 0:
                unplottable.append(label)
                continue
            axis.scatter(result["ndcg@10"], latency, label=label)
        if unplottable:
            axis.text(
                0.01,
                0.01,
                "No latency: " + ", ".join(unplottable),
                transform=axis.transAxes,
                fontsize=8,
            )
        axis.legend(fontsize=8)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
