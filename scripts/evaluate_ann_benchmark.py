#!/usr/bin/env python3
"""Merge ANNex and competitor JSONL results and print a Pareto comparison."""
import argparse
import csv
import datetime
import json
from pathlib import Path


def load(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]


def pareto_front(rows):
    front = []
    for row in sorted(rows, key=lambda item: (-item["recall"], item["p50_ms"])):
        if not front or row["p50_ms"] < front[-1]["p50_ms"]:
            front.append(row)
    return front


def label(row):
    return f'{row.get("lib", row.get("label", "?"))} {row.get("config", "")} ef={row.get("ef", "")}'.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--targets", default="0.87,0.90,0.92,0.94,0.96")
    args = parser.parse_args()
    rows = load(args.bench_dir / "results_annexdb.jsonl") + load(
        args.bench_dir / "results_competitors.jsonl"
    )
    if not rows:
        raise SystemExit("No results found; run the benchmark suite first.")

    fields = ["lib", "config", "ef", "recall", "p50_ms", "p95_ms", "p99_ms", "qps"]
    output = args.bench_dir / "results_all.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = args.bench_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["combined"] = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "rows": len(rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Results: {output} ({len(rows)} rows)\n")
    print(f"{'Engine':<40}  {f'recall@{args.k}':>9}  {'p50 ms':>8}  {'p99 ms':>8}  {'QPS':>7}")
    for row in sorted(pareto_front(rows), key=lambda item: -item["recall"]):
        print(f"{label(row):<40}  {row['recall']:>9.4f}  {row['p50_ms']:>8.3f}  "
              f"{row.get('p99_ms', 0):>8.3f}  {row.get('qps', 0):>7.0f}")

    print(f"\nRecall-matched p50 latency — {args.title}")
    libraries = sorted({row.get("lib", row.get("label", "?")) for row in rows})
    print(f"{'recall≥':>8}" + "".join(f"  {library[:14]:>14}" for library in libraries))
    for target in map(float, args.targets.split(",")):
        cells = []
        for library in libraries:
            matching = [
                row for row in rows
                if row.get("lib", row.get("label", "?")) == library and row["recall"] >= target
            ]
            cells.append(f"{min((row['p50_ms'] for row in matching), default=float('nan')):>14.3f}")
        print(f"{target:>8.2f}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
