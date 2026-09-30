#!/usr/bin/env python3
"""
Print H&M filtered-recall summary table from bench/hnm/results_annexdb.jsonl.

Usage:
    python3 bench/hnm/evaluate.py [--results bench/hnm/results_annexdb.jsonl]
"""
import argparse, json
from pathlib import Path


def _print_manifest() -> None:
    p = Path("bench/hnm/manifest.json")
    if not p.exists():
        return
    m = json.loads(p.read_text())
    if "annexdb" in m:
        a = m["annexdb"]
        print(f"Provenance: {a.get('timestamp','')}  {a.get('commit','')} ({a.get('branch','')})  {a.get('rust','')}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench/hnm/results_annexdb.jsonl")
    args = ap.parse_args()

    rows = []
    p = Path(args.results)
    if not p.exists():
        print(f"No results at {p}. Run ./bench/hnm/run_annexdb.sh first.")
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))

    if not rows:
        print("No rows parsed.")
        return

    _print_manifest()

    print("═" * 72)
    print("H&M Filtered Cosine Recall  (ANNex, 2048-D, 105K vectors)")
    print("═" * 72)
    print(f"{'ef':>6}  {'recall':>8}  {'p50 recall':>10}  {'p50 ms':>8}  {'p90 ms':>8}  {'p99 ms':>8}  {'QPS':>7}")
    print("─" * 72)
    for r in sorted(rows, key=lambda x: x.get("ef", 0)):
        print(
            f"{r.get('ef',0):>6}  "
            f"{r.get('recall',0):>8.4f}  "
            f"{r.get('recall_p50',0):>10.4f}  "
            f"{r.get('p50_ms',0):>8.3f}  "
            f"{r.get('p90_ms',0):>8.3f}  "
            f"{r.get('p99_ms',0):>8.3f}  "
            f"{r.get('qps',0):>7.0f}"
        )


if __name__ == "__main__":
    main()
