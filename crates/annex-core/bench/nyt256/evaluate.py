#!/usr/bin/env python3
"""
Merge ANNex and competitor results, compute Pareto frontier, print comparison table.

Usage:
    python3 bench/nyt256/evaluate.py \
        --annexdb bench/nyt256/results_annexdb.jsonl \
        --competitors bench/nyt256/results_competitors.jsonl

Output:
    - Pareto frontier table (recall@20 vs p50 latency)
    - Full results CSV: bench/nyt256/results_all.csv
"""
import argparse, json, csv, sys
from pathlib import Path

def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows

def pareto_front(rows, key_recall="recall", key_lat="p50_ms"):
    """Return rows that are not dominated (higher recall AND lower latency)."""
    front = []
    for r in sorted(rows, key=lambda x: (-x[key_recall], x[key_lat])):
        if not front or r[key_lat] < front[-1][key_lat]:
            front.append(r)
    return front

def label(r):
    lib = r.get("lib", r.get("label", "?"))
    cfg = r.get("config", "")
    ef  = r.get("ef", "")
    return f"{lib} {cfg} ef={ef}".strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annexdb",     default="bench/nyt256/results_annexdb.jsonl")
    parser.add_argument("--competitors", default="bench/nyt256/results_competitors.jsonl")
    parser.add_argument("--csv-out",     default="bench/nyt256/results_all.csv")
    args = parser.parse_args()

    all_rows = []
    if Path(args.annexdb).exists():
        all_rows += load(args.annexdb)
    if Path(args.competitors).exists():
        all_rows += load(args.competitors)

    if not all_rows:
        print("No results found. Run run_annexdb.sh and run_competitors.py first.")
        sys.exit(1)

    # Write full CSV
    fields = ["lib", "config", "ef", "recall", "p50_ms", "p95_ms", "p99_ms", "qps"]
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"Full results: {args.csv_out} ({len(all_rows)} rows)\n")

    # Pareto frontier
    front = pareto_front(all_rows)
    print("═" * 70)
    print("PARETO FRONTIER — recall@20 vs p50 latency, NYT-256-Angular")
    print("═" * 70)
    print(f"{'Engine':<40}  {'recall@20':>9}  {'p50 ms':>8}  {'p99 ms':>8}  {'QPS':>7}")
    print("─" * 70)
    for r in sorted(front, key=lambda x: -x["recall"]):
        print(f"{label(r):<40}  {r['recall']:>9.4f}  {r['p50_ms']:>8.3f}  "
              f"{r.get('p99_ms', 0):>8.3f}  {r.get('qps', 0):>7.0f}")

    # Recall-matched comparison at key targets
    print("\n" + "═" * 70)
    print("RECALL-MATCHED COMPARISON (p50 ms to reach each recall target)")
    print("═" * 70)
    targets = [0.87, 0.90, 0.92, 0.94, 0.96]
    libs = sorted(set(r.get("lib", r.get("label", "?")) for r in all_rows))

    print(f"{'recall≥':>8}", end="")
    for lib in libs:
        print(f"  {lib[:14]:>14}", end="")
    print()
    print("─" * 70)

    for tgt in targets:
        print(f"{tgt:>8.2f}", end="")
        for lib in libs:
            lib_rows = sorted(
                [r for r in all_rows if r.get("lib", r.get("label", "?")) == lib and r["recall"] >= tgt],
                key=lambda x: x["p50_ms"]
            )
            if lib_rows:
                best = lib_rows[0]
                print(f"  {best['p50_ms']:>14.3f}", end="")
            else:
                print(f"  {'—':>14}", end="")
        print()

if __name__ == "__main__":
    main()
